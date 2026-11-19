"""Discrete-time DDPM for short action chunks.

Trains a noise predictor, or a clean-action predictor, with the simplified
mean-squared-error objective. Sampling is ancestral: start from Gaussian noise
and walk back for `diffusion_steps` steps. Actions are scaled to [-1, 1], and
the clean-action estimate is clipped to that range at every step.
"""

import warnings
from copy import deepcopy

import torch
import torch.nn as nn

from utils.common import (
    at_least_ndim, cosine_beta_schedule, linear_beta_schedule, sigmoid_beta_schedule,
)


class DDPM:
    def __init__(
        self,
        nn_diffusion: nn.Module,
        nn_condition: nn.Module,
        diffusion_steps: int = 50,
        ema_rate: float = 0.995,
        grad_clip_norm: float = None,
        optim_params: dict = None,
        beta_schedule: str = "cosine",
        beta_schedule_params: dict = None,
        clip_action: bool = True,
        predict_noise: bool = True,
        device="cpu",
    ):
        optim_params = optim_params or {"lr": 2e-4, "weight_decay": 1e-5}
        self.device = device
        self.grad_clip_norm = grad_clip_norm
        self.diffusion_steps = diffusion_steps
        self.ema_rate = ema_rate
        self.clip_action = clip_action
        self.predict_noise = predict_noise
        self.beta_schedule = beta_schedule

        self.model = nn.ModuleDict({"diffusion": nn_diffusion.to(device), "condition": nn_condition.to(device)})
        self.model_ema = deepcopy(self.model).requires_grad_(False)
        self.model.train()
        self.model_ema.eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **optim_params)

        beta_schedule_params = dict(beta_schedule_params or {})
        beta_schedule_params["T"] = diffusion_steps
        schedules = {
            "linear": linear_beta_schedule,
            "cosine": cosine_beta_schedule,
            "sigmoid": sigmoid_beta_schedule,
        }
        if beta_schedule not in schedules:
            raise ValueError(f"Unknown beta_schedule {beta_schedule!r}. Choose from {sorted(schedules)}.")
        beta = schedules[beta_schedule](**beta_schedule_params)
        self.beta = torch.tensor(beta, device=device, dtype=torch.float32)
        self.alpha = 1 - self.beta
        self.bar_alpha = torch.cumprod(self.alpha.clone(), 0)
        print(
            f"Noise schedule: {beta_schedule}. "
            f"Clean-signal fraction at step 0: {float(self.bar_alpha[0]):.4f}, "
            f"at the last step: {float(self.bar_alpha[-1]):.4e}",
            flush=True,
        )

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def ema_update(self):
        with torch.no_grad():
            for p, p_ema in zip(self.model.parameters(), self.model_ema.parameters()):
                p_ema.data.mul_(self.ema_rate).add_(p.data, alpha=1.0 - self.ema_rate)

    def save(self, path: str, step=None, elapsed: float = 0.0):
        torch.save({
            "model": self.model.state_dict(),
            "model_ema": self.model_ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": step,
            "elapsed": elapsed,
            "beta_schedule": self.beta_schedule,
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model_ema.load_state_dict(checkpoint["model_ema"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        saved_schedule = checkpoint.get("beta_schedule")
        if saved_schedule is not None and saved_schedule != self.beta_schedule:
            print(
                f"Checkpoint was trained with the {saved_schedule} noise schedule, "
                f"but this model is using {self.beta_schedule}.",
                flush=True,
            )
        return checkpoint.get("step"), float(checkpoint.get("elapsed", 0.0))

    # -------------------------------------------------------------------
    # Training

    def add_noise(self, x0, t=None, eps=None):
        """Jump straight to a noisy chunk at step t.

        Returns xt, t, and the Gaussian noise eps that was added.
        eps does not depend on t. The schedule's bar_alpha already says how
        much noise belongs at that step.
        """
        # Randomly samples a point along the reverse-diffusion process t
        t = torch.randint(self.diffusion_steps, (x0.shape[0],), device=self.device) if t is None else t
        # Samples the alpha (time-dependent variance set by the scheduler)
        bar_alpha = at_least_ndim(self.bar_alpha[t], x0.dim())

        # One Gaussian draw jumps straight to time t:
        # x_t = sqrt(ᾱ_t) x_0 + sqrt(1 - ᾱ_t) ε, with ε ~ N(0, I).
        eps = torch.randn_like(x0) if eps is None else eps
        xt = bar_alpha.sqrt() * x0 + (1.0 - bar_alpha).sqrt() * eps
        return xt, t, eps

    def loss(self, x0, condition=None):
        """Mean squared error on the noise, or on the clean chunk.

        x0 has shape (batch, chunk length, action dim). predict_noise selects
        the target: the added noise, or the clean chunk.
        """

        # generates the condition vector (in our case, the embeddings from video)
        condition_vec = self.model["condition"](condition) if condition is not None else None

        xt, t, eps = self.add_noise(x0)
        pred = self.model["diffusion"](xt, t, condition_vec)
        # ε-prediction regresses the noise we added. x0-prediction regresses the clean action.
        target = eps if self.predict_noise else x0
        return ((pred - target) ** 2).mean()

    def update(self, x0, condition=None, update_ema=True):
        loss = self.loss(x0, condition)
        loss.backward()
        grad_norm = (
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) if self.grad_clip_norm else None
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        if update_ema:
            self.ema_update()
        return {"loss": loss.item(), "grad_norm": grad_norm}

    # -------------------------------------------------------------------
    # Sampling

    @torch.no_grad()
    def sample(self, 
               prior: torch.Tensor, 
               n_samples: int = 1, 
               sample_steps: int = None,
               use_ema: bool = True, 
               temperature: float = 1.0, 
               condition=None,
               record_sampling = False):
        """Ancestral DDPM reverse-diffusion sampler. `prior` sets the sample
        shape (n_samples, horizon, action_dim); its values are only used
        where the (unused, in this repo) fix_mask would apply, so here it
        just fixes the shape/device/dtype."""
        if sample_steps is not None and sample_steps != self.diffusion_steps:
            warnings.warn("sample_steps != diffusion_steps; DDPM's ancestral sampler always runs the full loop.")
        model = self.model_ema if use_ema else self.model

        condition_vec = model["condition"](condition) if condition is not None else None

        xt = torch.randn_like(prior, device=self.device) * temperature

        if record_sampling:
            sample_record = []

        for t in range(self.diffusion_steps - 1, -1, -1):
            if record_sampling:
                sample_record.append(xt)
            t_batch = torch.tensor(t, device=self.device, dtype=torch.long).repeat(n_samples)
            bar_alpha = self.bar_alpha[t]
            bar_alpha_prev = self.bar_alpha[t - 1] if t > 0 else torch.tensor(1.0, device=self.device)
            alpha = self.alpha[t]
            beta = self.beta[t]

            pred = model["diffusion"](xt, t_batch, condition_vec)

            # Turn either prediction into a clean-action estimate, then take one
            # ancestral step. Actions live in [-1, 1].
            if self.predict_noise:
                x0_pred = (xt - (1.0 - bar_alpha).sqrt() * pred) / bar_alpha.sqrt()
            else:
                x0_pred = pred
            if self.clip_action:
                x0_pred = x0_pred.clamp(-1.0, 1.0)

            xt = (
                alpha.sqrt() * (1.0 - bar_alpha_prev) * xt
                + beta * bar_alpha_prev.sqrt() * x0_pred
            ) / (1.0 - bar_alpha)
            if t > 0:
                sigma = (beta * (1.0 - bar_alpha_prev) / (1.0 - bar_alpha)).sqrt()
                xt = xt + sigma * torch.randn_like(xt)

        if record_sampling:
            sample_record.append(xt)
            return xt, sample_record
        
        return xt
