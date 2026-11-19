"""Behavior-cloning baseline for square-nut assembly.

Uses the exact same dataloader (utils/dataset.py's RobomimicImageDataset)
and network backbone (networks/'s DiT1d + MultiImageObsCondition, with
identical hyperparameters) as diffusion_policy.py -- the only difference is
how the shared DiT1d backbone is used: instead of denoising a noisy action
over many diffusion steps, BCPolicy calls it once with a dummy all-zeros
action and a fixed dummy timestep, and reads the output directly as the
predicted action (direct regression, MSE loss).

This works because DiT1d routes all conditioning (timestep + observation
embedding) through one combined embedding used for adaLN-Zero modulation at
every transformer block; the "noisy action" input only ever contributes a
position-dependent term. So even with that input zeroed out, the network's
output is still architecturally forced to come from the real observation
condition, via the same pathway the diffusion model uses (see
networks/dit1d.py's docstring).

Unlike diffusion_policy.py, BC predicts and executes a single action per
observation -- no action chunking. It replans every env step.
"""

import argparse
import os
import time
from copy import deepcopy

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from networks import DiT1d, MultiImageObsCondition
from utils.common import (
    CSVLogger, check_runtime, dataloader_kwargs, eval_step_logged, format_duration,
    make_single_env, mark_training_complete, print_run_banner, report_eval_progress,
    report_train_progress, report_parameters, resume_progress, run_label, set_seed,
    training_is_complete,
)
from utils.dataset import RobomimicImageDataset
from utils.dataset_utils import loop_dataloader

ROOT = os.path.dirname(os.path.abspath(__file__))

SHAPE_META = {
    "action": {"shape": [7]},  # delta pos(3) + delta axis-angle rot(3) + gripper(1)
    "obs": {
        "agentview_image": {"shape": [3, 84, 84], "type": "rgb"},
        "robot0_eef_pos": {"shape": [3], "type": "low_dim"},
        "robot0_eef_quat": {"shape": [4], "type": "low_dim"},
        "robot0_eye_in_hand_image": {"shape": [3, 84, 84], "type": "rgb"},
        "robot0_gripper_qpos": {"shape": [2], "type": "low_dim"},
    },
}


class BCPolicy:
    """Direct-regression policy sharing DDPM's bookkeeping shape (ModuleDict
    of backbone + condition net, an EMA shadow copy, AdamW, save/load in the
    same checkpoint format) for pedagogical symmetry with ddpm.py -- but
    without any noise schedule or sampling loop, since there's nothing to
    denoise. Always predicts a single action (no action chunking); the
    backbone is called with a length-1 dummy sequence."""

    def __init__(self, backbone: nn.Module, nn_condition: nn.Module, action_dim: int,
                 ema_rate: float = 0.995, grad_clip_norm: float = None, optim_params: dict = None,
                 clip_action: bool = True, device="cpu"):
        optim_params = optim_params or {"lr": 2e-4, "weight_decay": 1e-5}
        self.device = device
        self.action_dim = action_dim
        self.ema_rate = ema_rate
        self.grad_clip_norm = grad_clip_norm
        self.clip_action = clip_action

        self.model = nn.ModuleDict({"backbone": backbone.to(device), "condition": nn_condition.to(device)})
        self.model_ema = deepcopy(self.model).requires_grad_(False)
        self.model.train()
        self.model_ema.eval()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), **optim_params)

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
        }, path)

    def load(self, path: str):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model"])
        self.model_ema.load_state_dict(checkpoint["model_ema"])
        if "optimizer" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        return checkpoint.get("step"), float(checkpoint.get("elapsed", 0.0))

    def loss(self, action_gt: torch.Tensor, condition):
        condition_vec = self.model["condition"](condition)
        b = action_gt.shape[0]
        dummy_x = torch.zeros(b, 1, self.action_dim, device=self.device)
        dummy_t = torch.zeros(b, dtype=torch.long, device=self.device)
        pred = self.model["backbone"](dummy_x, dummy_t, condition_vec)
        return ((pred - action_gt) ** 2).mean()

    def update(self, action_gt, condition, update_ema=True):
        loss = self.loss(action_gt, condition)
        loss.backward()
        grad_norm = (
            nn.utils.clip_grad_norm_(self.model.parameters(), self.grad_clip_norm) if self.grad_clip_norm else None
        )
        self.optimizer.step()
        self.optimizer.zero_grad()
        if update_ema:
            self.ema_update()
        return {"loss": loss.item(), "grad_norm": grad_norm}

    @torch.no_grad()
    def predict(self, condition, use_ema: bool = True):
        model = self.model_ema if use_ema else self.model
        condition_vec = model["condition"](condition)
        b = next(iter(condition.values())).shape[0]
        dummy_x = torch.zeros(b, 1, self.action_dim, device=self.device)
        dummy_t = torch.zeros(b, dtype=torch.long, device=self.device)
        pred = model["backbone"](dummy_x, dummy_t, condition_vec)
        if self.clip_action:
            pred = pred.clamp(-1.0, 1.0)
        return pred


def build_arg_parser():
    p = argparse.ArgumentParser(description=__doc__)
    # env / task
    p.add_argument("--env_name", default="square")
    p.add_argument("--env_type", default="ph")
    p.add_argument("--abs_action", type=bool, default=False)
    p.add_argument("--dataset_path", default=os.path.join(
        ROOT, "data/robomimic/datasets/square/ph/image.hdf5"))
    p.add_argument("--max_episode_steps", type=int, default=500)
    p.add_argument("--render_obs_key", default="agentview_image")
    p.add_argument("--action_dim", type=int, default=7)
    # observation encoder -- identical to diffusion_policy.py's, so the
    # backbones being compared are genuinely the same architecture
    p.add_argument("--rgb_model", default="resnet18")
    p.add_argument("--crop_shape", type=int, nargs=2, default=(76, 76))
    p.add_argument("--random_crop", type=bool, default=True)
    p.add_argument("--use_group_norm", type=bool, default=True)
    # training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--gradient_steps", type=int, default=100000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    # eval / logging
    p.add_argument("--eval_episodes", type=int, default=25)
    p.add_argument("--eval_freq", type=int, default=10000)
    p.add_argument("--log_freq", type=int, default=50)
    p.add_argument("--save_freq", type=int, default=500)
    p.add_argument("--results_dir", default=os.path.join(ROOT, "results/bc"))
    # mode
    p.add_argument("--mode", choices=["train", "inference"], default="train")
    p.add_argument("--model_path", default=None)
    return p


def build_networks(args):
    nn_condition = MultiImageObsCondition(
        shape_meta=SHAPE_META, emb_dim=256, rgb_model_name=args.rgb_model,
        crop_shape=tuple(args.crop_shape), random_crop=args.random_crop,
        use_group_norm=args.use_group_norm)
    backbone = DiT1d(args.action_dim, emb_dim=256, d_model=320, n_heads=10, depth=2)
    return nn_condition, backbone


def evaluate(args, env, dataset, policy, use_ema=True):
    rewards, successes = [], []
    for ep in range(args.eval_episodes):
        obs = env.reset()
        ep_reward, t, done = 0.0, 0, False
        report_eval_progress(ep + 1, args.eval_episodes, 0, args.max_episode_steps)
        while not done and t < args.max_episode_steps:
            condition = {}
            for k, v in obs.items():
                nobs = dataset.normalizer["obs"][k].normalize(v.astype(np.float32))[None]
                condition[k] = torch.tensor(nobs, device=args.device, dtype=torch.float32)

            naction = policy.predict(condition, use_ema=use_ema)
            naction = naction.detach().cpu().numpy()
            action_pred = dataset.normalizer["action"].unnormalize(naction)
            if args.abs_action:
                action_pred = dataset.undo_transform_action(action_pred)
            action = action_pred[0, 0]

            obs, reward, done, info = env.step(action)
            ep_reward += reward
            t += 1
            if t % 100 == 0:
                report_eval_progress(ep + 1, args.eval_episodes, t, args.max_episode_steps)

        rewards.append(ep_reward)
        successes.append(1.0 if ep_reward > 0 else 0.0)
        print(
            f"{run_label('behavior cloning')} | eval episode {ep + 1} / {args.eval_episodes} finished"
            f" | reward {ep_reward:.2f} | success {successes[-1]:.0f}",
            flush=True,
        )

    mean_reward, mean_success = float(np.mean(rewards)), float(np.mean(successes))
    print(f"Mean reward: {mean_reward:.3f} | Mean success: {mean_success:.3f}", flush=True)
    return {"mean_reward": mean_reward, "mean_success": mean_success}


def main():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    if args.mode == "train" and training_is_complete(args.results_dir):
        print(
            f"{run_label('behavior cloning')} | this run already finished. "
            f"Found {os.path.join(args.results_dir, 'training_complete.txt')}. Nothing to train.",
            flush=True,
        )
        return
    check_runtime(args.dataset_path)
    logger = CSVLogger(args.results_dir)

    # horizon=1, pad_after=0: no action chunking -- each sample is a single
    # (observation, action) pair.
    dataset = RobomimicImageDataset(
        args.dataset_path, shape_meta=SHAPE_META, horizon=1,
        pad_before=0, pad_after=0, abs_action=args.abs_action)
    print(dataset)

    nn_condition, backbone = build_networks(args)
    report_parameters(backbone)
    report_parameters(nn_condition)

    policy = BCPolicy(
        backbone=backbone, nn_condition=nn_condition, action_dim=args.action_dim,
        optim_params={"lr": args.lr}, device=args.device)

    if args.mode == "inference":
        assert args.model_path, "--model_path is required for --mode inference"
        policy.load(args.model_path)
        policy.eval()
        env = make_single_env(args, SHAPE_META)
        evaluate(args, env, dataset, policy)
        return

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, **dataloader_kwargs())
    env = make_single_env(args, SHAPE_META)

    run_name = "behavior cloning, one action per step"
    latest_path = os.path.join(logger.model_dir, "model_latest.pt")
    n_gradient_step, elapsed_before = resume_progress(policy, latest_path)
    print_run_banner(
        run_name, args.gradient_steps, args.device, args.log_freq, args.save_freq,
        args.eval_freq, args.eval_episodes, args.max_episode_steps, args.results_dir,
        n_gradient_step, elapsed_before,
    )
    session_start = time.time()

    def elapsed_now():
        return elapsed_before + (time.time() - session_start)

    def save_latest(step):
        print(
            f"{run_label(run_name)} | saving checkpoint at step {step}. "
            "If the run stops, start the same command again and it continues from this step.",
            flush=True,
        )
        logger.save_agent(policy, identifier="latest", step=step, elapsed=elapsed_now())

    def run_robot_eval(step, final=False):
        print(
            f"{run_label(run_name)} | {'final ' if final else ''}robot eval at step {step}",
            flush=True,
        )
        eval_started = time.time()
        policy.eval()
        metrics = evaluate(args, env, dataset, policy)
        logger.log({"step": step, **metrics}, category="eval", echo=False)
        follow = "this run is continuing." if not final else "this was the last eval of this run."
        print(
            f"{run_label(run_name)} | eval finished in {format_duration(time.time() - eval_started)}"
            f" | mean success {metrics['mean_success']:.3f}"
            f" | mean reward {metrics['mean_reward']:.3f}\n"
            f"{run_label(run_name)} | {follow}",
            flush=True,
        )
        policy.train()

    finished_step = n_gradient_step - 1
    if finished_step > 0 and finished_step % args.eval_freq == 0 and not eval_step_logged(args.results_dir, finished_step):
        print(
            f"{run_label(run_name)} | checkpoint at step {finished_step} was saved, "
            "but its robot eval was not. Running that eval now, then continuing.",
            flush=True,
        )
        run_robot_eval(finished_step)

    losses = []
    if n_gradient_step <= args.gradient_steps:
        for batch in loop_dataloader(dataloader):
            condition = {k: v.to(args.device) for k, v in batch["obs"].items()}
            naction = batch["action"].to(args.device)

            loss = policy.update(naction, condition)["loss"]
            losses.append(loss)

            if n_gradient_step % args.log_freq == 0:
                elapsed = elapsed_now()
                loss_mean = float(np.mean(losses))
                progress = report_train_progress(
                    n_gradient_step, args.gradient_steps, loss_mean, elapsed,
                    run_name, args.eval_freq, args.save_freq)
                logger.log({
                    "step": n_gradient_step, "loss": loss_mean, "elapsed": elapsed,
                    "percent": progress["percent"], "eta_seconds": progress["eta_seconds"],
                }, category="train", echo=False)
                losses = []

            if n_gradient_step % args.save_freq == 0 and n_gradient_step > 0:
                save_latest(n_gradient_step)

            if n_gradient_step % args.eval_freq == 0 and n_gradient_step > 0:
                run_robot_eval(n_gradient_step)

            n_gradient_step += 1
            if n_gradient_step > args.gradient_steps:
                break
    else:
        print(
            f"{run_label(run_name)} | weight updates already finished. Moving to the final save and eval.",
            flush=True,
        )

    last_completed = n_gradient_step - 1
    print(f"{run_label(run_name)} | training loop finished. Saving the final model and running the last eval.", flush=True)
    save_latest(last_completed)
    logger.save_agent(policy, identifier="final", step=last_completed, elapsed=elapsed_now())
    if eval_step_logged(args.results_dir, n_gradient_step):
        print(f"{run_label(run_name)} | final eval was already saved. Not running it again.", flush=True)
    else:
        run_robot_eval(n_gradient_step, final=True)
    mark_training_complete(args.results_dir, last_completed)
    print(
        f"{run_label(run_name)} | this run is finished through gradient step {last_completed}. "
        "Starting it again will skip it.",
        flush=True,
    )


if __name__ == "__main__":
    main()
