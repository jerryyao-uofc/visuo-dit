"""VisuoDiT: Visuomotor Diffusion Transformer for Square-Nut Assembly.

A ResNet-18 pair encodes two cameras and the arm state. A small diffusion
transformer denoises a 10-step action chunk. See bc.py for the direct-regression
baseline, which shares the dataset and the network backbone.
"""

import argparse
import os
import time

import numpy as np
import torch
from torch.utils.data import DataLoader

from ddpm import DDPM
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
    p.add_argument("--action_steps", type=int, default=8)
    p.add_argument("--horizon", type=int, default=10)
    # observation encoder
    p.add_argument("--rgb_model", default="resnet18")
    p.add_argument("--crop_shape", type=int, nargs=2, default=(76, 76))
    p.add_argument("--random_crop", type=bool, default=True)
    p.add_argument("--use_group_norm", type=bool, default=True)
    # diffusion
    p.add_argument("--diffusion_steps", type=int, default=50)
    p.add_argument("--beta_schedule", choices=["cosine", "linear", "sigmoid"], default="cosine",
                    help="Noise schedule. cosine is the main run. sigmoid is the schedule comparison.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--predict_x0", action="store_true",
                    help="Predict the clean action x0 directly instead of the noise epsilon.")
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
    p.add_argument("--results_dir", default=None,
                    help="Defaults to results/diffusion_policy_x0 or results/diffusion_policy_eps. "
                         "A sigmoid run defaults to a folder whose name ends in _sigmoid_noise_schedule.")
    # mode
    p.add_argument("--mode", choices=["train", "inference"], default="train")
    p.add_argument("--model_path", default=None)
    return p


def build_networks(args):
    nn_condition = MultiImageObsCondition(
        shape_meta=SHAPE_META, emb_dim=256, rgb_model_name=args.rgb_model,
        crop_shape=tuple(args.crop_shape), random_crop=args.random_crop,
        use_group_norm=args.use_group_norm)
    nn_diffusion = DiT1d(args.action_dim, emb_dim=256, d_model=320, n_heads=10, depth=2)
    return nn_condition, nn_diffusion


def checkpoint_id(predict_x0, beta_schedule):
    """Filename tag for this run. A sigmoid run keeps a suffix so it cannot overwrite the cosine checkpoints."""
    mode_tag = "x0" if predict_x0 else "eps"
    if beta_schedule == "sigmoid":
        return f"{mode_tag}_sigmoid_noise_schedule"
    return mode_tag


def evaluate(args, env, dataset, agent, use_ema=True):
    rewards, successes = [], []
    for ep in range(args.eval_episodes):
        obs = env.reset()
        ep_reward, t, done = 0.0, 0, False
        report_eval_progress(ep + 1, args.eval_episodes, 0, args.max_episode_steps)
        last_hundred = 0
        while not done and t < args.max_episode_steps:
            condition = {}
            for k, v in obs.items():
                nobs = dataset.normalizer["obs"][k].normalize(v.astype(np.float32))[None]
                condition[k] = torch.tensor(nobs, device=args.device, dtype=torch.float32)

            prior = torch.zeros((1, args.horizon, args.action_dim), device=args.device)
            naction = agent.sample(
                prior=prior, n_samples=1, sample_steps=args.diffusion_steps,
                use_ema=use_ema, temperature=args.temperature, condition=condition)
            naction = naction.detach().cpu().numpy()
            action_pred = dataset.normalizer["action"].unnormalize(naction)

            if args.abs_action:
                action_pred = dataset.undo_transform_action(action_pred)
            action_chunk = action_pred[0]

            # Execute the chunk one real env step at a time, then replan.
            # The sparse reward stays at 1 for every step the nut is on the peg,
            # so the chunk score is the max reward inside the chunk. One placement
            # then counts once.
            chunk_reward = 0.0
            for action in action_chunk[: args.action_steps]:
                obs, reward, done, info = env.step(action)
                chunk_reward = max(chunk_reward, float(reward))
                t += 1
                if done or t >= args.max_episode_steps:
                    break
            ep_reward += chunk_reward
            hundred = t // 100
            if hundred > last_hundred:
                last_hundred = hundred
                report_eval_progress(ep + 1, args.eval_episodes, t, args.max_episode_steps)

        rewards.append(ep_reward)
        successes.append(1.0 if ep_reward > 0 else 0.0)
        print(
            f"{run_label('diffusion policy')} | eval episode {ep + 1} / {args.eval_episodes} finished"
            f" | reward {ep_reward:.2f} | success {successes[-1]:.0f}",
            flush=True,
        )

    mean_reward, mean_success = float(np.mean(rewards)), float(np.mean(successes))
    print(f"Mean reward: {mean_reward:.3f} | Mean success: {mean_success:.3f}", flush=True)
    return {"mean_reward": mean_reward, "mean_success": mean_success}


def main():
    args = build_arg_parser().parse_args()
    set_seed(args.seed)
    file_tag = checkpoint_id(args.predict_x0, args.beta_schedule)
    if args.results_dir is None:
        args.results_dir = os.path.join(ROOT, "results", f"diffusion_policy_{file_tag}")
    if args.mode == "train" and training_is_complete(args.results_dir):
        print(
            f"{run_label('diffusion policy')} | this run already finished. "
            f"Found {os.path.join(args.results_dir, 'training_complete.txt')}. Nothing to train.",
            flush=True,
        )
        return
    check_runtime(args.dataset_path)
    logger = CSVLogger(args.results_dir)

    dataset = RobomimicImageDataset(
        args.dataset_path, shape_meta=SHAPE_META, horizon=args.horizon,
        pad_before=0, pad_after=args.action_steps - 1, abs_action=args.abs_action)
    print(dataset)

    nn_condition, nn_diffusion = build_networks(args)
    report_parameters(nn_diffusion)
    report_parameters(nn_condition)

    agent = DDPM(
        nn_diffusion=nn_diffusion, nn_condition=nn_condition, diffusion_steps=args.diffusion_steps,
        beta_schedule=args.beta_schedule,
        predict_noise=not args.predict_x0, optim_params={"lr": args.lr}, device=args.device)

    if args.mode == "inference":
        assert args.model_path, "--model_path is required for --mode inference"
        agent.load(args.model_path)
        agent.eval()
        env = make_single_env(args, SHAPE_META)
        evaluate(args, env, dataset, agent)
        return

    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True, **dataloader_kwargs())
    env = make_single_env(args, SHAPE_META)

    run_name = "diffusion policy, clean-action prediction" if args.predict_x0 else "diffusion policy, noise prediction"
    if args.beta_schedule == "sigmoid":
        run_name += ", sigmoid noise schedule"
    latest_path = os.path.join(logger.model_dir, f"model_{file_tag}_latest.pt")
    n_gradient_step, elapsed_before = resume_progress(agent, latest_path)
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
        logger.save_agent(agent, identifier=f"{file_tag}_latest", step=step, elapsed=elapsed_now())

    def run_robot_eval(step, final=False):
        print(
            f"{run_label(run_name)} | {'final ' if final else ''}robot eval at step {step}",
            flush=True,
        )
        eval_started = time.time()
        agent.eval()
        metrics = evaluate(args, env, dataset, agent)
        logger.log({"step": step, **metrics}, category="eval", echo=False)
        follow = "this run is continuing." if not final else "this was the last eval of this run."
        print(
            f"{run_label(run_name)} | eval finished in {format_duration(time.time() - eval_started)}"
            f" | mean success {metrics['mean_success']:.3f}"
            f" | mean reward {metrics['mean_reward']:.3f}\n"
            f"{run_label(run_name)} | {follow}",
            flush=True,
        )
        agent.train()

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

            loss = agent.update(naction, condition)["loss"]
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
    logger.save_agent(agent, identifier=f"{file_tag}_final", step=last_completed, elapsed=elapsed_now())
    if eval_step_logged(args.results_dir, n_gradient_step):
        print(f"{run_label(run_name)} | final eval was already saved. Not running it again.", flush=True)
    else:
        run_robot_eval(n_gradient_step, final=True)
    note = "sigmoid noise schedule" if args.beta_schedule == "sigmoid" else ""
    mark_training_complete(args.results_dir, last_completed, note=note)
    print(
        f"{run_label(run_name)} | this run is finished through gradient step {last_completed}. "
        "Starting it again will skip it.",
        flush=True,
    )


if __name__ == "__main__":
    main()
