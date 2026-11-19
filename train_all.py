"""Train the three main policies one after another.

1. Diffusion policy, noise prediction          (diffusion_policy.py)
2. Diffusion policy, clean-action prediction   (diffusion_policy.py --predict_x0)
3. Behavior cloning, one action per step       (bc.py)

    python train_all.py
    python train_all.py --results_root /path/to/results
"""

import argparse
import os
import subprocess
import sys
import time

from utils.common import format_duration, training_is_complete

ROOT = os.path.dirname(os.path.abspath(__file__))

RUNS = [
    {
        "name": "diffusion policy, noise prediction",
        "script": "diffusion_policy.py",
        "extra": [],
        "folder": "diffusion_policy_eps",
    },
    {
        "name": "diffusion policy, clean-action prediction",
        "script": "diffusion_policy.py",
        "extra": ["--predict_x0"],
        "folder": "diffusion_policy_x0",
    },
    {
        "name": "behavior cloning, one action per step",
        "script": "bc.py",
        "extra": [],
        "folder": "bc",
    },
]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_root", default=None,
        help="Folder for all three runs. Default: each script's own results/ folder.",
    )
    parser.add_argument(
        "--dataset_path", default=None,
        help="Demonstration file. Default: each script's own data/robomimic path.",
    )
    args = parser.parse_args()

    print("Three trainings will run, one after another:", flush=True)
    for index, run in enumerate(RUNS, start=1):
        print(f"  {index}. {run['name']}", flush=True)
    print(
        "Each run trains for 100000 gradient steps, saves a checkpoint every 500 steps,\n"
        "and evaluates the robot every 10000 steps. A progress line prints every 50 steps.\n"
        "If the run stops, start this same command again.\n"
        "A finished run is skipped. A run that stopped in the middle continues from its last checkpoint.",
        flush=True,
    )

    started = time.time()
    finished_this_session = 0
    for index, run in enumerate(RUNS, start=1):
        if args.results_root:
            results_dir = os.path.join(args.results_root, run["folder"])
        else:
            results_dir = os.path.join(ROOT, "results", run["folder"])
        if training_is_complete(results_dir):
            print(
                f"\nTraining {index} of 3 already finished: {run['name']}. "
                f"Skipping {results_dir}.",
                flush=True,
            )
            continue
        cmd = [sys.executable, run["script"], *run["extra"]]
        if args.results_root:
            cmd.extend(["--results_dir", results_dir])
        if args.dataset_path:
            cmd.extend(["--dataset_path", args.dataset_path])
        env = os.environ.copy()
        env["TRAIN_RUN_LABEL"] = f"Training {index} of 3: {run['name']}"
        env["PYTHONUNBUFFERED"] = "1"

        print("\n" + "=" * 72, flush=True)
        print(f"Starting training {index} of 3: {run['name']}", flush=True)
        print("Command: " + " ".join(cmd), flush=True)
        print("=" * 72 + "\n", flush=True)

        result = subprocess.run(cmd, cwd=ROOT, env=env)
        if result.returncode != 0:
            print(
                f"Training {index} of 3 stopped with exit code {result.returncode}. "
                "The remaining trainings were not started.",
                flush=True,
            )
            raise SystemExit(result.returncode)

        finished_this_session += 1
        elapsed = time.time() - started
        runs_left = len(RUNS) - index
        print("\n" + "=" * 72, flush=True)
        print(f"Finished training {index} of 3: {run['name']}", flush=True)
        print(f"Time so far in this session: {format_duration(elapsed)}", flush=True)
        if runs_left:
            print(
                f"About {format_duration(elapsed / finished_this_session * runs_left)} left"
                f" for the remaining {runs_left} training(s), if they take about as long.",
                flush=True,
            )
        print("=" * 72, flush=True)

    print("\nAll 3 trainings finished.", flush=True)
    print("Make the completion plot and the two denoising videos with:", flush=True)
    print("  python plot_completion_vs_gradient_steps.py", flush=True)
    print("  python plot_diffusion_generation.py", flush=True)
    print("  python plot_diffusion_generation.py --predict_x0", flush=True)
    if args.results_root:
        print("The video scripts need --results_dir pointing at each diffusion folder:", flush=True)
        print(f"  --results_dir {os.path.join(args.results_root, 'diffusion_policy_eps')}", flush=True)
        print(f"  --results_dir {os.path.join(args.results_root, 'diffusion_policy_x0')}", flush=True)


if __name__ == "__main__":
    main()
