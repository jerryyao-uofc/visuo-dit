"""Plots eval success rate vs. gradient steps for every run found under
results/. Auto-detects any results/<name>/ directory containing
eval_metrics.csv (step, mean_success), and labels each line with <name>.

    python plot_completion_vs_gradient_steps.py
    python plot_completion_vs_gradient_steps.py --results_dir /path/to/results
"""

import argparse
import csv
import os

import matplotlib.pyplot as plt
import numpy as np

from utils.common import check_runtime

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# Fixed-order categorical palette (dataviz skill reference palette, light mode).
CATEGORICAL_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
    "#e34948",  # red
]

SURFACE = "#fcfcfb"
PRIMARY_INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED_INK = "#898781"
GRIDLINE = "#e1e0d9"
BASELINE = "#c3c2b7"


def read_csv_columns(path, *fields):
    # A results_dir that was reused across a crashed + restarted run has the
    # header re-appended mid-file; only the last such block is current data.
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    header_positions = [i for i, row in enumerate(rows) if row == rows[0]]
    rows = rows[header_positions[-1]:]
    header, data_rows = rows[0], rows[1:]
    field_idx = {field: header.index(field) for field in fields}
    columns = {f: [float(row[field_idx[f]]) for row in data_rows] for f in fields}
    return {f: np.array(v) for f, v in columns.items()}


def find_runs(results_dir):
    runs = []
    if not os.path.isdir(results_dir):
        return runs
    for name in sorted(os.listdir(results_dir)):
        eval_csv = os.path.join(results_dir, name, "eval_metrics.csv")
        if os.path.isfile(eval_csv):
            runs.append((name, eval_csv))
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results_dir", default=RESULTS_DIR,
        help="Folder that contains the diffusion_policy_eps, diffusion_policy_x0, and bc runs.",
    )
    parser.add_argument(
        "--output", default=None,
        help="Plot path. Default: <results_dir>/completion_vs_gradient_steps.png.",
    )
    args = parser.parse_args()
    check_runtime()
    runs = find_runs(args.results_dir)
    output_path = args.output or os.path.join(args.results_dir, "completion_vs_gradient_steps.png")
    if not runs:
        print(f"No runs with eval_metrics.csv found under {args.results_dir}/")
        return

    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    for i, (name, eval_csv) in enumerate(runs):
        eval_ = read_csv_columns(eval_csv, "step", "mean_success")
        order = np.argsort(eval_["step"])
        color = CATEGORICAL_COLORS[i % len(CATEGORICAL_COLORS)]
        ax.plot(
            eval_["step"][order], eval_["mean_success"][order],
            label=name, color=color, linewidth=2, marker="o", markersize=6,
        )

    ax.set_xlabel("Gradient steps", color=SECONDARY_INK)
    ax.set_ylabel("Eval mean success", color=SECONDARY_INK)
    ax.set_title("Completion vs. gradient steps", color=PRIMARY_INK, fontsize=13, loc="left")

    ax.grid(True, color=GRIDLINE, linewidth=1)
    ax.set_axisbelow(True)
    for spine_name, spine in ax.spines.items():
        if spine_name in ("left", "bottom"):
            spine.set_color(BASELINE)
        else:
            spine.set_visible(False)
    ax.tick_params(colors=MUTED_INK)

    ax.legend(frameon=False, labelcolor=SECONDARY_INK)

    fig.tight_layout()
    os.makedirs(args.results_dir, exist_ok=True)
    fig.savefig(output_path, dpi=150, facecolor=SURFACE)
    print(f"Saved plot with {len(runs)} run(s) to {output_path}")


if __name__ == "__main__":
    main()
