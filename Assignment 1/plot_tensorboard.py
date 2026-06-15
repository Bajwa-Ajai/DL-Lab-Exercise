"""
plot_tensorboard.py

Reads TensorBoard event files and produces report-quality plots
using matplotlib / seaborn.

Usage:
    python plot_tensorboard.py --logdir ./tensorboard --out ./plots

The script auto-discovers all runs inside --logdir and groups them
by the tag names it finds (e.g. train_loss, train_accuracy, val_accuracy
for IL; train/episode_reward, eval/mean_episode_reward for DQN).

Requirements:
    pip install tensorboard matplotlib seaborn pandas
"""

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Style ──────────────────────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", font_scale=1.2)
PALETTE = sns.color_palette("tab10")

# ── Helpers ────────────────────────────────────────────────────────────────────


def load_run(run_dir: str) -> dict[str, pd.DataFrame]:
    """
    Load all scalar tags from a single TensorBoard run directory.
    Returns {tag: DataFrame(step, value)}.
    """
    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])

    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        df = pd.DataFrame(
            {"step": [e.step for e in events], "value": [e.value for e in events]}
        )
        data[tag] = df
    return data


def discover_runs(logdir: str) -> dict[str, dict]:
    """
    Walk logdir and collect all runs.
    Returns {run_name: {tag: DataFrame}}.
    """
    runs = {}
    logdir = Path(logdir)

    # A run directory contains at least one tfevents file
    for path in sorted(logdir.rglob("events.out.tfevents.*")):
        run_dir = str(path.parent)
        run_name = path.parent.name
        try:
            data = load_run(run_dir)
            if data:
                runs[run_name] = data
                print(f"  Loaded run '{run_name}' — tags: {list(data.keys())}")
        except Exception as e:
            print(f"  Warning: could not load {run_dir}: {e}")

    return runs


def smooth(values: pd.Series, window: int = 10) -> pd.Series:
    """Simple rolling mean smoothing."""
    return values.rolling(window=window, min_periods=1).mean()


# ── Plot functions ─────────────────────────────────────────────────────────────


def plot_il_learning_curves(runs: dict, out_dir: str):
    """
    Imitation Learning plot:
    Left axis  — train_loss
    Right axis — train_accuracy + val_accuracy
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        "Imitation Learning — Training Progress", fontsize=14, fontweight="bold"
    )

    for idx, (run_name, data) in enumerate(runs.items()):
        color = PALETTE[idx % len(PALETTE)]

        # ── Loss plot ──────────────────────────────────────────────────────────
        if "train_loss" in data:
            df = data["train_loss"]
            ax1.plot(
                df["step"],
                smooth(df["value"]),
                color=color,
                label=run_name,
                linewidth=2,
            )
            ax1.plot(df["step"], df["value"], color=color, alpha=0.15, linewidth=0.8)

        # ── Accuracy plot ──────────────────────────────────────────────────────
        if "train_accuracy" in data:
            df = data["train_accuracy"]
            ax2.plot(
                df["step"],
                smooth(df["value"]),
                color=color,
                label=f"{run_name} — train",
                linewidth=2,
                linestyle="-",
            )
            ax2.plot(df["step"], df["value"], color=color, alpha=0.15, linewidth=0.8)

        if "val_accuracy" in data:
            df = data["val_accuracy"]
            ax2.plot(
                df["step"],
                smooth(df["value"]),
                color=color,
                label=f"{run_name} — val",
                linewidth=2,
                linestyle="--",
            )
            ax2.plot(df["step"], df["value"], color=color, alpha=0.15, linewidth=0.8)

    ax1.set_title("Cross-Entropy Loss")
    ax1.set_xlabel("Minibatch")
    ax1.set_ylabel("Loss")
    ax1.legend(fontsize=9)

    ax2.set_title("Classification Accuracy")
    ax2.set_xlabel("Minibatch")
    ax2.set_ylabel("Accuracy")
    ax2.set_ylim(0, 1)
    ax2.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
    ax2.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(out_dir, "il_learning_curves.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


def plot_dqn_learning_curves(runs: dict, out_dir: str, title: str = "DQN"):
    """
    DQN plot:
    Top    — train episode reward (with smoothing)
    Bottom — eval mean episode reward
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=False)
    fig.suptitle(f"{title} — Training Progress", fontsize=14, fontweight="bold")

    for idx, (run_name, data) in enumerate(runs.items()):
        color = PALETTE[idx % len(PALETTE)]

        # ── Train reward ───────────────────────────────────────────────────────
        train_key = next(
            (k for k in data if "episode_reward" in k and "eval" not in k), None
        )
        if train_key:
            df = data[train_key]
            ax1.plot(
                df["step"],
                smooth(df["value"], window=20),
                color=color,
                label=run_name,
                linewidth=2,
            )
            ax1.plot(df["step"], df["value"], color=color, alpha=0.12, linewidth=0.7)

        # ── Eval reward ────────────────────────────────────────────────────────
        eval_key = next((k for k in data if "eval" in k and "reward" in k), None)
        if eval_key:
            df = data[eval_key]
            ax2.plot(
                df["step"],
                df["value"],
                color=color,
                label=run_name,
                linewidth=2,
                marker="o",
                markersize=4,
            )

    ax1.set_title("Train Episode Reward (smoothed)")
    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.legend(fontsize=9)

    ax2.set_title("Eval Mean Episode Reward (greedy, 5 episodes)")
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Mean Reward")
    ax2.legend(fontsize=9)

    # Reference line for CartPole solved threshold
    if "CartPole" in title or "cartpole" in title.lower():
        ax2.axhline(
            490, color="red", linestyle="--", linewidth=1.2, label="Solved (490)"
        )
        ax2.legend(fontsize=9)

    plt.tight_layout()
    out_path = os.path.join(
        out_dir, f"{title.lower().replace(' ', '_')}_learning_curves.png"
    )
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {out_path}")
    plt.close()


def plot_action_distribution(runs: dict, out_dir: str, title: str = "DQN"):
    """
    Stacked area plot of action usage over training episodes.
    """
    action_tags = {
        "straight": "STRAIGHT",
        "left": "LEFT",
        "right": "RIGHT",
        "accel": "ACCEL",
        "brake": "BRAKE",
    }

    for run_name, data in runs.items():
        found = {k: v for k, v in action_tags.items() if any(k in tag for tag in data)}
        if not found:
            continue

        fig, ax = plt.subplots(figsize=(12, 5))
        fig.suptitle(
            f"{title} — Action Distribution during Training\n({run_name})",
            fontsize=13,
            fontweight="bold",
        )

        steps = None
        series = {}
        for tag_key, label in action_tags.items():
            match = next((t for t in data if tag_key in t), None)
            if match:
                df = data[match]
                if steps is None:
                    steps = df["step"].values
                series[label] = smooth(df["value"], window=20).values

        if series and steps is not None:
            df_plot = pd.DataFrame(series, index=steps)
            df_plot.plot.area(ax=ax, alpha=0.7, colormap="tab10")
            ax.set_xlabel("Episode")
            ax.set_ylabel("Fraction of Actions")
            ax.set_ylim(0, 1)
            ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))
            ax.legend(loc="upper right", fontsize=9)

        plt.tight_layout()
        safe_name = run_name.replace(" ", "_").replace("/", "_")
        out_path = os.path.join(out_dir, f"{title.lower()}_action_dist_{safe_name}.png")
        plt.savefig(out_path, dpi=150, bbox_inches="tight")
        print(f"  Saved: {out_path}")
        plt.close()


# ── Main ───────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="TensorBoard → matplotlib plots")
    parser.add_argument(
        "--logdir",
        type=str,
        default="./tensorboard",
        help="Root tensorboard log directory",
    )
    parser.add_argument(
        "--out", type=str, default="./plots", help="Output directory for saved plots"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="auto",
        choices=["auto", "il", "cartpole", "carracing"],
        help="Which plots to generate. 'auto' detects from tag names.",
    )
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print(f"\nScanning: {args.logdir}")
    runs = discover_runs(args.logdir)

    if not runs:
        print("No TensorBoard event files found. Check --logdir path.")
        return

    # Auto-detect mode from tag names
    all_tags = {tag for data in runs.values() for tag in data}
    print(f"\nAll tags found: {all_tags}\n")

    mode = args.mode
    if mode == "auto":
        if "train_loss" in all_tags:
            mode = "il"
        elif any("carracing" in t.lower() or "accel" in t for t in all_tags):
            mode = "carracing"
        else:
            mode = "cartpole"
        print(f"Auto-detected mode: {mode}")

    print(f"\nGenerating plots → {args.out}")

    if mode == "il":
        plot_il_learning_curves(runs, args.out)

    elif mode == "cartpole":
        plot_dqn_learning_curves(runs, args.out, title="CartPole DQN")

    elif mode == "carracing":
        plot_dqn_learning_curves(runs, args.out, title="CarRacing DQN")
        plot_action_distribution(runs, args.out, title="CarRacing")

    print("\nDone.")


if __name__ == "__main__":
    main()
