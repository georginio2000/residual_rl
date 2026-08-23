"""Generate report figures for the Square residual-RL experiment (Thread B).

Reads the metrics.csv files written by last_millimeter.train / the BC-RNN
training run and produces PNG figures comparing the frozen baseline against
the residual and gated attempts. Run with the project's own venv:

    .venv/bin/python scripts/reporting/make_figures.py
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = REPO_ROOT / "runs"
OUT_DIR = REPO_ROOT / "docs" / "figures"
FROZEN_BASELINE = 0.85

RUN_LABELS = {
    "robomimic_square_residual": "Residual (always-on)",
    "robomimic_square_gated_lambda0p01": "Gated (bias=0.5, λ_gate=0.01)",
    "robomimic_square_gated": "Gated (bias=0.1 fixed, λ_gate=0.02)",
    "robomimic_square_triggered": "Triggered (heuristic gate)",
}
RUN_COLORS = {
    "robomimic_square_residual": "#d97757",
    "robomimic_square_gated_lambda0p01": "#6a8caf",
    "robomimic_square_gated": "#4c9a6b",
    "robomimic_square_triggered": "#9b6dd6",
}


def load_episode_rows(run_dir: Path) -> list[dict[str, str]]:
    metrics_path = run_dir / "metrics.csv"
    if not metrics_path.exists():
        return []
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        return [row for row in reader if row["phase"] == "train_episode"]


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return np.full_like(values, np.nan, dtype=float)
    kernel = np.ones(window) / window
    smoothed = np.convolve(values, kernel, mode="valid")
    pad = np.full(window - 1, np.nan)
    return np.concatenate([pad, smoothed])


def plot_success_rate_comparison() -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5))
    window = 12
    for run_name, label in RUN_LABELS.items():
        rows = load_episode_rows(RUNS / run_name)
        if not rows:
            continue
        success = np.array([float(r["success_rate"]) for r in rows])
        episodes = np.arange(len(success))
        smoothed = rolling_mean(success, window)
        ax.plot(episodes, smoothed, label=label, color=RUN_COLORS[run_name], linewidth=2)

    ax.axhline(FROZEN_BASELINE, color="#333333", linestyle="--", linewidth=1.5, label="Frozen baseline (85%)")
    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Success rate ({window}-episode rolling mean)")
    ax.set_title("NutAssemblySquare: residual RL attempts vs. frozen BC-RNN baseline")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "success_rate_comparison.png", dpi=160)
    plt.close(fig)


def plot_gate_trend() -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    window = 12
    gated_runs = [
        "robomimic_square_gated_lambda0p01",
        "robomimic_square_gated",
        "robomimic_square_triggered",
    ]
    for run_name in gated_runs:
        rows = load_episode_rows(RUNS / run_name)
        if not rows:
            continue
        gate = np.array([float(r["gate"]) for r in rows])
        episodes = np.arange(len(gate))
        smoothed = rolling_mean(gate, window)
        ax.plot(episodes, smoothed, label=RUN_LABELS[run_name], color=RUN_COLORS[run_name], linewidth=2)

    ax.axhline(0.5, color="#999999", linestyle=":", linewidth=1, label="Initial bias = 0.5 (old)")
    ax.axhline(0.1, color="#333333", linestyle=":", linewidth=1, label="Initial bias = 0.1 (fixed)")
    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Gate value / trigger-active fraction ({window}-episode rolling mean)")
    ax.set_title("Intervention strength over training: learned gate vs. heuristic trigger")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gate_trend.png", dpi=160)
    plt.close(fig)


def plot_bc_rnn_baseline_curve() -> None:
    # Hand-transcribed from the robomimic BC-RNN training rollout evaluations
    # (runs/robomimic_train/.../trained_models/.../logs/log.txt); robomimic
    # writes per-checkpoint rollout stats to that log, not a flat CSV.
    epochs = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
    success = [0.0, 0.60, 0.75, 0.85, 0.60, 0.65, 0.85, 0.70, 0.85, 0.55]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, success, marker="o", color="#4c6a9a", linewidth=2)
    best_epoch, best_success = 800, 0.85
    ax.scatter([best_epoch], [best_success], color="#d97757", zorder=5, s=70, label="Checkpoint used (epoch 800)")
    ax.set_xlabel("Training epoch")
    ax.set_ylabel("Rollout success rate (20 episodes)")
    ax.set_title("BC-RNN base policy training (Square, PH, low-dim)")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "bc_rnn_baseline_training.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_success_rate_comparison()
    plot_gate_trend()
    plot_bc_rnn_baseline_curve()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
