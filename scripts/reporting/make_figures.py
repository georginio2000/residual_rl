"""Generate report figures for the Square residual-RL experiment (Thread B).

Reads the metrics.csv files written by last_millimeter.train / the BC-RNN
training run and produces PNG figures comparing the frozen baseline against
the residual and gated attempts. Run with the project's own venv:

    .venv/bin/python scripts/reporting/make_figures.py
"""

from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS = REPO_ROOT / "runs"
RESULTS = REPO_ROOT / "results"
OUT_DIR = REPO_ROOT / "docs" / "figures"
FROZEN_BASELINE = 0.85

EPOCH_BLOCK_RE = re.compile(r"(Train|Validation) Epoch (\d+)\n(\{.*?\n\})", re.DOTALL)
ROLLOUT_RE = re.compile(
    r"Epoch (\d+) Rollouts took.*?\n(\{.*?\n\})", re.DOTALL
)


def parse_bc_rnn_log(log_path: Path) -> dict[str, list]:
    """Parse robomimic's log.txt for per-epoch train/val loss and rollout success."""
    text = log_path.read_text(errors="replace")

    train_epochs, train_loss = [], []
    val_epochs, val_loss = [], []
    for kind, epoch_str, blob in EPOCH_BLOCK_RE.findall(text):
        epoch = int(epoch_str)
        loss = json.loads(blob)["Loss"]
        if kind == "Train":
            train_epochs.append(epoch)
            train_loss.append(loss)
        else:
            val_epochs.append(epoch)
            val_loss.append(loss)

    rollout_epochs, rollout_success = [], []
    for epoch_str, blob in ROLLOUT_RE.findall(text):
        rollout_epochs.append(int(epoch_str))
        rollout_success.append(json.loads(blob)["Success_Rate"])

    return {
        "train_epochs": train_epochs,
        "train_loss": train_loss,
        "val_epochs": val_epochs,
        "val_loss": val_loss,
        "rollout_epochs": rollout_epochs,
        "rollout_success": rollout_success,
    }

RUN_LABELS = {
    "robomimic_square_residual": "Residual (always-on)",
    "robomimic_square_gated_lambda0p01": "Gated (bias=0.5, λ_gate=0.01)",
    "robomimic_square_gated": "Gated (bias=0.1 fixed, λ_gate=0.02)",
    "robomimic_square_triggered_precriticfix": "Triggered (pre critic-fix)",
    "robomimic_square_triggered_50k_criticfix": "Triggered (critic-fix, 50k)",
    "robomimic_square_triggered": "Triggered (critic-fix, 100k extended)",
}
RUN_COLORS = {
    "robomimic_square_residual": "#d97757",
    "robomimic_square_gated_lambda0p01": "#6a8caf",
    "robomimic_square_gated": "#4c9a6b",
    "robomimic_square_triggered_precriticfix": "#c9a8ea",
    "robomimic_square_triggered_50k_criticfix": "#7a4fb5",
    "robomimic_square_triggered": "#9b6dd6",
}


HISTORICAL_RUNS = {
    "robomimic_square_residual",
    "robomimic_square_gated_lambda0p01",
    "robomimic_square_gated",
    "robomimic_square_triggered_precriticfix",
}
PRIMARY_RUNS = {
    "robomimic_square_triggered_50k_criticfix",
    "robomimic_square_triggered",
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

    # Historical attempts (always-on residual, both gated variants, TRIGGERED
    # pre critic-fix): thin, gray, one shared legend entry -- context for how
    # the project arrived at TRIGGERED + critic-fix, not a competing result.
    historical_labeled = False
    for run_name in HISTORICAL_RUNS:
        rows = load_episode_rows(RESULTS / run_name)
        if not rows:
            continue
        success = np.array([float(r["success_rate"]) for r in rows])
        episodes = np.arange(len(success))
        smoothed = rolling_mean(success, window)
        ax.plot(
            episodes,
            smoothed,
            color="#b5b5b5",
            linewidth=1,
            alpha=0.7,
            label="Earlier attempts (historical: residual / gated / pre-fix)" if not historical_labeled else None,
        )
        historical_labeled = True

    # Primary result: TRIGGERED mode, critic-fix lineage (50k + 100k extension).
    for run_name in ["robomimic_square_triggered_50k_criticfix", "robomimic_square_triggered"]:
        rows = load_episode_rows(RESULTS / run_name)
        if not rows:
            continue
        success = np.array([float(r["success_rate"]) for r in rows])
        episodes = np.arange(len(success))
        smoothed = rolling_mean(success, window)
        ax.plot(episodes, smoothed, label=RUN_LABELS[run_name], color=RUN_COLORS[run_name], linewidth=2.2)

    ax.axhline(FROZEN_BASELINE, color="#333333", linestyle="--", linewidth=1.5, label="Frozen baseline (85%)")
    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Success rate ({window}-episode rolling mean)")
    ax.set_title("NutAssemblySquare: TRIGGERED critic-fix vs. frozen BC-RNN baseline")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "success_rate_comparison.png", dpi=160)
    plt.close(fig)


def plot_gate_trend() -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    window = 12

    historical_gated = ["robomimic_square_gated_lambda0p01", "robomimic_square_gated"]
    historical_labeled = False
    for run_name in historical_gated:
        rows = load_episode_rows(RESULTS / run_name)
        if not rows:
            continue
        gate = np.array([float(r["gate"]) for r in rows])
        episodes = np.arange(len(gate))
        smoothed = rolling_mean(gate, window)
        ax.plot(
            episodes,
            smoothed,
            color="#b5b5b5",
            linewidth=1,
            alpha=0.7,
            label="Earlier attempts (historical: learned gate)" if not historical_labeled else None,
        )
        historical_labeled = True

    # Primary result: TRIGGERED mode's heuristic trigger-active fraction,
    # critic-fix lineage (pre-fix run omitted -- same heuristic, not a
    # different result worth foregrounding here).
    for run_name in ["robomimic_square_triggered_50k_criticfix", "robomimic_square_triggered"]:
        rows = load_episode_rows(RESULTS / run_name)
        if not rows:
            continue
        gate = np.array([float(r["gate"]) for r in rows])
        episodes = np.arange(len(gate))
        smoothed = rolling_mean(gate, window)
        ax.plot(episodes, smoothed, label=RUN_LABELS[run_name], color=RUN_COLORS[run_name], linewidth=2.2)

    ax.set_xlabel("Training episode")
    ax.set_ylabel(f"Gate value / trigger-active fraction ({window}-episode rolling mean)")
    ax.set_title("Intervention strength over training: learned gate (historical) vs. heuristic trigger")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "gate_trend.png", dpi=160)
    plt.close(fig)


def plot_bc_rnn_baseline_curve() -> None:
    # Parsed directly from results/bc_rnn_baseline/log.txt (robomimic writes
    # per-checkpoint rollout stats there, not to a flat CSV).
    log_path = RESULTS / "bc_rnn_baseline" / "log.txt"
    data = parse_bc_rnn_log(log_path)
    epochs = data["rollout_epochs"]
    success = data["rollout_success"]

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


def plot_bc_rnn_loss_curves() -> None:
    # Parsed directly from results/bc_rnn_baseline/log.txt: robomimic reports
    # BC-RNN "Loss" as the negative log-likelihood under the GMM action head,
    # which can go negative on the training set once the model is confident
    # (density > 1), so the train panel stays on a linear scale. Validation
    # loss never goes negative but is dominated by rare high-density-mismatch
    # batches and spans orders of magnitude, so it needs a log scale. It drops
    # sharply and bottoms out into a broad, noisy plateau starting around
    # epoch 200-800, then drifts gently upward for the rest of training -- a
    # mild-overfitting signal, but too broad/noisy to pick a single best
    # epoch from on its own. That is why epoch 800 was chosen from the
    # rollout success curve above (first epoch to hit the run's peak 85%)
    # rather than from this loss curve directly.
    log_path = RESULTS / "bc_rnn_baseline" / "log.txt"
    data = parse_bc_rnn_log(log_path)

    fig, (ax_train, ax_val) = plt.subplots(2, 1, figsize=(8, 7.5), sharex=True)

    ax_train.plot(data["train_epochs"], data["train_loss"], color="#4c6a9a", linewidth=1, alpha=0.85)
    ax_train.axvline(800, color="#d97757", linestyle="--", linewidth=1.5, label="Checkpoint used (epoch 800)")
    ax_train.set_ylabel("Train loss (NLL)")
    ax_train.set_title("BC-RNN training loss (Square, PH, low-dim)")
    ax_train.legend(loc="upper right", fontsize=9)
    ax_train.grid(alpha=0.25)

    ax_val.plot(data["val_epochs"], data["val_loss"], color="#9b6dd6", linewidth=1, alpha=0.85)
    ax_val.axvline(800, color="#d97757", linestyle="--", linewidth=1.5)
    ax_val.set_yscale("log")
    ax_val.set_xlabel("Training epoch")
    ax_val.set_ylabel("Validation loss (NLL, log scale)")
    ax_val.grid(alpha=0.25, which="both")

    fig.tight_layout()
    fig.savefig(OUT_DIR / "bc_rnn_loss_curves.png", dpi=160)
    plt.close(fig)


def load_train_episode_arrays(run_name: str) -> dict[str, np.ndarray]:
    rows = load_episode_rows(RESULTS / run_name)
    return {
        "step": np.array([int(r["step"]) for r in rows], dtype=float),
        "critic_loss": np.array([float(r["critic_loss"]) if r["critic_loss"] else np.nan for r in rows]),
        "actor_loss": np.array([float(r["actor_loss"]) if r["actor_loss"] else np.nan for r in rows]),
        "alpha": np.array([float(r["alpha"]) if r["alpha"] else np.nan for r in rows]),
    }


def plot_sac_loss_curves() -> None:
    # Scoped to the TRIGGERED critic-fix lineage only (50k run + its 100k
    # extension) -- the pre-fix run trained the critic/actor on the wrong
    # action quantity (see README) and isn't a meaningful comparison here.
    # Critic loss is plotted on a log scale: it rises during early learning
    # as the critic's Q-scale grows from near-zero, then settles into a
    # noisy but bounded band (no runaway growth) -- occasional spikes are
    # visible in the raw trace but the 10-episode rolling mean is flat past
    # roughly step 50,000. Actor loss climbs steadily throughout both runs
    # without saturating; since success rate stays flat over the same range
    # (see extended_training_curve.png), this reads as numerical drift in
    # the critic's value scale rather than the policy itself getting worse.
    fig, (ax_c, ax_a) = plt.subplots(2, 1, figsize=(9, 7), sharex=True)
    window = 10
    for run_name in ["robomimic_square_triggered_50k_criticfix", "robomimic_square_triggered"]:
        data = load_train_episode_arrays(run_name)
        color = RUN_COLORS[run_name]
        label = RUN_LABELS[run_name]

        ax_c.plot(data["step"], data["critic_loss"], color=color, alpha=0.25, linewidth=1)
        smoothed = rolling_mean(data["critic_loss"], window)
        ax_c.plot(data["step"], smoothed, color=color, linewidth=2, label=label)

        ax_a.plot(data["step"], data["actor_loss"], color=color, alpha=0.25, linewidth=1)
        smoothed = rolling_mean(data["actor_loss"], window)
        ax_a.plot(data["step"], smoothed, color=color, linewidth=2, label=label)

    ax_c.set_yscale("log")
    ax_c.set_ylabel("Critic loss (log scale)")
    ax_c.set_title(f"SAC critic/actor loss, TRIGGERED critic-fix lineage ({window}-episode rolling mean; raw in light)")
    ax_c.legend(loc="upper left", fontsize=9)
    ax_c.grid(alpha=0.25, which="both")

    ax_a.set_ylabel("Actor loss")
    ax_a.set_xlabel("Training step")
    ax_a.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(OUT_DIR / "loss_curves.png", dpi=160)
    plt.close(fig)


def plot_alpha_curve() -> None:
    # Every committed config (see configs/robomimic/*.yaml) fixes
    # automatic_entropy_tuning: false, alpha: 0.1 -- this curve is flat by
    # construction. It exists to make that explicit: the historical unbounded
    # -entropy-temperature bug (LOG_ALPHA_MIN/MAX in rl/sac.py) never applied
    # to any run in this report, since auto-tuning was off throughout.
    fig, ax = plt.subplots(figsize=(8, 3.5))
    for run_name in ["robomimic_square_triggered_50k_criticfix", "robomimic_square_triggered"]:
        data = load_train_episode_arrays(run_name)
        ax.plot(data["step"], data["alpha"], color=RUN_COLORS[run_name], linewidth=2, label=RUN_LABELS[run_name])
    ax.set_ylim(0, 0.2)
    ax.set_xlabel("Training step")
    ax.set_ylabel("SAC entropy temperature (alpha)")
    ax.set_title("Entropy temperature: fixed at 0.1, not auto-tuned")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "alpha_curve.png", dpi=160)
    plt.close(fig)


def plot_speed_comparison() -> None:
    # From a direct apples-to-apples comparison: both policies evaluated
    # through the same trigger-instrumented bridge (frozen: 30 episodes,
    # seed 10000; trained TRIGGERED final.pt with the critic/actor trigger-
    # masking fix: 30 episodes, seed 10000), restricted to successful
    # episodes only. See the README's "Does RLT deliver a speed win" section
    # for the exact commands.
    labels = ["Total episode\n(successes only)", "Critical phase\n(trigger-active steps)"]
    frozen = [153.3, 40.7]
    trained = [141.6, 33.1]

    x = np.arange(len(labels))
    width = 0.32
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(x - width / 2, frozen, width, label="Frozen baseline (76.7% success, n=23/30)", color="#6a8caf")
    ax.bar(x + width / 2, trained, width, label="Trained TRIGGERED, critic-fix (83.3% success, n=25/30)", color="#9b6dd6")
    for i, (f, t) in enumerate(zip(frozen, trained)):
        ax.text(i - width / 2, f + 3, f"{f:.0f}", ha="center", fontsize=9)
        ax.text(i + width / 2, t + 3, f"{t:.0f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Mean control steps")
    ax.set_title("Speed on successful episodes: frozen vs. trained (both faster and more successful)")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "speed_comparison.png", dpi=160)
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    plot_success_rate_comparison()
    plot_gate_trend()
    plot_bc_rnn_baseline_curve()
    plot_bc_rnn_loss_curves()
    plot_sac_loss_curves()
    plot_alpha_curve()
    plot_speed_comparison()
    print(f"Wrote figures to {OUT_DIR}")


if __name__ == "__main__":
    main()
