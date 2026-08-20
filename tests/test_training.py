import csv
import json
from pathlib import Path

from last_millimeter.config import (
    AgentConfig,
    ExperimentConfig,
    ProjectConfig,
    TrainingConfig,
)
from last_millimeter.train import run_training


def test_frozen_evaluation_writes_reproducible_outputs(tmp_path: Path) -> None:
    output_dir = tmp_path / "frozen"
    config = ProjectConfig(
        experiment=ExperimentConfig(
            name="test_frozen", mode="frozen", output_dir=str(output_dir)
        ),
        training=TrainingConfig(total_steps=0, batch_size=8, evaluation_episodes=3),
    )

    summary = run_training(config)

    assert summary["mode"] == "frozen"
    assert (output_dir / "config.yaml").is_file()
    assert (output_dir / "metrics.csv").is_file()
    assert (output_dir / "summary.json").is_file()


def test_short_residual_training_run(tmp_path: Path) -> None:
    output_dir = tmp_path / "residual"
    config = ProjectConfig(
        experiment=ExperimentConfig(
            name="test_residual", mode="residual", output_dir=str(output_dir)
        ),
        agent=AgentConfig(hidden_dims=[16, 16]),
        training=TrainingConfig(
            total_steps=24,
            replay_capacity=100,
            batch_size=8,
            start_steps=4,
            evaluation_interval=12,
            evaluation_episodes=2,
            checkpoint_interval=0,
        ),
    )

    summary = run_training(config)

    assert summary["steps"] == 24
    assert summary["updates"] > 0
    assert (output_dir / "checkpoints" / "final.pt").is_file()
    with (output_dir / "summary.json").open(encoding="utf-8") as handle:
        saved_summary = json.load(handle)
    assert saved_summary["mode"] == "residual"
    with (output_dir / "metrics.csv").open(encoding="utf-8") as handle:
        phases = {row["phase"] for row in csv.DictReader(handle)}
    assert "evaluation" in phases

