"""Lightweight CSV/JSON experiment output without extra dependencies."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


METRIC_FIELDS = (
    "step",
    "phase",
    "episode",
    "success_rate",
    "task_return",
    "regularized_return",
    "final_distance",
    "episode_length",
    "correction_norm",
    "gate",
    "critic_loss",
    "actor_loss",
    "alpha_loss",
    "alpha",
    "mean_log_prob",
)


class MetricWriter:
    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(self._handle, fieldnames=METRIC_FIELDS)
        self._writer.writeheader()

    def write(self, **values: float | int | str) -> None:
        unknown = set(values).difference(METRIC_FIELDS)
        if unknown:
            raise KeyError(f"unknown metric fields: {sorted(unknown)}")
        self._writer.writerow(values)
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()

    def __enter__(self) -> MetricWriter:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def write_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")

