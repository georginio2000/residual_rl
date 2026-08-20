"""Evaluate a frozen policy or saved SAC checkpoint."""

from __future__ import annotations

import argparse
import json

from last_millimeter.config import load_config
from last_millimeter.evaluation import evaluate_components
from last_millimeter.factory import make_components
from last_millimeter.policies import ControlMode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", help="Required unless mode is frozen")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", help="Override experiment.device")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.device is not None:
        config.experiment.device = args.device
    components = make_components(config)

    if config.mode is not ControlMode.FROZEN:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required for a learned policy")
        assert components.agent is not None
        components.agent.load(args.checkpoint, load_optimizers=False)

    metrics = evaluate_components(
        config,
        components,
        episodes=args.episodes or config.training.evaluation_episodes,
        seed=args.seed if args.seed is not None else config.experiment.seed + 10_000,
    )
    print(json.dumps(metrics, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

