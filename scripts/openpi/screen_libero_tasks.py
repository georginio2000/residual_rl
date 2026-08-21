"""Screen multiple LIBERO tasks with an already-running frozen OpenPI server.

Run this inside the isolated OpenPI LIBERO client image. Each task uses the
pinned single-task evaluator, batch-size-1 inference, and its own artifact
directory. The aggregate summary identifies zero-success, perfect, and
imperfect tasks for follow-up evaluation.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from eval_libero_task import MAX_STEPS, evaluate


OPENPI_COMMIT = "15a9616a00943ada6c20a0f158e3adb39df2ccac"


def parse_task_ids(value: str) -> list[int]:
    try:
        task_ids = [int(part.strip()) for part in value.split(",") if part.strip()]
    except ValueError as exc:
        raise argparse.ArgumentTypeError("task IDs must be comma-separated integers") from exc
    if not task_ids:
        raise argparse.ArgumentTypeError("at least one task ID is required")
    if len(set(task_ids)) != len(task_ids):
        raise argparse.ArgumentTypeError("task IDs must be unique")
    if any(task_id < 0 for task_id in task_ids):
        raise argparse.ArgumentTypeError("task IDs cannot be negative")
    return task_ids


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="host.docker.internal")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument(
        "--task-suite-name",
        choices=sorted(MAX_STEPS),
        default="libero_spatial",
    )
    parser.add_argument(
        "--task-ids",
        type=parse_task_ids,
        default=parse_task_ids("0,1,2,3,4,5,6,7,8,9"),
    )
    parser.add_argument("--num-trials", type=int, default=1)
    parser.add_argument("--initial-state-offset", type=int, default=0)
    parser.add_argument("--resize-size", type=int, default=224)
    parser.add_argument("--replan-steps", type=int, default=5)
    parser.add_argument("--num-steps-wait", type=int, default=10)
    parser.add_argument("--action-bias-x", type=float, default=0.0)
    parser.add_argument("--scene-shift-x", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("/data/libero_screen"))
    return parser.parse_args()


def build_summary(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if "error" not in record]
    imperfect = [
        int(record["task_id"])
        for record in completed
        if 0 < int(record["successes"]) < int(record["trials"])
    ]
    zero_success = [
        int(record["task_id"])
        for record in completed
        if int(record["successes"]) == 0
    ]
    perfect = [
        int(record["task_id"])
        for record in completed
        if int(record["successes"]) == int(record["trials"])
    ]
    total_trials = sum(int(record["trials"]) for record in completed)
    total_successes = sum(int(record["successes"]) for record in completed)
    return {
        "policy": "frozen pi05_libero",
        "openpi_commit": OPENPI_COMMIT,
        "batch_size": 1,
        "task_suite": args.task_suite_name,
        "task_ids": args.task_ids,
        "trials_per_task": args.num_trials,
        "initial_state_offset": args.initial_state_offset,
        "action_bias_x": args.action_bias_x,
        "scene_shift_x": args.scene_shift_x,
        "seed": args.seed,
        "total_trials": total_trials,
        "total_successes": total_successes,
        "aggregate_success_rate": total_successes / total_trials if total_trials else 0.0,
        "imperfect_candidate_task_ids": imperfect,
        "zero_success_task_ids": zero_success,
        "perfect_task_ids": perfect,
        "tasks": records,
    }


def write_summary(args: argparse.Namespace, records: list[dict[str, Any]]) -> dict[str, Any]:
    summary = build_summary(args, records)
    result_path = args.output_dir / "screen_result.json"
    with result_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    args = parse_args()
    if args.num_trials <= 0:
        raise ValueError("num_trials must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for task_id in args.task_ids:
        task_output_dir = args.output_dir / f"task_{task_id}"
        task_args = argparse.Namespace(
            host=args.host,
            port=args.port,
            task_suite_name=args.task_suite_name,
            task_id=task_id,
            num_trials=args.num_trials,
            initial_state_offset=args.initial_state_offset,
            resize_size=args.resize_size,
            replan_steps=args.replan_steps,
            num_steps_wait=args.num_steps_wait,
            action_bias_x=args.action_bias_x,
            scene_shift_x=args.scene_shift_x,
            seed=args.seed,
            output_dir=task_output_dir,
        )
        try:
            record = evaluate(task_args)
        except Exception as exc:
            logging.exception("task %d failed during screening", task_id)
            record = {
                "task_id": task_id,
                "error": type(exc).__name__,
                "message": str(exc),
            }
        records.append(record)
        write_summary(args, records)

    summary = write_summary(args, records)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
