"""Aggregate seed-level metrics without reading model checkpoints."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def _run_key(config: dict[str, object]) -> str:
    fields = (
        "backbone",
        "num_layers",
        "complex_kind",
        "fusion",
        "topology_encoder",
    )
    return "|".join(str(config.get(field, "na")) for field in fields)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/summary.json"))
    parser.add_argument("--pattern", default="**/metrics.json")
    args = parser.parse_args()
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    seeds: dict[str, set[int]] = defaultdict(set)
    folds: dict[str, set[int]] = defaultdict(set)
    for path in args.runs.glob(args.pattern):
        record = json.loads(path.read_text(encoding="utf-8"))
        config = record["config"]
        key = _run_key(config)
        dataset_key = f"{record['dataset']}|{key}"
        groups[dataset_key].append(
            (record["test"]["accuracy"], record["test"]["macro_f1"])
        )
        seeds[dataset_key].add(int(config["seed"]))
        if "-fold" in path.parent.name:
            fold = int(path.parent.name.rsplit("-fold", maxsplit=1)[1])
            folds[dataset_key].add(fold)
    summary: dict[str, dict[str, float | int]] = {}
    for key, values in groups.items():
        array = np.asarray(values)
        count = len(values)
        ci_scale = 1.96 / np.sqrt(count)
        summary[key] = {
            "runs": count,
            "accuracy_mean": float(array[:, 0].mean()),
            "accuracy_std": float(array[:, 0].std(ddof=1)) if count > 1 else 0.0,
            "accuracy_ci95": float(array[:, 0].std(ddof=1) * ci_scale)
            if count > 1
            else 0.0,
            "macro_f1_mean": float(array[:, 1].mean()),
            "macro_f1_std": float(array[:, 1].std(ddof=1)) if count > 1 else 0.0,
            "macro_f1_ci95": float(array[:, 1].std(ddof=1) * ci_scale)
            if count > 1
            else 0.0,
            "seeds": len(seeds[key]),
            "folds": len(folds[key]),
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
