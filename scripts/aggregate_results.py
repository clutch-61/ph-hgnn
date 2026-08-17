"""Aggregate seed-level metrics without reading model checkpoints."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/summary.json"))
    args = parser.parse_args()
    groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for path in args.runs.glob("**/metrics.json"):
        record = json.loads(path.read_text(encoding="utf-8"))
        config = record["config"]
        key = "|".join(
            str(config[field])
            for field in ("backbone", "num_layers", "complex_kind", "fusion")
        )
        groups[f"{record['dataset']}|{key}"].append(
            (record["test"]["accuracy"], record["test"]["macro_f1"])
        )
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
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


if __name__ == "__main__":
    main()
