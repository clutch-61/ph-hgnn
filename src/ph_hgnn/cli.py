from __future__ import annotations

import argparse
import json
from dataclasses import fields, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import yaml

from ph_hgnn.data import load_dataset, make_local_global_witnesses, make_rhg_toy
from ph_hgnn.training import TrainConfig, run_experiment


def _load_config(path: Path) -> TrainConfig:
    payload: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    allowed = {field.name for field in fields(TrainConfig)}
    unknown = set(payload) - allowed
    if unknown:
        raise ValueError(f"unknown config keys: {sorted(unknown)}")
    return TrainConfig(**payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="PH-HGNN experiment runner")
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--dataset", default="rhg_toy")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--quick", action="store_true")
    return parser


def main() -> None:
    args = _parser().parse_args()
    project_root = Path.cwd()
    config = _load_config(args.config)
    if config.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("config requests CUDA, but torch.cuda.is_available() is false")
    if args.quick:
        config = replace(config, epochs=2, patience=2, hidden_dim=16, topo_dim=16)
    if args.dataset == "rhg_toy":
        dataset = make_rhg_toy(num_per_class=10, seed=config.seed)
    elif args.dataset == "local_global_witnesses":
        dataset = make_local_global_witnesses(
            depth=config.num_layers, num_pairs=10, seed=config.seed
        )
    else:
        dataset = load_dataset(args.dataset, args.data_root, seed=config.seed)
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = args.run_root / dataset.name / f"{timestamp}-seed{config.seed}"
    result = run_experiment(dataset, config, project_root, run_dir)
    print(json.dumps({"run_dir": str(run_dir), "test": result["test"]}, indent=2))


if __name__ == "__main__":
    main()
