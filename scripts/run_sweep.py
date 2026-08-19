"""Run seed/fold sweeps for holdout and 10-fold datasets."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml

K_FOLD_DATASETS = {"highschool", "primary", "makam", "bbc"}


def _build_config(base: dict[str, object], seed: int) -> dict[str, object]:
    config = dict(base)
    config["seed"] = seed
    return config


def _run_once(
    *,
    config_path: Path,
    dataset: str,
    fold: int,
    run_root: Path,
    quick: bool,
) -> None:
    command = [
        sys.executable,
        "-m",
        "ph_hgnn.cli",
        "--config",
        str(config_path),
        "--dataset",
        dataset,
        "--run-root",
        str(run_root),
    ]
    if dataset in K_FOLD_DATASETS:
        command.extend(["--fold", str(fold)])
    if quick:
        command.append("--quick")
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--run-root", type=Path, default=Path("runs"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--folds", type=int, nargs="+", default=list(range(10)))
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    folds = args.folds if args.dataset in K_FOLD_DATASETS else [0]
    with tempfile.TemporaryDirectory(prefix="ph-hgnn-sweep-") as temporary:
        temporary_root = Path(temporary)
        run_id = 0
        for seed in args.seeds:
            config = _build_config(base, seed)
            config_path = temporary_root / f"seed-{seed}.yaml"
            config_path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
            for fold in folds:
                run_id += 1
                print(f"[{run_id}] dataset={args.dataset} seed={seed} fold={fold}")
                if not args.dry_run:
                    _run_once(
                        config_path=config_path,
                        dataset=args.dataset,
                        fold=fold,
                        run_root=args.run_root,
                        quick=args.quick,
                    )


if __name__ == "__main__":
    main()
