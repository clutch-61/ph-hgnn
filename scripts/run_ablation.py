"""Launch the frozen depth, complex, and fusion ablation grid."""

from __future__ import annotations

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import yaml


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--base", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    base = yaml.safe_load(args.base.read_text(encoding="utf-8"))
    grid = [
        (depth, complex_kind, fusion, seed)
        for depth in (1, 2, 3, 4)
        for complex_kind in ("scc", "resbs", "relbs", "incidence")
        for fusion in ("hgnn", "ph", "gated")
        for seed in args.seeds
    ]
    with tempfile.TemporaryDirectory(prefix="ph-hgnn-configs-") as temporary:
        temporary_root = Path(temporary)
        for run_number, (depth, complex_kind, fusion, seed) in enumerate(grid, start=1):
            config = {
                **base,
                "num_layers": depth,
                "complex_kind": complex_kind,
                "fusion": fusion,
                "seed": seed,
            }
            path = temporary_root / f"run-{run_number}.yaml"
            path.write_text(yaml.safe_dump(config, sort_keys=True), encoding="utf-8")
            command = [
                sys.executable,
                "-m",
                "ph_hgnn.cli",
                "--config",
                str(path),
                "--dataset",
                args.dataset,
            ]
            print(" ".join(command))
            if not args.dry_run:
                subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
