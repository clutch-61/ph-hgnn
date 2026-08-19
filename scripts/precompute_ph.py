"""Precompute persistent diagrams on the implementation server."""

from __future__ import annotations

import argparse
from pathlib import Path

from ph_hgnn.data import load_dataset, make_local_global_witnesses, make_rhg_toy
from ph_hgnn.topology.complexes import ComplexKind
from ph_hgnn.topology.persistence import PersistenceCache


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="rhg_toy")
    parser.add_argument("--data-root", type=Path, default=Path("data/processed"))
    parser.add_argument("--cache-root", type=Path, default=Path("artifacts/cache/persistence"))
    parser.add_argument("--complex", choices=[kind.value for kind in ComplexKind], default="scc")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fold", type=int, default=0)
    args = parser.parse_args()

    if args.dataset == "rhg_toy":
        dataset = make_rhg_toy(seed=args.seed)
    elif args.dataset == "local_global_witnesses":
        dataset = make_local_global_witnesses(seed=args.seed)
    else:
        dataset = load_dataset(
            args.dataset, args.data_root, seed=args.seed, fold_index=args.fold
        )
    cache = PersistenceCache(args.cache_root)
    for position, sample in enumerate(dataset.samples, start=1):
        cache.get_or_compute(sample, args.complex)
        print(f"\r{position}/{len(dataset.samples)}", end="", flush=True)
    print()


if __name__ == "__main__":
    main()
