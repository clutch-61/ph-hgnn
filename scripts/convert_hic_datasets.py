"""Convert HIC hypergraph .txt files into PH-HGNN JSON interchange format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

HIC_HYPERGRAPH_FILES: dict[str, tuple[str, str]] = {
    "rhg_3": ("RHG", "RHG_3.txt"),
    "rhg_10": ("RHG", "RHG_10.txt"),
    "imdb_dir_form": ("IMDB", "IMDB_dir_form.txt"),
    "imdb_dir_genre": ("IMDB", "IMDB_dir_genre.txt"),
    "steam_player": ("STEAM", "stream_player.txt"),
    "twitter_friend": ("TWITTER", "twitter_friend.txt"),
}


def convert_file(source: Path, dataset_name: str) -> list[dict[str, object]]:
    with source.open("r", encoding="utf-8") as handle:
        num_graphs = int(handle.readline().strip())
        samples: list[dict[str, object]] = []
        for index in range(num_graphs):
            header = handle.readline().strip().split()
            num_nodes = int(header[0])
            num_edges = int(header[1])
            label = int(header[2])
            _ = handle.readline()  # vertex labels are optional for our loader
            hyperedges: list[list[int]] = []
            for _ in range(num_edges):
                edge = [int(value) for value in handle.readline().strip().split()]
                if len(edge) >= 2:
                    hyperedges.append(edge)
            if not hyperedges:
                hyperedges = [[0, min(1, num_nodes - 1)]] if num_nodes > 1 else [[0]]
            samples.append(
                {
                    "id": f"{dataset_name}-{index}",
                    "label": label,
                    "num_nodes": num_nodes,
                    "hyperedges": hyperedges,
                }
            )
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hic-root",
        type=Path,
        default=Path("data/raw/HIC/data/hypergraph"),
        help="Path to HIC data/hypergraph directory",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/processed"),
        help="Directory for generated JSON files",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=list(HIC_HYPERGRAPH_FILES),
        choices=list(HIC_HYPERGRAPH_FILES),
    )
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.datasets:
        folder, filename = HIC_HYPERGRAPH_FILES[dataset_name]
        source = args.hic_root / folder / filename
        if not source.exists():
            raise FileNotFoundError(f"missing HIC source file: {source}")
        samples = convert_file(source, dataset_name)
        target = args.output_root / f"{dataset_name}.json"
        target.write_text(json.dumps({"samples": samples}), encoding="utf-8")
        print(f"{dataset_name}: {len(samples)} samples -> {target}")


if __name__ == "__main__":
    main()
