from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import train_test_split

from ph_hgnn.types import DatasetBundle, HypergraphSample


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    canonical_name: str
    upstream_name: str
    source: str
    num_classes: int


DATASET_SPECS: dict[str, DatasetSpec] = {
    "rhg_3": DatasetSpec("rhg_3", "RHG_3", "https://github.com/iMoonLab/HIC", 3),
    "rhg_10": DatasetSpec("rhg_10", "RHG_10", "https://github.com/iMoonLab/HIC", 10),
    "imdb_dir_form": DatasetSpec(
        "imdb_dir_form", "IMDB_dir_form", "https://github.com/iMoonLab/HIC", 3
    ),
    "imdb_dir_genre": DatasetSpec(
        "imdb_dir_genre", "IMDB_dir_genre", "https://github.com/iMoonLab/HIC", 3
    ),
    "steam_player": DatasetSpec(
        "steam_player", "stream_player", "https://github.com/iMoonLab/HIC", 2
    ),
    "twitter_friend": DatasetSpec(
        "twitter_friend", "twitter_friend", "https://github.com/iMoonLab/HIC", 2
    ),
}


def _incidence_from_hyperedges(hyperedges: list[list[int]]) -> tuple[torch.Tensor, int]:
    pairs = [(vertex, edge_id) for edge_id, edge in enumerate(hyperedges) for vertex in edge]
    if not pairs:
        raise ValueError("hyperedges must contain at least one membership")
    return torch.tensor(pairs, dtype=torch.long).t().contiguous(), len(hyperedges)


def _structural_features(num_nodes: int, incidence: torch.Tensor) -> torch.Tensor:
    vertex_degree = torch.bincount(incidence[0], minlength=num_nodes).float()
    edge_size = torch.bincount(incidence[1]).float()
    mean_edge_size = torch.zeros(num_nodes, dtype=torch.float32)
    mean_edge_size.index_add_(0, incidence[0], edge_size[incidence[1]])
    mean_edge_size /= vertex_degree.clamp_min(1)
    features = torch.stack((vertex_degree, mean_edge_size), dim=1)
    scale = features.amax(dim=0, keepdim=True).clamp_min(1)
    return features / scale


def _parse_sample(record: dict[str, Any], index: int) -> HypergraphSample:
    hyperedges = [[int(v) for v in edge] for edge in record["hyperedges"]]
    incidence, num_hyperedges = _incidence_from_hyperedges(hyperedges)
    max_vertex = int(incidence[0].max())
    num_nodes = int(record.get("num_nodes", max_vertex + 1))
    if "x" in record:
        x = torch.tensor(record["x"], dtype=torch.float32)
    else:
        x = _structural_features(num_nodes, incidence)
    sample = HypergraphSample(
        sample_id=str(record.get("id", index)),
        x=x,
        incidence=incidence,
        y=int(record["label"]),
        num_hyperedges=num_hyperedges,
        vertex_filtration=torch.tensor(record["vertex_filtration"], dtype=torch.float32)
        if "vertex_filtration" in record
        else None,
        hyperedge_filtration=torch.tensor(
            record["hyperedge_filtration"], dtype=torch.float32
        )
        if "hyperedge_filtration" in record
        else None,
    )
    sample.validate()
    return sample


def stratified_split(
    labels: list[int], seed: int, train_ratio: float = 0.8, val_ratio: float = 0.1
) -> tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]:
    """Create a deterministic, disjoint split without fitting on sample features."""

    if train_ratio <= 0 or val_ratio <= 0 or train_ratio + val_ratio >= 1:
        raise ValueError("split ratios must be positive and sum to less than one")
    indices = np.arange(len(labels))
    train, remainder = train_test_split(
        indices, train_size=train_ratio, random_state=seed, stratify=labels
    )
    remainder_labels = np.asarray(labels)[remainder]
    relative_val_ratio = val_ratio / (1 - train_ratio)
    val, test = train_test_split(
        remainder,
        train_size=relative_val_ratio,
        random_state=seed,
        stratify=remainder_labels,
    )
    return tuple(map(int, train)), tuple(map(int, val)), tuple(map(int, test))


def load_json_dataset(path: Path, name: str, seed: int = 0) -> DatasetBundle:
    """Load the safe interchange format documented in ``data/README.md``."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload["samples"]
    samples = tuple(_parse_sample(record, i) for i, record in enumerate(records))
    labels = [sample.y for sample in samples]
    train, val, test = stratified_split(labels, seed=seed)
    inferred_classes = max(labels) + 1
    bundle = DatasetBundle(name, samples, train, val, test, inferred_classes)
    bundle.validate()
    return bundle


def load_dataset(name: str, root: Path, seed: int = 0) -> DatasetBundle:
    if name not in DATASET_SPECS:
        choices = ", ".join(sorted(DATASET_SPECS))
        raise KeyError(f"unknown dataset {name!r}; choose one of: {choices}")
    path = root / f"{name}.json"
    if not path.exists():
        spec = DATASET_SPECS[name]
        raise FileNotFoundError(
            f"{path} is missing. Export {spec.upstream_name!r} from {spec.source} "
            "to the safe JSON format described in data/README.md."
        )
    return load_json_dataset(path, name=name, seed=seed)
