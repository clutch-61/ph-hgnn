from __future__ import annotations

import hashlib
import json
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import gudhi
import numpy as np
import torch

from ph_hgnn.topology.complexes import ComplexKind, build_filtered_complex
from ph_hgnn.types import HypergraphSample

PersistenceDiagrams = dict[int, torch.Tensor]


def compute_persistence(
    sample: HypergraphSample,
    kind: ComplexKind | str = ComplexKind.SCC,
    homology_dims: tuple[int, ...] = (0, 1),
    max_complex_dim: int = 2,
) -> PersistenceDiagrams:
    tree = gudhi.SimplexTree()
    for simplex, filtration in build_filtered_complex(sample, kind, max_complex_dim):
        tree.insert(simplex, filtration=filtration)
    tree.make_filtration_non_decreasing()
    tree.persistence(homology_coeff_field=2, min_persistence=-1.0)
    diagrams: PersistenceDiagrams = {}
    for dim in homology_dims:
        array = tree.persistence_intervals_in_dimension(dim)
        if array.size == 0:
            diagrams[dim] = torch.empty((0, 2), dtype=torch.float32)
            continue
        finite = array[np.isfinite(array[:, 1]), 1]
        cap = float(finite.max() + 1.0) if finite.size else 1.0
        array = np.nan_to_num(array, posinf=cap)
        diagrams[dim] = torch.tensor(array, dtype=torch.float32)
    return diagrams


def barcode_statistics(diagram: torch.Tensor) -> torch.Tensor:
    """Five stable-size statistics used by the Aktas et al. baseline."""

    if diagram.numel() == 0:
        return torch.zeros(5, dtype=torch.float32)
    birth, death = diagram[:, 0], diagram[:, 1]
    lifetime = (death - birth).clamp_min(0)
    death_max = death.max()
    return torch.stack(
        (
            torch.tensor(float(diagram.shape[0])),
            torch.sum(birth * lifetime),
            torch.sum((death_max - death) * lifetime),
            torch.sum(birth.square() * lifetime.pow(4)),
            torch.sum((death_max - death).square() * lifetime.pow(4)),
        )
    )


def _sample_hash(sample: HypergraphSample, kind: str, max_dim: int) -> str:
    digest = hashlib.sha256()
    digest.update(sample.incidence.detach().cpu().numpy().tobytes())
    if sample.vertex_filtration is not None:
        digest.update(sample.vertex_filtration.detach().cpu().numpy().tobytes())
    if sample.hyperedge_filtration is not None:
        digest.update(sample.hyperedge_filtration.detach().cpu().numpy().tobytes())
    digest.update(
        json.dumps({"kind": kind, "max_dim": max_dim}, sort_keys=True).encode("utf-8")
    )
    return digest.hexdigest()[:20]


@dataclass(slots=True)
class PersistenceCache:
    root: Path

    def get_or_compute(
        self,
        sample: HypergraphSample,
        kind: ComplexKind | str,
        max_complex_dim: int = 2,
    ) -> PersistenceDiagrams:
        selected = ComplexKind(kind).value
        key = _sample_hash(sample, selected, max_complex_dim)
        path = self.root / selected / f"{sample.sample_id}-{key}.npz"
        if path.exists():
            with np.load(path, allow_pickle=False) as cached:
                return {
                    int(name[1:]): torch.tensor(cached[name], dtype=torch.float32)
                    for name in cached.files
                }
        diagrams = compute_persistence(
            sample, selected, max_complex_dim=max_complex_dim
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {f"d{dim}": diagram.cpu().numpy() for dim, diagram in diagrams.items()}
        with tempfile.NamedTemporaryFile(
            dir=path.parent, prefix=path.stem, suffix=".npz", delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
        try:
            np.savez_compressed(temporary_path, **payload)
            os.replace(temporary_path, path)
        finally:
            temporary_path.unlink(missing_ok=True)
        return diagrams
