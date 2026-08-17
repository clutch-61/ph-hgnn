from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(slots=True)
class HypergraphSample:
    """One hypergraph-level classification example."""

    sample_id: str
    x: torch.Tensor
    incidence: torch.Tensor
    y: int
    num_hyperedges: int
    vertex_filtration: torch.Tensor | None = None
    hyperedge_filtration: torch.Tensor | None = None

    def validate(self) -> None:
        if self.x.ndim != 2:
            raise ValueError("x must have shape [num_nodes, num_features]")
        if self.incidence.ndim != 2 or self.incidence.shape[0] != 2:
            raise ValueError("incidence must have shape [2, num_memberships]")
        if self.incidence.dtype != torch.long:
            raise TypeError("incidence must use torch.long indices")
        if self.num_hyperedges < 1:
            raise ValueError("a sample must contain at least one hyperedge")
        if self.incidence.numel() == 0:
            raise ValueError("a sample must contain at least one incidence")
        vertices, hyperedges = self.incidence
        if int(vertices.min()) < 0 or int(vertices.max()) >= self.x.shape[0]:
            raise ValueError("incidence contains an invalid vertex index")
        if int(hyperedges.min()) < 0 or int(hyperedges.max()) >= self.num_hyperedges:
            raise ValueError("incidence contains an invalid hyperedge index")
        if self.vertex_filtration is not None and self.vertex_filtration.shape != (
            self.num_nodes,
        ):
            raise ValueError("vertex_filtration must have one value per node")
        if self.hyperedge_filtration is not None and self.hyperedge_filtration.shape != (
            self.num_hyperedges,
        ):
            raise ValueError("hyperedge_filtration must have one value per hyperedge")

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]


@dataclass(slots=True, frozen=True)
class DatasetBundle:
    """Samples and immutable train/validation/test indices."""

    name: str
    samples: tuple[HypergraphSample, ...]
    train_indices: tuple[int, ...]
    val_indices: tuple[int, ...]
    test_indices: tuple[int, ...]
    num_classes: int

    def validate(self) -> None:
        if not self.samples:
            raise ValueError("dataset is empty")
        for sample in self.samples:
            sample.validate()
        splits = (set(self.train_indices), set(self.val_indices), set(self.test_indices))
        if splits[0] & splits[1] or splits[0] & splits[2] or splits[1] & splits[2]:
            raise ValueError("dataset splits overlap")
        covered = splits[0] | splits[1] | splits[2]
        if covered != set(range(len(self.samples))):
            raise ValueError("dataset splits must cover every sample exactly once")
