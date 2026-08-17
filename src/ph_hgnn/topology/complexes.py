from __future__ import annotations

import itertools
from enum import Enum

import torch

from ph_hgnn.types import HypergraphSample

FilteredSimplex = tuple[tuple[int, ...], float]


class ComplexKind(str, Enum):
    SCC = "scc"
    RESBS = "resbs"
    RELBS = "relbs"
    INCIDENCE = "incidence"


def _hyperedges(sample: HypergraphSample) -> list[frozenset[int]]:
    edges: list[set[int]] = [set() for _ in range(sample.num_hyperedges)]
    for vertex, edge in sample.incidence.t().tolist():
        edges[edge].add(vertex)
    return [frozenset(edge) for edge in edges]


def _normalized(values: torch.Tensor) -> list[float]:
    values = values.float()
    low, high = values.min(), values.max()
    if float(high - low) == 0:
        return [0.0] * values.numel()
    return ((values - low) / (high - low)).tolist()


def _vertex_filtration(sample: HypergraphSample) -> list[float]:
    if sample.vertex_filtration is not None:
        return _normalized(sample.vertex_filtration)
    degree = torch.bincount(sample.incidence[0], minlength=sample.num_nodes)
    return _normalized(degree)


def _edge_filtration(
    edges: list[frozenset[int]], provided: torch.Tensor | None = None
) -> list[float]:
    if provided is not None:
        return _normalized(provided)
    return _normalized(torch.tensor([len(edge) for edge in edges]))


def _scc(sample: HypergraphSample, max_dim: int) -> list[FilteredSimplex]:
    values = _vertex_filtration(sample)
    simplices: dict[tuple[int, ...], float] = {}
    for edge in _hyperedges(sample):
        ordered = sorted(edge)
        for size in range(1, min(len(ordered), max_dim + 1) + 1):
            for simplex in itertools.combinations(ordered, size):
                simplices[simplex] = max(values[vertex] for vertex in simplex)
    return sorted(simplices.items(), key=lambda item: (item[1], len(item[0]), item[0]))


def _chains(edges: list[frozenset[int]], max_length: int) -> list[tuple[int, ...]]:
    chains: set[tuple[int, ...]] = {(i,) for i in range(len(edges))}
    for length in range(2, max_length + 1):
        for indices in itertools.combinations(range(len(edges)), length):
            ordered = tuple(sorted(indices, key=lambda i: (len(edges[i]), i)))
            if all(edges[a] < edges[b] for a, b in zip(ordered, ordered[1:], strict=False)):
                chains.add(ordered)
    return sorted(chains, key=lambda chain: (len(chain), chain))


def _resbs(sample: HypergraphSample, max_dim: int) -> list[FilteredSimplex]:
    edges = _hyperedges(sample)
    values = _edge_filtration(edges, sample.hyperedge_filtration)
    return [
        (chain, max(values[index] for index in chain))
        for chain in _chains(edges, max_length=max_dim + 1)
    ]


def _all_closure_faces(edges: list[frozenset[int]]) -> set[frozenset[int]]:
    faces: set[frozenset[int]] = set()
    for edge in edges:
        ordered = sorted(edge)
        for size in range(1, len(ordered) + 1):
            faces.update(frozenset(face) for face in itertools.combinations(ordered, size))
    return faces


def _relbs(sample: HypergraphSample, max_dim: int) -> list[FilteredSimplex]:
    """Relative barycentric model sufficient for H0/H1.

    Observed hyperedges retain their order-complex simplices. All missing faces
    in the simplicial closure are collapsed to one quotient vertex. Edges from
    that vertex preserve observed/missing face incidences; higher simplices
    incident to the quotient are discarded, matching the simplicial
    modification used to avoid duplicate quotient edges.
    """

    edges = _hyperedges(sample)
    observed = set(edges)
    missing = _all_closure_faces(edges) - observed
    values = _edge_filtration(edges, sample.hyperedge_filtration)
    simplices: dict[tuple[int, ...], float] = {
        chain: max(values[index] for index in chain)
        for chain in _chains(edges, max_length=max_dim + 1)
    }
    if not missing:
        return list(simplices.items())
    quotient = len(edges)
    simplices[(quotient,)] = 0.0
    for index, edge in enumerate(edges):
        if any(face < edge or edge < face for face in missing):
            simplices[tuple(sorted((index, quotient)))] = values[index]
    return sorted(simplices.items(), key=lambda item: (item[1], len(item[0]), item[0]))


def _incidence(sample: HypergraphSample) -> list[FilteredSimplex]:
    vertex_values = _vertex_filtration(sample)
    edge_values = _edge_filtration(_hyperedges(sample), sample.hyperedge_filtration)
    offset = sample.num_nodes
    simplices: list[FilteredSimplex] = [
        ((vertex,), value) for vertex, value in enumerate(vertex_values)
    ]
    simplices.extend(
        ((offset + edge,), value) for edge, value in enumerate(edge_values)
    )
    for vertex, edge in sample.incidence.t().tolist():
        filtration = max(vertex_values[vertex], edge_values[edge])
        simplices.append(((vertex, offset + edge), filtration))
    return simplices


def build_filtered_complex(
    sample: HypergraphSample,
    kind: ComplexKind | str = ComplexKind.SCC,
    max_dim: int = 2,
) -> list[FilteredSimplex]:
    """Build a permutation-invariant filtered complex from one hypergraph."""

    sample.validate()
    if max_dim < 1:
        raise ValueError("max_dim must be at least one")
    selected = ComplexKind(kind)
    if selected is ComplexKind.SCC:
        return _scc(sample, max_dim)
    if selected is ComplexKind.RESBS:
        return _resbs(sample, max_dim)
    if selected is ComplexKind.RELBS:
        return _relbs(sample, max_dim)
    return _incidence(sample)
