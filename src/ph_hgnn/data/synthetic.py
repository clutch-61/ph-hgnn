from __future__ import annotations

import random

import torch

from ph_hgnn.data.datasets import stratified_split
from ph_hgnn.types import DatasetBundle, HypergraphSample


def _sample_from_edges(sample_id: str, edges: list[list[int]], label: int) -> HypergraphSample:
    pairs = [(vertex, edge_id) for edge_id, edge in enumerate(edges) for vertex in edge]
    incidence = torch.tensor(pairs, dtype=torch.long).t().contiguous()
    num_nodes = int(incidence[0].max()) + 1
    degree = torch.bincount(incidence[0], minlength=num_nodes).float()
    x = torch.stack((degree / degree.max(), torch.ones_like(degree)), dim=1)
    return HypergraphSample(sample_id, x, incidence, label, len(edges))


def _cycle_hypergraph(size: int) -> list[list[int]]:
    return [[i, (i + 1) % size, (i + 2) % size] for i in range(size)]


def _flower_hypergraph(petals: int) -> list[list[int]]:
    edges: list[list[int]] = []
    for petal in range(petals):
        left = 2 * petal + 1
        edges.append([0, left, left + 1])
    return edges


def _wheel_hypergraph(rim: int) -> list[list[int]]:
    return [[0, i + 1, ((i + 1) % rim) + 1] for i in range(rim)]


def make_rhg_toy(num_per_class: int = 20, seed: int = 0) -> DatasetBundle:
    """Small structural benchmark used by CPU CI and end-to-end smoke tests."""

    if num_per_class < 10:
        raise ValueError("num_per_class must be at least 10 for stratified 80/10/10 splits")
    rng = random.Random(seed)
    generators = (_cycle_hypergraph, _flower_hypergraph, _wheel_hypergraph)
    samples: list[HypergraphSample] = []
    for label, generator in enumerate(generators):
        for item in range(num_per_class):
            size = rng.randint(6, 10)
            edges = generator(size)
            samples.append(_sample_from_edges(f"class{label}-{item}", edges, label))
    rng.shuffle(samples)
    labels = [sample.y for sample in samples]
    train, val, test = stratified_split(labels, seed)
    bundle = DatasetBundle("rhg_toy", tuple(samples), train, val, test, 3)
    bundle.validate()
    return bundle


def _cycle_edges(vertices: list[int]) -> list[list[int]]:
    return [[vertices[i], vertices[(i + 1) % len(vertices)]] for i in range(len(vertices))]


def make_local_global_witnesses(
    depth: int = 2, num_pairs: int = 20, seed: int = 0
) -> DatasetBundle:
    """Cycles versus two cycles with identical depth-L rooted local views.

    For component length greater than ``2 * depth + 1``, every rooted
    incidence neighborhood seen by an L-layer message-passing network is a
    path with the same constant features. Both classes have the same node and
    incidence counts; H0 distinguishes one connected component from two.
    """

    if depth < 1 or num_pairs < 5:
        raise ValueError("depth must be positive and num_pairs must be at least five")
    rng = random.Random(seed)
    component_size = 2 * depth + 4
    num_nodes = 2 * component_size
    samples: list[HypergraphSample] = []
    for pair in range(num_pairs):
        permutation = list(range(num_nodes))
        rng.shuffle(permutation)
        connected = _cycle_edges(permutation)
        disconnected = _cycle_edges(permutation[:component_size]) + _cycle_edges(
            permutation[component_size:]
        )
        samples.append(_sample_from_edges(f"connected-{pair}", connected, 0))
        samples.append(_sample_from_edges(f"disconnected-{pair}", disconnected, 1))
    rng.shuffle(samples)
    labels = [sample.y for sample in samples]
    train, val, test = stratified_split(labels, seed)
    bundle = DatasetBundle("local_global_witnesses", tuple(samples), train, val, test, 2)
    bundle.validate()
    return bundle
