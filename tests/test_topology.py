from pathlib import Path

import torch

from ph_hgnn.data.synthetic import make_local_global_witnesses, make_rhg_toy
from ph_hgnn.topology.complexes import ComplexKind, build_filtered_complex
from ph_hgnn.topology.persistence import PersistenceCache, compute_persistence
from ph_hgnn.types import HypergraphSample


def _permuted(sample: HypergraphSample) -> HypergraphSample:
    permutation = torch.randperm(sample.num_nodes, generator=torch.Generator().manual_seed(3))
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(sample.num_nodes)
    incidence = sample.incidence.clone()
    incidence[0] = inverse[incidence[0]]
    return HypergraphSample(
        f"{sample.sample_id}-permuted",
        sample.x[permutation],
        incidence,
        sample.y,
        sample.num_hyperedges,
    )


def test_all_complex_constructions_are_nonempty() -> None:
    sample = make_rhg_toy(num_per_class=10).samples[0]
    for kind in ComplexKind:
        assert build_filtered_complex(sample, kind)


def test_persistence_is_vertex_permutation_invariant() -> None:
    sample = make_rhg_toy(num_per_class=10).samples[0]
    relabeled = _permuted(sample)
    for kind in (ComplexKind.SCC, ComplexKind.INCIDENCE):
        first = compute_persistence(sample, kind)
        second = compute_persistence(relabeled, kind)
        assert torch.allclose(first[0].sort(dim=0).values, second[0].sort(dim=0).values)
        assert torch.allclose(first[1].sort(dim=0).values, second[1].sort(dim=0).values)


def test_cache_round_trip(tmp_path: Path) -> None:
    sample = make_rhg_toy(num_per_class=10).samples[0]
    cache = PersistenceCache(tmp_path)
    first = cache.get_or_compute(sample, ComplexKind.SCC)
    second = cache.get_or_compute(sample, ComplexKind.SCC)
    assert first.keys() == second.keys()
    assert all(torch.equal(first[dim], second[dim]) for dim in first)


def test_h0_separates_registered_local_global_witness() -> None:
    dataset = make_local_global_witnesses(depth=2, num_pairs=5)
    connected = next(sample for sample in dataset.samples if sample.y == 0)
    disconnected = next(sample for sample in dataset.samples if sample.y == 1)
    connected_h0 = compute_persistence(connected, ComplexKind.SCC)[0]
    disconnected_h0 = compute_persistence(disconnected, ComplexKind.SCC)[0]
    connected_essential = ((connected_h0[:, 1] - connected_h0[:, 0]) > 0).sum()
    disconnected_essential = ((disconnected_h0[:, 1] - disconnected_h0[:, 0]) > 0).sum()
    assert connected_essential == 1
    assert disconnected_essential == 2
