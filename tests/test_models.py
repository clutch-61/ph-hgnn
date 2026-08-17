import pytest
import torch

from ph_hgnn.data.synthetic import make_local_global_witnesses, make_rhg_toy
from ph_hgnn.models.backbones import HypergraphEncoder
from ph_hgnn.models.fusion import PHHGNNClassifier
from ph_hgnn.topology.persistence import compute_persistence


@pytest.mark.parametrize("fusion", ["hgnn", "ph", "concat", "add", "gated"])
def test_all_fusion_modes_forward_backward(fusion: str) -> None:
    sample = make_rhg_toy(num_per_class=10).samples[0]
    diagrams = compute_persistence(sample)
    model = PHHGNNClassifier(
        input_dim=sample.x.shape[1],
        num_classes=3,
        hidden_dim=8,
        topo_dim=8,
        fusion=fusion,  # type: ignore[arg-type]
    )
    logits = model(sample, diagrams)
    assert logits.shape == (3,)
    torch.nn.functional.cross_entropy(logits.unsqueeze(0), torch.tensor([sample.y])).backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


@pytest.mark.parametrize(
    "backbone",
    ["hgnn", "hnhn", "unigcn", "unigin", "alldeepsets", "allsettransformer"],
)
def test_supported_backbones(backbone: str) -> None:
    sample = make_rhg_toy(num_per_class=10).samples[0]
    model = PHHGNNClassifier(
        input_dim=sample.x.shape[1],
        num_classes=3,
        hidden_dim=8,
        topo_dim=8,
        backbone=backbone,
    )
    assert model(sample, compute_persistence(sample)).shape == (3,)


@pytest.mark.parametrize("encoder", ["statistics", "image", "deepset", "perslay"])
def test_supported_topology_encoders(encoder: str) -> None:
    sample = make_rhg_toy(num_per_class=10).samples[0]
    model = PHHGNNClassifier(
        input_dim=sample.x.shape[1],
        num_classes=3,
        hidden_dim=8,
        topo_dim=8,
        topology_encoder=encoder,
    )
    assert model(sample, compute_persistence(sample)).shape == (3,)


def test_finite_hop_encoder_cannot_separate_registered_witness() -> None:
    dataset = make_local_global_witnesses(depth=2, num_pairs=5)
    connected = next(sample for sample in dataset.samples if sample.y == 0)
    disconnected = next(sample for sample in dataset.samples if sample.y == 1)
    encoder = HypergraphEncoder(input_dim=2, hidden_dim=8, num_layers=2, dropout=0)
    encoder.eval()
    assert torch.allclose(encoder(connected), encoder(disconnected), atol=1e-6)
