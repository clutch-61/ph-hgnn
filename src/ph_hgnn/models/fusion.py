from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from ph_hgnn.models.backbones import HypergraphEncoder
from ph_hgnn.models.topology import (
    PersistenceDeepSetEncoder,
    PersistenceImageEncoder,
    PersistenceStatsEncoder,
    PersLayEncoder,
)
from ph_hgnn.topology.persistence import PersistenceDiagrams
from ph_hgnn.types import HypergraphSample

FusionMode = Literal["hgnn", "ph", "concat", "add", "gated"]


class PHHGNNClassifier(nn.Module):
    """Hypergraph classifier with an optional topological residual branch."""

    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        hidden_dim: int = 64,
        topo_dim: int = 64,
        num_layers: int = 2,
        backbone: str = "hgnn",
        fusion: FusionMode = "gated",
        topology_encoder: str = "deepset",
    ):
        super().__init__()
        if fusion not in {"hgnn", "ph", "concat", "add", "gated"}:
            raise ValueError(f"unknown fusion mode: {fusion}")
        self.fusion = fusion
        self.local_encoder = HypergraphEncoder(
            input_dim, hidden_dim, num_layers, backbone=backbone
        )
        topology_encoders: dict[str, nn.Module] = {
            "statistics": PersistenceStatsEncoder(output_dim=topo_dim),
            "image": PersistenceImageEncoder(output_dim=topo_dim),
            "deepset": PersistenceDeepSetEncoder(hidden_dim=topo_dim, output_dim=topo_dim),
            "perslay": PersLayEncoder(output_dim=topo_dim),
        }
        if topology_encoder not in topology_encoders:
            raise ValueError(
                f"unknown topology encoder {topology_encoder!r}; "
                f"choose {sorted(topology_encoders)}"
            )
        self.topology_encoder = topology_encoders[topology_encoder]
        fusion_dim = 2 * hidden_dim
        self.local_projection = nn.Linear(self.local_encoder.output_dim, fusion_dim)
        self.topology_projection = nn.Linear(topo_dim, fusion_dim)
        self.gate = nn.Sequential(nn.Linear(2 * fusion_dim, fusion_dim), nn.Sigmoid())
        classifier_input = 2 * fusion_dim if fusion == "concat" else fusion_dim
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, num_classes),
        )
        if fusion == "hgnn":
            for module in (self.topology_encoder, self.topology_projection, self.gate):
                module.requires_grad_(False)
        elif fusion == "ph":
            for module in (self.local_encoder, self.local_projection, self.gate):
                module.requires_grad_(False)

    def encode(
        self, sample: HypergraphSample, diagrams: PersistenceDiagrams
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        if self.fusion == "hgnn":
            return self.local_projection(self.local_encoder(sample)), None
        if self.fusion == "ph":
            return self.topology_projection(self.topology_encoder(diagrams)), None
        local = self.local_projection(self.local_encoder(sample))
        topology = self.topology_projection(self.topology_encoder(diagrams))
        if self.fusion == "concat":
            return torch.cat((local, topology), dim=0), None
        if self.fusion == "add":
            return local + topology, None
        gate = self.gate(torch.cat((local, topology), dim=0))
        return local + gate * topology, gate

    def forward(
        self, sample: HypergraphSample, diagrams: PersistenceDiagrams
    ) -> torch.Tensor:
        embedding, _ = self.encode(sample, diagrams)
        return self.classifier(embedding)
