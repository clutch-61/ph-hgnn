from __future__ import annotations

import torch
from torch import nn

from ph_hgnn.types import HypergraphSample

SUPPORTED_BACKBONES = {
    "hgnn": "mean",
    "hnhn": "sum",
    "unigcn": "mean",
    "unigin": "sum",
    "alldeepsets": "mean",
    "allsettransformer": "attention",
}


def _aggregate(
    source: torch.Tensor,
    target_index: torch.Tensor,
    num_targets: int,
    reduction: str,
) -> torch.Tensor:
    output = source.new_zeros((num_targets, source.shape[1]))
    output.index_add_(0, target_index, source)
    if reduction == "mean":
        count = torch.bincount(target_index, minlength=num_targets).to(source.dtype)
        output /= count.clamp_min(1).unsqueeze(1)
    return output


class HypergraphConv(nn.Module):
    """Vertex-to-hyperedge-to-vertex incidence message passing."""

    def __init__(self, hidden_dim: int, reduction: str = "mean", edge_mlp: bool = False):
        super().__init__()
        self.reduction = reduction
        self.vertex_projection = nn.Linear(hidden_dim, hidden_dim)
        self.edge_update: nn.Module = (
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.ReLU())
            if edge_mlp
            else nn.Identity()
        )
        self.vertex_update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self, x: torch.Tensor, incidence: torch.Tensor, num_hyperedges: int
    ) -> torch.Tensor:
        vertex_index, edge_index = incidence
        messages = self.vertex_projection(x)[vertex_index]
        edge_features = _aggregate(messages, edge_index, num_hyperedges, self.reduction)
        edge_features = self.edge_update(edge_features)
        returned = _aggregate(
            edge_features[edge_index], vertex_index, x.shape[0], self.reduction
        )
        return self.vertex_update(torch.cat((x, returned), dim=1))


class SetAttentionHypergraphConv(nn.Module):
    """Two-stage set attention over each hyperedge and each vertex star."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        if hidden_dim % 4:
            raise ValueError("AllSetTransformer hidden_dim must be divisible by four")
        self.edge_seed = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.node_to_edge = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.edge_to_node = nn.MultiheadAttention(hidden_dim, 4, batch_first=True)
        self.update = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(
        self, x: torch.Tensor, incidence: torch.Tensor, num_hyperedges: int
    ) -> torch.Tensor:
        vertex_index, edge_index = incidence
        edge_features: list[torch.Tensor] = []
        for edge in range(num_hyperedges):
            members = x[vertex_index[edge_index == edge]].unsqueeze(0)
            query = self.edge_seed.expand(1, -1, -1)
            attended, _ = self.node_to_edge(query, members, members, need_weights=False)
            edge_features.append(attended.squeeze(0).squeeze(0))
        stacked_edges = torch.stack(edge_features)
        returned: list[torch.Tensor] = []
        for vertex in range(x.shape[0]):
            neighbors = stacked_edges[edge_index[vertex_index == vertex]].unsqueeze(0)
            query = x[vertex].reshape(1, 1, -1)
            attended, _ = self.edge_to_node(query, neighbors, neighbors, need_weights=False)
            returned.append(attended.squeeze(0).squeeze(0))
        return self.update(torch.cat((x, torch.stack(returned)), dim=1))


class HypergraphEncoder(nn.Module):
    """Finite-hop HGNN family with a common graph-level readout."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        num_layers: int = 2,
        backbone: str = "hgnn",
        dropout: float = 0.2,
    ):
        super().__init__()
        if backbone not in SUPPORTED_BACKBONES:
            raise ValueError(
                f"unsupported backbone {backbone!r}; choose {sorted(SUPPORTED_BACKBONES)}"
            )
        if num_layers < 1:
            raise ValueError("num_layers must be positive")
        reduction = SUPPORTED_BACKBONES[backbone]
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        if backbone == "allsettransformer":
            self.layers = nn.ModuleList(
                SetAttentionHypergraphConv(hidden_dim) for _ in range(num_layers)
            )
        else:
            self.layers = nn.ModuleList(
                HypergraphConv(
                    hidden_dim,
                    reduction=reduction,
                    edge_mlp=backbone in {"hnhn", "alldeepsets"},
                )
                for _ in range(num_layers)
            )
        self.dropout = nn.Dropout(dropout)
        self.output_dim = 2 * hidden_dim

    def forward(self, sample: HypergraphSample) -> torch.Tensor:
        x = torch.relu(self.input_projection(sample.x))
        for layer in self.layers:
            x = self.dropout(layer(x, sample.incidence, sample.num_hyperedges))
        return torch.cat((x.mean(dim=0), x.amax(dim=0)), dim=0)
