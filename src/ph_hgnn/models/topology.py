from __future__ import annotations

import torch
from torch import nn

from ph_hgnn.topology.persistence import PersistenceDiagrams, barcode_statistics


class PersistenceStatsEncoder(nn.Module):
    def __init__(self, output_dim: int = 64):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(10, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )
        self.output_dim = output_dim

    def forward(self, diagrams: PersistenceDiagrams) -> torch.Tensor:
        device = next(self.parameters()).device
        statistics = torch.cat(
            (
                barcode_statistics(diagrams.get(0, torch.empty(0, 2))),
                barcode_statistics(diagrams.get(1, torch.empty(0, 2))),
            )
        ).to(device)
        return self.network(statistics)


class PersistenceDeepSetEncoder(nn.Module):
    """Learnable permutation-invariant persistence diagram encoder."""

    def __init__(self, hidden_dim: int = 64, output_dim: int = 64):
        super().__init__()
        self.point_encoders = nn.ModuleList(
            nn.Sequential(
                nn.Linear(3, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
            )
            for _ in range(2)
        )
        self.readout = nn.Sequential(
            nn.Linear(4 * hidden_dim, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim

    def _encode_dimension(self, diagram: torch.Tensor, dim: int) -> torch.Tensor:
        device = next(self.parameters()).device
        diagram = diagram.to(device)
        if diagram.numel() == 0:
            return torch.zeros(2 * self.hidden_dim, device=device)
        lifetime = (diagram[:, 1] - diagram[:, 0]).unsqueeze(1)
        points = torch.cat((diagram, lifetime), dim=1)
        encoded = self.point_encoders[dim](points)
        return torch.cat((encoded.sum(dim=0), encoded.amax(dim=0)), dim=0)

    def forward(self, diagrams: PersistenceDiagrams) -> torch.Tensor:
        encoded = torch.cat(
            (
                self._encode_dimension(diagrams.get(0, torch.empty(0, 2)), 0),
                self._encode_dimension(diagrams.get(1, torch.empty(0, 2)), 1),
            ),
            dim=0,
        )
        return self.readout(encoded)


class PersistenceImageEncoder(nn.Module):
    """Gaussian persistence-image baseline followed by a learnable projection."""

    def __init__(self, resolution: int = 8, sigma: float = 0.12, output_dim: int = 64):
        super().__init__()
        axis = torch.linspace(0, 1, resolution)
        grid_x, grid_y = torch.meshgrid(axis, axis, indexing="ij")
        self.register_buffer("grid", torch.stack((grid_x, grid_y), dim=-1))
        self.sigma = sigma
        self.network = nn.Sequential(
            nn.Linear(2 * resolution * resolution, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )
        self.output_dim = output_dim

    def _image(self, diagram: torch.Tensor) -> torch.Tensor:
        device = self.grid.device
        if diagram.numel() == 0:
            return torch.zeros(self.grid.shape[:2], device=device)
        points = diagram.to(device)
        birth = points[:, 0]
        lifetime = (points[:, 1] - points[:, 0]).clamp_min(0)
        scale = torch.stack((birth, lifetime), dim=1).amax(dim=0).clamp_min(1)
        normalized = torch.stack((birth, lifetime), dim=1) / scale
        squared_distance = (
            self.grid.unsqueeze(0) - normalized[:, None, None, :]
        ).square().sum(dim=-1)
        weights = lifetime[:, None, None]
        return (weights * torch.exp(-squared_distance / (2 * self.sigma**2))).sum(dim=0)

    def forward(self, diagrams: PersistenceDiagrams) -> torch.Tensor:
        images = torch.cat(
            (
                self._image(diagrams.get(0, torch.empty(0, 2))).flatten(),
                self._image(diagrams.get(1, torch.empty(0, 2))).flatten(),
            )
        )
        return self.network(images)


class PersLayEncoder(nn.Module):
    """Compact learnable RBF implementation of a PersLay-style diagram layer."""

    def __init__(self, elements: int = 16, output_dim: int = 64):
        super().__init__()
        self.centers = nn.Parameter(torch.rand(2, elements, 2))
        self.log_scales = nn.Parameter(torch.zeros(2, elements))
        self.element_weights = nn.Parameter(torch.ones(2, elements))
        self.network = nn.Sequential(
            nn.Linear(2 * elements, output_dim),
            nn.ReLU(),
            nn.LayerNorm(output_dim),
        )
        self.output_dim = output_dim
        self.elements = elements

    def _encode(self, diagram: torch.Tensor, dim: int) -> torch.Tensor:
        device = self.centers.device
        if diagram.numel() == 0:
            return torch.zeros(self.elements, device=device)
        points = diagram.to(device)
        birth_lifetime = torch.stack(
            (points[:, 0], (points[:, 1] - points[:, 0]).clamp_min(0)), dim=1
        )
        scale = birth_lifetime.amax(dim=0).clamp_min(1)
        normalized = birth_lifetime / scale
        squared_distance = (
            normalized[:, None, :] - self.centers[dim][None, :, :]
        ).square().sum(dim=-1)
        response = torch.exp(
            -squared_distance * torch.exp(-2 * self.log_scales[dim])[None, :]
        )
        return (birth_lifetime[:, 1, None] * response).sum(dim=0) * self.element_weights[dim]

    def forward(self, diagrams: PersistenceDiagrams) -> torch.Tensor:
        return self.network(
            torch.cat((self._encode(diagrams.get(0, torch.empty(0, 2)), 0),
                       self._encode(diagrams.get(1, torch.empty(0, 2)), 1)))
        )
