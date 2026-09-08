"""Deep learning model: cell-state encoder + gene embedding(s) -> predicted expression delta."""

from __future__ import annotations

import torch
from torch import nn


class CellEncoder(nn.Module):
    """Encodes control-cell expression into a latent cell-state vector."""

    def __init__(self, n_genes: int, hidden_dim: int = 256, latent_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PerturbationModel(nn.Module):
    """Combines cell-state latent + perturbation embedding(s) to predict expression delta."""

    def __init__(self, n_genes: int, embed_dim: int = 64, hidden_dim: int = 256):
        super().__init__()
        self.cell_encoder = CellEncoder(n_genes, hidden_dim=hidden_dim, latent_dim=embed_dim)
        self.decoder = nn.Sequential(
            nn.Linear(embed_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, n_genes),
        )

    def forward(self, control_expression: torch.Tensor, perturbation_embedding: torch.Tensor) -> torch.Tensor:
        cell_latent = self.cell_encoder(control_expression)
        combined = torch.cat([cell_latent, perturbation_embedding], dim=-1)
        return self.decoder(combined)


def load_model(path: str) -> tuple["PerturbationModel", dict]:
    """Load a checkpoint saved by scripts/05_train_model.py."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = PerturbationModel(n_genes=checkpoint["n_hvg"], embed_dim=checkpoint["pair_dim"], hidden_dim=256)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model, checkpoint
