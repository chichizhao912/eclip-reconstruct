"""Selected one-dimensional encoder-decoder architecture."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class CoverageAutoencoder(nn.Module):
    """Predict a 300 nt IP profile from sequence, source IP, and SMInput."""

    def __init__(self, dropout: float = 0.10, input_channels: int = 6):
        super().__init__()
        if input_channels not in {2, 5, 6}:
            raise ValueError("input_channels must be 2, 5, or 6")
        self.input_channels = input_channels
        self.encoder = nn.Sequential(
            nn.Conv1d(input_channels, 32, 7, padding=3),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 5, padding=2),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.MaxPool1d(2),
            nn.Conv1d(64, 64, 5, padding=2),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.dec1 = nn.Conv1d(64, 32, 5, padding=2)
        self.dec2 = nn.Conv1d(32, 16, 5, padding=2)
        self.out = nn.Conv1d(16, 1, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if x.ndim != 3 or x.shape[1:] != (self.input_channels, 300):
            raise ValueError(f"input must have shape [batch, {self.input_channels}, 300]")
        latent = self.encoder(x)
        decoded = F.interpolate(latent, scale_factor=2, mode="linear", align_corners=False)
        decoded = F.gelu(self.dec1(decoded))
        decoded = F.interpolate(decoded, size=300, mode="linear", align_corners=False)
        decoded = F.gelu(self.dec2(decoded))
        return F.softplus(self.out(decoded)), latent


def model_input(sequence: torch.Tensor, source_ip: torch.Tensor,
                sminput: torch.Tensor) -> torch.Tensor:
    """Arrange the six channels in the order used for model training."""
    return torch.cat((sequence, source_ip[:, None, :], sminput[:, None, :]), dim=1)

