"""Model architectures for the classical GAN baseline.

The discriminator here will be reused unchanged in the quantum GAN notebook,
so that the only difference between the two experiments is the generator.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClassicalGenerator(nn.Module):
    """Tiny MLP generator: latent vector -> log-return window (in tanh range).

    Kept deliberately small so the comparison against a small quantum generator is fair.
    """

    def __init__(self, latent_dim: int = 8, window: int = 20, hidden: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.window     = window
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),     nn.LeakyReLU(0.2),
            nn.Linear(hidden, window),     #nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """MLP discriminator returning logits (use BCEWithLogitsLoss).

    Reused identically by both classical and quantum experiments — that's the point.
    """

    def __init__(self, window: int = 20, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden // 2), nn.LeakyReLU(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    """WGAN critic. Same architecture as Discriminator (intentional, for fair
    comparison), but used with Wasserstein loss + gradient penalty rather than BCE.

    No batch norm anywhere — would conflict with the per-sample gradients required
    by the gradient penalty.
    """

    def __init__(self, window: int = 20, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(window, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden // 2), nn.LeakyReLU(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
