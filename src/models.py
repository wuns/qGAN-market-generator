"""Model architectures.

The generator outputs a flat vector of size `window * n_assets`; the data layer
reshapes to (window, n_assets) for evaluation. The discriminator/critic operates
on the same flat vector — so the training loop is identical for univariate and
multivariate cases.
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClassicalGenerator(nn.Module):
    """Tiny MLP generator: latent vector -> flattened return window.

    Output shape: (batch, window * n_assets), in tanh range.
    """

    def __init__(self, latent_dim: int = 8, window: int = 20,
                 n_assets: int = 1, hidden: int = 32):
        super().__init__()
        self.latent_dim = latent_dim
        self.window     = window
        self.n_assets   = n_assets
        self.out_dim    = window * n_assets
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden),     nn.LeakyReLU(0.2),
            nn.Linear(hidden, self.out_dim), nn.Tanh(),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)


class Discriminator(nn.Module):
    """MLP discriminator returning logits (use BCEWithLogitsLoss).

    Operates on flat windows of size window * n_assets.
    """

    def __init__(self, window: int = 20, n_assets: int = 1, hidden: int = 64):
        super().__init__()
        in_dim = window * n_assets
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden // 2), nn.LeakyReLU(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Critic(nn.Module):
    """WGAN critic. Same architecture as Discriminator (intentional, for fair
    comparison), but used with Wasserstein loss + gradient penalty.

    No batch norm anywhere — would conflict with per-sample gradients required
    by the gradient penalty.
    """

    def __init__(self, window: int = 20, n_assets: int = 1, hidden: int = 64):
        super().__init__()
        in_dim = window * n_assets
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden // 2), nn.LeakyReLU(0.2),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
