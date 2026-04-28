"""Shared training loop for the GAN, plus seed control.

The same `train_gan` function is used for both the classical and the hybrid
quantum experiments — only the generator passed in differs.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    """Set seeds for python, numpy, and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainHistory:
    d_loss: list[float] = field(default_factory=list)
    g_loss: list[float] = field(default_factory=list)
    train_time_sec: float = 0.0


def make_dataloader(windows: np.ndarray, batch_size: int = 64) -> DataLoader:
    ds = TensorDataset(torch.tensor(windows, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


def train_gan(
    G: nn.Module,
    D: nn.Module,
    dataloader: DataLoader,
    *,
    latent_dim: int,
    epochs:     int   = 80,
    lr_g:       float = 2e-4,
    lr_d:       float = 2e-4,
    betas:      tuple[float, float] = (0.5, 0.999),
    device:     str | torch.device = "cpu",
    log_every:  int   = 10,
) -> TrainHistory:
    """Standard non-saturating GAN training loop.

    Works with any generator that accepts a (batch, latent_dim) tensor and
    returns a (batch, window) tensor — including quantum generators wrapped
    via PennyLane's TorchLayer.
    """
    G, D = G.to(device), D.to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=lr_g, betas=betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr_d, betas=betas)
    bce = nn.BCEWithLogitsLoss()

    history = TrainHistory()
    t0 = time.time()

    for epoch in range(epochs):
        d_losses, g_losses = [], []
        for (real,) in dataloader:
            real = real.to(device)
            bsz = real.size(0)
            ones  = torch.ones (bsz, 1, device=device)
            zeros = torch.zeros(bsz, 1, device=device)

            # ---- D step ----
            z = torch.randn(bsz, latent_dim, device=device)
            with torch.no_grad():
                fake = G(z)
            d_loss = bce(D(real), ones) + bce(D(fake), zeros)
            opt_D.zero_grad(); d_loss.backward(); opt_D.step()

            # ---- G step ----
            z = torch.randn(bsz, latent_dim, device=device)
            fake = G(z)
            g_loss = bce(D(fake), ones)
            opt_G.zero_grad(); g_loss.backward(); opt_G.step()

            d_losses.append(d_loss.item())
            g_losses.append(g_loss.item())

        history.d_loss.append(float(np.mean(d_losses)))
        history.g_loss.append(float(np.mean(g_losses)))

        if log_every and (epoch + 1) % log_every == 0:
            print(f"  epoch {epoch+1:3d}/{epochs}  D={history.d_loss[-1]:.3f}  "
                  f"G={history.g_loss[-1]:.3f}")

    history.train_time_sec = time.time() - t0
    return history


@torch.no_grad()
def generate(G: nn.Module, n_samples: int, latent_dim: int,
             device: str | torch.device = "cpu") -> tuple[np.ndarray, float]:
    """Sample `n_samples` windows from G and return (samples, elapsed_seconds)."""
    G.eval()
    t0 = time.time()
    z = torch.randn(n_samples, latent_dim, device=device)
    out = G(z).cpu().numpy()
    return out, time.time() - t0
