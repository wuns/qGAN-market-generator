"""Shared training loops for both vanilla GAN and WGAN-GP.

A single dispatcher `build_experiment(variant)` returns the right pieces so the
notebook only needs to flip one config flag.

Note: in this file the "discriminator" (`D`) and the WGAN "critic" (`C`) are
both `nn.Module` instances with identical architecture (in our case). The
difference is purely in the loss they're trained with.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Callable, Literal, NamedTuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


Variant = Literal["gan", "wgan_gp"]


# --------------------------------------------------------------------------- #
# Common utilities                                                            #
# --------------------------------------------------------------------------- #

def set_seed(seed: int) -> None:
    """Set seeds for python, numpy, and torch (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@dataclass
class TrainHistory:
    d_loss: list[float] = field(default_factory=list)   # discriminator OR critic loss
    g_loss: list[float] = field(default_factory=list)
    train_time_sec: float = 0.0
    variant: str = "gan"


def make_dataloader(windows: np.ndarray, batch_size: int = 64) -> DataLoader:
    ds = TensorDataset(torch.tensor(windows, dtype=torch.float32))
    return DataLoader(ds, batch_size=batch_size, shuffle=True, drop_last=True)


# --------------------------------------------------------------------------- #
# Vanilla GAN                                                                 #
# --------------------------------------------------------------------------- #

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
    """Standard non-saturating GAN training loop with BCEWithLogitsLoss."""
    G, D = G.to(device), D.to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=lr_g, betas=betas)
    opt_D = torch.optim.Adam(D.parameters(), lr=lr_d, betas=betas)
    bce = nn.BCEWithLogitsLoss()

    history = TrainHistory(variant="gan")
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


# --------------------------------------------------------------------------- #
# Wasserstein GAN with Gradient Penalty (WGAN-GP)                             #
# --------------------------------------------------------------------------- #

def gradient_penalty(critic: nn.Module, real: torch.Tensor, fake: torch.Tensor,
                     device: str | torch.device) -> torch.Tensor:
    """Two-sided GP: encourages |grad_x C(x)|_2 -> 1 along lines between
    real and fake samples (1-Lipschitz constraint)."""
    bsz = real.size(0)
    alpha = torch.rand(bsz, 1, device=device)
    interp = (alpha * real + (1 - alpha) * fake).requires_grad_(True)
    score = critic(interp)
    grads = torch.autograd.grad(
        outputs=score, inputs=interp,
        grad_outputs=torch.ones_like(score),
        create_graph=True, retain_graph=True,
    )[0]
    grad_norm = grads.norm(2, dim=1)
    return ((grad_norm - 1) ** 2).mean()


def train_wgan_gp(
    G: nn.Module,
    C: nn.Module,
    dataloader: DataLoader,
    *,
    latent_dim: int,
    epochs:     int   = 80,
    lr_g:       float = 1e-4,
    lr_c:       float = 1e-4,
    betas:      tuple[float, float] = (0.5, 0.9),
    n_critic:   int   = 5,
    gp_weight:  float = 10.0,
    device:     str | torch.device = "cpu",
    log_every:  int   = 10,
) -> TrainHistory:
    """WGAN-GP training loop.

    Loss values are NOT bounded and don't have textbook equilibria like ln(2).
    Use sample quality / metrics to judge convergence.
    """
    G, C = G.to(device), C.to(device)
    opt_G = torch.optim.Adam(G.parameters(), lr=lr_g, betas=betas)
    opt_C = torch.optim.Adam(C.parameters(), lr=lr_c, betas=betas)

    history = TrainHistory(variant="wgan_gp")
    t0 = time.time()

    for epoch in range(epochs):
        c_losses, g_losses = [], []
        for batch_idx, (real,) in enumerate(dataloader):
            real = real.to(device)
            bsz = real.size(0)

            # ---- Critic step ----
            z = torch.randn(bsz, latent_dim, device=device)
            with torch.no_grad():
                fake = G(z)
            c_real = C(real).mean()
            c_fake = C(fake).mean()
            gp = gradient_penalty(C, real, fake, device)
            c_loss = -(c_real - c_fake) + gp_weight * gp
            opt_C.zero_grad(); c_loss.backward(); opt_C.step()
            c_losses.append(c_loss.item())

            # ---- Generator step (every n_critic batches) ----
            if (batch_idx + 1) % n_critic == 0:
                z = torch.randn(bsz, latent_dim, device=device)
                fake = G(z)
                g_loss = -C(fake).mean()
                opt_G.zero_grad(); g_loss.backward(); opt_G.step()
                g_losses.append(g_loss.item())

        history.d_loss.append(float(np.mean(c_losses)))
        history.g_loss.append(float(np.mean(g_losses)) if g_losses else float("nan"))

        if log_every and (epoch + 1) % log_every == 0:
            print(f"  epoch {epoch+1:3d}/{epochs}  C={history.d_loss[-1]:.3f}  "
                  f"G={history.g_loss[-1]:.3f}")

    history.train_time_sec = time.time() - t0
    return history


# --------------------------------------------------------------------------- #
# Dispatcher: one switch, the rest of the notebook stays the same             #
# --------------------------------------------------------------------------- #

class Experiment(NamedTuple):
    """Bundle of a generator, an adversary (D or C), and a configured train fn.

    The train function is already curried with all variant-specific kwargs;
    you only need to pass the runtime args (G, adversary, dataloader,
    latent_dim, epochs, device).
    """
    generator:    nn.Module
    adversary:    nn.Module
    train_fn:     Callable[..., TrainHistory]
    label:        str
    adversary_role: str   # 'discriminator' or 'critic'


def build_experiment(
    variant:    Variant,
    *,
    latent_dim: int,
    window:     int,
    n_assets:   int = 1,
    generator_cls,
    adversary_cls,        # Discriminator or Critic — same architecture in this project
    generator_kwargs: dict | None = None,
    **model_kwargs,
) -> Experiment:
    """Construct G, adversary, and the right training function for the chosen variant.

    `generator_kwargs` lets the caller pass kwargs specific to a generator class
    (e.g., n_qubits and n_layers for QuantumGenerator). For the classical generator,
    leave it None and latent_dim/window/n_assets will be passed positionally.

    Usage:

        # Classical:
        exp = build_experiment(
            variant=MODEL_VARIANT,
            latent_dim=LATENT_DIM, window=WINDOW, n_assets=N_ASSETS,
            generator_cls=ClassicalGenerator,
            adversary_cls=Discriminator,
        )

        # Quantum:
        exp = build_experiment(
            variant=MODEL_VARIANT,
            latent_dim=N_QUBITS, window=WINDOW, n_assets=N_ASSETS,
            generator_cls=QuantumGenerator,
            adversary_cls=Discriminator,
            generator_kwargs={'n_qubits': N_QUBITS, 'n_layers': N_LAYERS},
        )
    """
    if variant not in ("gan", "wgan_gp"):
        raise ValueError(f"Unknown variant: {variant!r}. Choose 'gan' or 'wgan_gp'.")

    if generator_kwargs is None:
        # Classical generator: pass latent_dim, window, n_assets directly.
        G = generator_cls(latent_dim=latent_dim, window=window, n_assets=n_assets,
                          **model_kwargs.get("G_kwargs", {}))
    else:
        # Quantum (or any custom) generator: caller supplies all kwargs explicitly.
        # latent_dim / window / n_assets only added if not already in generator_kwargs.
        gk = dict(generator_kwargs)
        gk.setdefault("window", window)
        gk.setdefault("n_assets", n_assets)
        G = generator_cls(**gk)
    A = adversary_cls(window=window, n_assets=n_assets,
                      **model_kwargs.get("A_kwargs", {}))

    if variant == "gan":
        return Experiment(G, A, train_fn=train_gan, label="vanilla GAN",
                          adversary_role="discriminator")
    else:
        return Experiment(G, A, train_fn=train_wgan_gp, label="WGAN-GP",
                          adversary_role="critic")


# --------------------------------------------------------------------------- #
# Inference                                                                   #
# --------------------------------------------------------------------------- #

@torch.no_grad()
def generate(G: nn.Module, n_samples: int, latent_dim: int,
             device: str | torch.device = "cpu") -> tuple[np.ndarray, float]:
    """Sample n_samples windows from G; return (samples, elapsed_seconds)."""
    G.eval()
    t0 = time.time()
    z = torch.randn(n_samples, latent_dim, device=device)
    out = G(z).cpu().numpy()
    return out, time.time() - t0
