"""End-to-end experiment pipeline as a callable function.

This is the single source of truth for a "run":
  config -> data -> model -> training -> evaluation -> save artefacts.

Notebooks 01, 02, and 03 all delegate to `run_experiment` so they cannot
accidentally diverge in behaviour. Each call produces a deterministic folder
in `results/` that downstream analysis (notebook 03) can discover.
"""
from __future__ import annotations

import json
import pathlib
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np
import torch

from .data           import prepare_smi_data
from .models         import ClassicalGenerator, Discriminator, Critic, count_parameters
from .quantum_models import QuantumGenerator, count_quantum_parameters
from .training       import set_seed, make_dataloader, build_experiment, generate
from .evaluation     import build_report


@dataclass
class ExperimentConfig:
    """All knobs needed to reproduce a single run.

    The `folder_name()` is computed from these fields so the same config always
    lands in the same folder — useful for caching.

    For backwards compatibility with existing folders, the classical folder name
    only includes hidden/latent_dim suffixes when they DIFFER from defaults.
    Defaults: latent_dim=8, hidden=32 (these match the original notebook 01 setup).
    """
    family:        str             # 'classical' or 'quantum'
    variant:       str             # 'gan' or 'wgan_gp'
    tickers:       list[str]       # e.g. ['^SSMI', '^GDAXI'] (single-element list ok)
    seed:          int    = 42
    window:        int    = 20
    epochs:        int    = 80
    batch_size:    int    = 64
    latent_dim:    int    = 8       # used for classical only; quantum forces latent = n_qubits
    # Classical-only architecture knob:
    hidden:        int    = 32      # hidden width of classical generator MLP
    # Quantum-only:
    n_qubits:      int    = 6
    n_layers:      int    = 3

    def asset_tag(self) -> str:
        return '_'.join(t.replace('^', '') for t in self.tickers)

    def folder_name(self) -> str:
        if self.family == 'classical':
            # Only add suffix when off-default, so existing cached runs are preserved.
            suffix = ''
            if self.hidden != 32 or self.latent_dim != 8:
                suffix = f'_h{self.hidden}d{self.latent_dim}'
            return f'classical_{self.variant}_{self.asset_tag()}{suffix}'
        elif self.family == 'quantum':
            return f'quantum_{self.variant}_{self.asset_tag()}_q{self.n_qubits}L{self.n_layers}'
        else:
            raise ValueError(f"Unknown family: {self.family!r}")


def run_experiment(
    cfg:     ExperimentConfig,
    *,
    results_root: str | pathlib.Path,
    force_rerun:  bool = False,
    device:       str | None = None,
    verbose:      bool = True,
) -> dict:
    """Run one experiment end-to-end and save artefacts. Returns the metrics dict.

    If `force_rerun=False` (the default) and `results/<folder>/metrics.json`
    already exists, just load and return it without retraining.
    """
    results_root = pathlib.Path(results_root)
    out_dir = results_root / cfg.folder_name()
    metrics_path = out_dir / 'metrics.json'

    if metrics_path.is_file() and not force_rerun:
        if verbose:
            print(f'[cache hit]   {cfg.folder_name()}')
        with open(metrics_path) as f:
            return json.load(f)

    if verbose:
        print(f'[running]     {cfg.folder_name()}')
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    if device is None:
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ---- Data ----
    data = prepare_smi_data(
        tickers=cfg.tickers,
        window=cfg.window,
        cache_path=results_root / f'prices_{cfg.asset_tag()}.pkl',
    )

    # ---- Model ----
    adversary_cls = Discriminator if cfg.variant == 'gan' else Critic

    if cfg.family == 'classical':
        latent_dim = cfg.latent_dim
        exp = build_experiment(
            variant=cfg.variant, latent_dim=latent_dim, window=cfg.window,
            n_assets=data.n_assets,
            generator_cls=ClassicalGenerator,
            adversary_cls=adversary_cls,
            G_kwargs={'hidden': cfg.hidden},
        )
        n_params_G = count_parameters(exp.generator)
        extras_pq  = {'hidden': cfg.hidden}
    else:  # quantum
        latent_dim = cfg.n_qubits
        exp = build_experiment(
            variant=cfg.variant, latent_dim=latent_dim, window=cfg.window,
            n_assets=data.n_assets,
            generator_cls=QuantumGenerator,
            adversary_cls=adversary_cls,
            generator_kwargs={'n_qubits': cfg.n_qubits, 'n_layers': cfg.n_layers,
                              'window': cfg.window, 'n_assets': data.n_assets},
        )
        qparams = count_quantum_parameters(exp.generator)
        n_params_G = qparams['total_params']
        extras_pq  = {'n_qubits': cfg.n_qubits, 'n_layers': cfg.n_layers,
                      'quantum_params': qparams['quantum_params'],
                      'classical_params_in_G': qparams['classical_params']}

    n_params_A = count_parameters(exp.adversary)

    if verbose:
        print(f'  variant={exp.label}  assets={data.tickers}  '
              f'G params={n_params_G}  A params={n_params_A}')

    # ---- Train ----
    train_flat = data.flatten(data.train_windows)
    loader     = make_dataloader(train_flat, batch_size=cfg.batch_size)
    history = exp.train_fn(
        exp.generator, exp.adversary, loader,
        latent_dim=latent_dim, epochs=cfg.epochs, device=device,
        log_every=max(1, cfg.epochs // 4) if verbose else 0,
    )

    # ---- Generate & evaluate ----
    n_eval = len(data.test_windows)
    fake_flat_scaled, t_inf = generate(exp.generator, n_eval, latent_dim, device=device)
    fake_returns = data.unscale(data.unflatten(fake_flat_scaled))
    real_returns = data.unscale(data.test_windows)
    samples_per_sec = (n_eval * cfg.window * data.n_assets) / max(t_inf, 1e-6)

    report = build_report(
        real_windows=real_returns, fake_windows=fake_returns, tickers=data.tickers,
        n_params_G=n_params_G, n_params_D=n_params_A,
        train_time_sec=history.train_time_sec,
        inference_samples_per_sec=samples_per_sec,
        extras={'seed': cfg.seed, 'epochs': cfg.epochs, 'window': cfg.window,
                'latent_dim': latent_dim,
                'model': f'{cfg.family}_{cfg.variant}', **extras_pq},
    )

    # ---- Persist ----
    torch.save(exp.generator.state_dict(), out_dir / 'generator.pt')
    np.save(out_dir / 'fake_returns.npy', fake_returns)
    np.save(out_dir / 'real_returns_test.npy', real_returns)
    np.save(out_dir / 'scale.npy', data.scale)
    with open(metrics_path, 'w') as f:
        json.dump(report, f, indent=2, default=float)

    if verbose:
        print(f'  saved -> {out_dir}\n')
    return report


def run_many(
    configs:     list[ExperimentConfig],
    results_root: str | pathlib.Path,
    *,
    force_rerun:  bool = False,
    device:       str | None = None,
    verbose:      bool = True,
) -> list[dict]:
    """Run a batch of experiments. Caches per-config; total time = sum of new runs only."""
    return [
        run_experiment(c, results_root=results_root, force_rerun=force_rerun,
                       device=device, verbose=verbose)
        for c in configs
    ]
