"""Evaluation metrics and plots — shared by both notebooks.

Every metric is computed on flat (1-D) arrays of log-returns. ACF plots
are computed per-window then averaged.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


@dataclass
class StatSummary:
    name:     str
    n:        int
    mean:     float
    std:      float
    skew:     float
    kurtosis: float    # excess kurtosis (Fisher)
    minimum:  float
    maximum:  float

    def to_dict(self) -> dict:
        return asdict(self)


def summarise(returns: np.ndarray, name: str) -> StatSummary:
    x = np.asarray(returns).ravel()
    return StatSummary(
        name=name,
        n=int(x.size),
        mean=float(x.mean()),
        std=float(x.std()),
        skew=float(stats.skew(x)),
        kurtosis=float(stats.kurtosis(x)),
        minimum=float(x.min()),
        maximum=float(x.max()),
    )


def ks_distance(real: np.ndarray, fake: np.ndarray) -> tuple[float, float]:
    """Two-sample Kolmogorov–Smirnov: returns (statistic, p-value)."""
    s, p = stats.ks_2samp(np.asarray(real).ravel(), np.asarray(fake).ravel())
    return float(s), float(p)


def autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Sample autocorrelation function up to `max_lag`, including lag 0."""
    x = np.asarray(x) - np.mean(x)
    var = np.var(x)
    if var == 0:
        return np.full(max_lag + 1, np.nan)
    out = [1.0]
    for k in range(1, max_lag + 1):
        out.append(float(np.mean(x[:-k] * x[k:]) / var))
    return np.array(out)


def average_acf(windows: np.ndarray, max_lag: int) -> np.ndarray:
    """Mean ACF across all rows (windows) of a 2-D array."""
    return np.mean([autocorr(w, max_lag) for w in windows], axis=0)


# -------- Plots --------

def plot_distributions(real: np.ndarray, fake: np.ndarray, ax=None):
    real_flat = np.asarray(real).ravel()
    fake_flat = np.asarray(fake).ravel()
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))

    bins = np.linspace(min(real_flat.min(), fake_flat.min()),
                       max(real_flat.max(), fake_flat.max()), 80)
    ax[0].hist(real_flat, bins=bins, alpha=0.6, density=True, label="real")
    ax[0].hist(fake_flat, bins=bins, alpha=0.6, density=True, label="fake")
    ax[0].set_title("Return distribution")
    ax[0].set_xlabel("log-return"); ax[0].legend()

    qs = np.linspace(0.01, 0.99, 99)
    ax[1].plot(np.quantile(real_flat, qs), np.quantile(fake_flat, qs), "o-", ms=3)
    lim = max(np.abs(real_flat).max(), np.abs(fake_flat).max())
    ax[1].plot([-lim, lim], [-lim, lim], "k--", lw=0.7)
    ax[1].set_xlabel("real quantile"); ax[1].set_ylabel("fake quantile"); ax[1].set_title("Q-Q")
    return ax


def plot_acf_comparison(real_windows: np.ndarray, fake_windows: np.ndarray,
                        max_lag: int = 10, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.5))

    acf_real    = average_acf(real_windows,        max_lag)
    acf_fake    = average_acf(fake_windows,        max_lag)
    acf_real_sq = average_acf(real_windows ** 2,   max_lag)
    acf_fake_sq = average_acf(fake_windows ** 2,   max_lag)
    lags = np.arange(max_lag + 1)

    ax[0].stem(lags - 0.15, acf_real, linefmt="C0-", markerfmt="C0o", basefmt=" ", label="real")
    ax[0].stem(lags + 0.15, acf_fake, linefmt="C1-", markerfmt="C1o", basefmt=" ", label="fake")
    ax[0].axhline(0, color="k", lw=0.4); ax[0].set_title("ACF of returns"); ax[0].legend()

    ax[1].stem(lags - 0.15, acf_real_sq, linefmt="C0-", markerfmt="C0o", basefmt=" ", label="real")
    ax[1].stem(lags + 0.15, acf_fake_sq, linefmt="C1-", markerfmt="C1o", basefmt=" ", label="fake")
    ax[1].axhline(0, color="k", lw=0.4)
    ax[1].set_title("ACF of squared returns (vol clustering)"); ax[1].legend()
    return ax


def plot_sample_paths(real_windows: np.ndarray, fake_windows: np.ndarray, n: int = 4):
    fig, ax = plt.subplots(2, n, figsize=(3.2 * n, 5), sharey=True)
    for i in range(n):
        ax[0, i].plot(real_windows[i]); ax[0, i].set_title(f"Real #{i}")
        ax[1, i].plot(fake_windows[i]); ax[1, i].set_title(f"Fake #{i}")
    for a in ax.flat:
        a.axhline(0, color="k", lw=0.3)
    ax[0, 0].set_ylabel("log-return"); ax[1, 0].set_ylabel("log-return")
    return fig


def build_report(
    *,
    real_returns: np.ndarray,
    fake_returns: np.ndarray,
    n_params_G:   int,
    n_params_D:   int,
    train_time_sec: float,
    inference_samples_per_sec: float,
    extras: dict | None = None,
) -> dict:
    """Single dict containing everything you'll write to metrics.json."""
    real_flat = np.asarray(real_returns).ravel()
    fake_flat = np.asarray(fake_returns).ravel()
    ks_stat, ks_p = ks_distance(real_flat, fake_flat)
    real_summary = summarise(real_flat, "real")
    fake_summary = summarise(fake_flat, "fake")

    report = {
        "real_summary":             real_summary.to_dict(),
        "fake_summary":             fake_summary.to_dict(),
        "ks_statistic":             round(ks_stat, 5),
        "ks_pvalue":                ks_p,
        "n_params_generator":       n_params_G,
        "n_params_discriminator":   n_params_D,
        "training_time_sec":        round(float(train_time_sec), 2),
        "inference_samples_per_sec": round(float(inference_samples_per_sec), 1),
    }
    if extras:
        report.update(extras)
    return report
