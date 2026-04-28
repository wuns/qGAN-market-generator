"""Evaluation metrics and plots — supports multi-asset windows.

All input arrays expected with shape (n_windows, window, n_assets).
For univariate, n_assets = 1 and the second axis is squeezed transparently.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


# ----------------------- Per-asset summary -----------------------

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
    """Stats over flattened returns (works on any shape — flattens internally)."""
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


def per_asset_summaries(windows: np.ndarray, tickers: list[str], suffix: str
                        ) -> dict[str, dict]:
    """windows shape (n, window, n_assets). Returns one summary per asset."""
    if windows.ndim == 2:
        windows = windows[..., None]
    out = {}
    for i, t in enumerate(tickers):
        out[t] = summarise(windows[:, :, i], f"{t}_{suffix}").to_dict()
    return out


def ks_distance(real: np.ndarray, fake: np.ndarray) -> tuple[float, float]:
    s, p = stats.ks_2samp(np.asarray(real).ravel(), np.asarray(fake).ravel())
    return float(s), float(p)


# ----------------------- Autocorrelation -----------------------

def autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    x = np.asarray(x) - np.mean(x)
    var = np.var(x)
    if var == 0:
        return np.full(max_lag + 1, np.nan)
    out = [1.0]
    for k in range(1, max_lag + 1):
        out.append(float(np.mean(x[:-k] * x[k:]) / var))
    return np.array(out)


def average_acf(windows: np.ndarray, max_lag: int) -> np.ndarray:
    """Mean ACF across rows (windows) of a 2-D array (n_windows, window)."""
    return np.mean([autocorr(w, max_lag) for w in windows], axis=0)


# ----------------------- Cross-asset correlation -----------------------

def correlation_matrix(windows: np.ndarray) -> np.ndarray:
    """Pearson correlation between assets, computed on flattened windows.

    windows shape (n, window, n_assets). Returns (n_assets, n_assets) matrix.
    """
    if windows.ndim == 2:
        windows = windows[..., None]
    n, w, k = windows.shape
    flat = windows.reshape(n * w, k)   # treat all (window, asset) values as samples
    return np.corrcoef(flat, rowvar=False)


def correlation_error(real: np.ndarray, fake: np.ndarray) -> dict:
    """Frobenius norm of (corr_real - corr_fake), plus the matrices themselves."""
    Cr = correlation_matrix(real)
    Cf = correlation_matrix(fake)
    return {
        "real_corr":     Cr.tolist(),
        "fake_corr":     Cf.tolist(),
        "frobenius_err": float(np.linalg.norm(Cr - Cf)),
    }


# ----------------------- Plots -----------------------

def _ensure_3d(w):
    return w if w.ndim == 3 else w[..., None]


def plot_distributions(real: np.ndarray, fake: np.ndarray, tickers: list[str], ax=None):
    """One row per asset: histogram + Q-Q plot."""
    real = _ensure_3d(real); fake = _ensure_3d(fake)
    n_assets = real.shape[2]
    if ax is None:
        fig, ax = plt.subplots(n_assets, 2, figsize=(11, 3.5 * n_assets), squeeze=False)

    for i, ticker in enumerate(tickers):
        rf = real[:, :, i].ravel()
        ff = fake[:, :, i].ravel()
        bins = np.linspace(min(rf.min(), ff.min()), max(rf.max(), ff.max()), 80)
        ax[i, 0].hist(rf, bins=bins, alpha=0.6, density=True, label="real")
        ax[i, 0].hist(ff, bins=bins, alpha=0.6, density=True, label="fake")
        ax[i, 0].set_title(f"{ticker}: return distribution"); ax[i, 0].legend()

        qs = np.linspace(0.01, 0.99, 99)
        ax[i, 1].plot(np.quantile(rf, qs), np.quantile(ff, qs), "o-", ms=3)
        lim = max(np.abs(rf).max(), np.abs(ff).max())
        ax[i, 1].plot([-lim, lim], [-lim, lim], "k--", lw=0.7)
        ax[i, 1].set_title(f"{ticker}: Q-Q")
        ax[i, 1].set_xlabel("real quantile"); ax[i, 1].set_ylabel("fake quantile")
    return ax


def plot_acf_comparison(real: np.ndarray, fake: np.ndarray, tickers: list[str],
                        max_lag: int = 10, ax=None):
    """One row per asset: ACF of returns, ACF of squared returns."""
    real = _ensure_3d(real); fake = _ensure_3d(fake)
    n_assets = real.shape[2]
    if ax is None:
        fig, ax = plt.subplots(n_assets, 2, figsize=(11, 3.5 * n_assets), squeeze=False)

    lags = np.arange(max_lag + 1)
    for i, ticker in enumerate(tickers):
        r = real[:, :, i]; f = fake[:, :, i]
        acf_r  = average_acf(r,      max_lag); acf_f  = average_acf(f,      max_lag)
        acf_r2 = average_acf(r ** 2, max_lag); acf_f2 = average_acf(f ** 2, max_lag)

        ax[i, 0].stem(lags - 0.15, acf_r, linefmt="C0-", markerfmt="C0o", basefmt=" ", label="real")
        ax[i, 0].stem(lags + 0.15, acf_f, linefmt="C1-", markerfmt="C1o", basefmt=" ", label="fake")
        ax[i, 0].axhline(0, color="k", lw=0.4)
        ax[i, 0].set_title(f"{ticker}: ACF of returns"); ax[i, 0].legend()

        ax[i, 1].stem(lags - 0.15, acf_r2, linefmt="C0-", markerfmt="C0o", basefmt=" ", label="real")
        ax[i, 1].stem(lags + 0.15, acf_f2, linefmt="C1-", markerfmt="C1o", basefmt=" ", label="fake")
        ax[i, 1].axhline(0, color="k", lw=0.4)
        ax[i, 1].set_title(f"{ticker}: ACF of squared returns")
        ax[i, 1].legend()
    return ax


def plot_sample_paths(real: np.ndarray, fake: np.ndarray, tickers: list[str], n: int = 4):
    """Per-asset sample paths: 2 rows (real / fake) x n columns. One figure per asset."""
    real = _ensure_3d(real); fake = _ensure_3d(fake)
    figs = []
    for i, ticker in enumerate(tickers):
        fig, ax = plt.subplots(2, n, figsize=(3.2 * n, 5), sharey=True)
        for j in range(n):
            ax[0, j].plot(real[j, :, i]); ax[0, j].set_title(f"{ticker} real #{j}")
            ax[1, j].plot(fake[j, :, i]); ax[1, j].set_title(f"{ticker} fake #{j}")
        for a in ax.flat:
            a.axhline(0, color="k", lw=0.3)
        ax[0, 0].set_ylabel("log-return"); ax[1, 0].set_ylabel("log-return")
        figs.append(fig)
    return figs


def plot_correlation_comparison(real: np.ndarray, fake: np.ndarray, tickers: list[str]):
    """Side-by-side heatmap of cross-asset correlation matrices.

    Only meaningful for n_assets >= 2.
    """
    real = _ensure_3d(real); fake = _ensure_3d(fake)
    if real.shape[2] < 2:
        return None
    Cr = correlation_matrix(real)
    Cf = correlation_matrix(fake)

    fig, ax = plt.subplots(1, 3, figsize=(13, 4))
    for a, M, title in zip(ax, [Cr, Cf, Cr - Cf],
                           ["real correlation", "fake correlation", "real - fake"]):
        im = a.imshow(M, vmin=-1, vmax=1, cmap="RdBu_r")
        a.set_xticks(range(len(tickers))); a.set_yticks(range(len(tickers)))
        a.set_xticklabels(tickers, rotation=45); a.set_yticklabels(tickers)
        a.set_title(title)
        for i in range(len(tickers)):
            for j in range(len(tickers)):
                a.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                       color="black", fontsize=9)
        plt.colorbar(im, ax=a, fraction=0.046)
    plt.tight_layout()
    return fig


# ----------------------- Final report -----------------------

def build_report(
    *,
    real_windows: np.ndarray,
    fake_windows: np.ndarray,
    tickers:      list[str],
    n_params_G:   int,
    n_params_D:   int,
    train_time_sec: float,
    inference_samples_per_sec: float,
    extras: dict | None = None,
) -> dict:
    """Single dict containing everything — multivariate-aware."""
    real = _ensure_3d(real_windows); fake = _ensure_3d(fake_windows)
    ks_stat, ks_p = ks_distance(real, fake)

    report = {
        "tickers":                tickers,
        "n_assets":               int(real.shape[2]),
        "real_per_asset":         per_asset_summaries(real, tickers, "real"),
        "fake_per_asset":         per_asset_summaries(fake, tickers, "fake"),
        "ks_statistic_overall":   round(ks_stat, 5),
        "ks_pvalue_overall":      ks_p,
        "n_params_generator":     n_params_G,
        "n_params_discriminator": n_params_D,
        "training_time_sec":      round(float(train_time_sec), 2),
        "inference_samples_per_sec": round(float(inference_samples_per_sec), 1),
    }
    if real.shape[2] >= 2:
        report["correlation"] = correlation_error(real, fake)
    if extras:
        report.update(extras)
    return report
