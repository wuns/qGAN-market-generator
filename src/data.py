"""Data loading for the SMI log-return GAN project.

All functions are pure and deterministic given their inputs (no global state).
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SMIData:
    """Container for SMI log-return data, ready for GAN training."""
    train_windows: np.ndarray   # shape (n_train, window) — scaled to ~[-1, 1]
    test_windows:  np.ndarray   # shape (n_test, window)  — scaled to ~[-1, 1]
    scale:         float        # multiplier to recover log-returns from scaled values
    raw_returns:   np.ndarray   # full unscaled log-return series
    window_size:   int

    def unscale(self, scaled_windows: np.ndarray) -> np.ndarray:
        """Convert scaled windows back to log-return units."""
        return scaled_windows * self.scale


def download_smi_prices(
    start: str = "2005-01-01",
    end:   str = "2025-01-01",
    cache_path: str | pathlib.Path | None = None,
) -> pd.Series:
    """Download SMI daily closes from Yahoo Finance.

    Caches to disk if `cache_path` is given, so repeated runs (and your live demo)
    don't depend on Yahoo being reachable.
    """
    cache_path = pathlib.Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        return pd.read_pickle(cache_path)

    import yfinance as yf
    raw = yf.download("^SSMI", start=start, end=end, progress=False, auto_adjust=True)
    if raw.empty:
        raise RuntimeError(
            "Yahoo returned no data for ^SSMI. Check connection or date range."
        )
    prices = raw["Close"].dropna().squeeze()
    prices.name = "SMI_close"

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_pickle(cache_path)

    return prices


def log_returns(prices: pd.Series) -> np.ndarray:
    """Compute daily log-returns from a price series."""
    return np.log(prices / prices.shift(1)).dropna().values.astype(np.float32)


def make_windows(series: np.ndarray, window: int) -> np.ndarray:
    """Slice a 1-D series into overlapping windows of length `window`."""
    if len(series) < window:
        raise ValueError(f"Series of length {len(series)} too short for window {window}.")
    return np.stack([series[i:i + window] for i in range(len(series) - window + 1)])


def prepare_smi_data(
    window:      int   = 20,
    train_frac:  float = 0.8,
    scale_sigma: float = 4.0,
    start:       str   = "2005-01-01",
    end:         str   = "2025-01-01",
    cache_path:  str | pathlib.Path | None = None,
) -> SMIData:
    """End-to-end pipeline: download → log-returns → scale → split → window.

    Scaling: divide by `scale_sigma * std(returns)` so most values land in [-1, 1],
    matching the tanh output range of the generator.

    Split: chronological (NOT random) — important for time series.
    """
    prices  = download_smi_prices(start=start, end=end, cache_path=cache_path)
    returns = log_returns(prices)

    scale  = scale_sigma * float(returns.std())
    scaled = returns / scale

    split = int(train_frac * len(scaled))
    train = scaled[:split]
    test  = scaled[split:]

    return SMIData(
        train_windows=make_windows(train, window),
        test_windows=make_windows(test, window),
        scale=scale,
        raw_returns=returns,
        window_size=window,
    )
