"""Data loading for the multi-asset SMI/DAX log-return GAN project.

Supports both univariate (single ticker) and multivariate (list of tickers).
Windows have shape (n_windows, window, n_assets); use SMIData.flatten() /
SMIData.unflatten() at the model boundary.
"""
from __future__ import annotations

import pathlib
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class SMIData:
    """Container for multi-asset log-return data, ready for GAN training.

    Shapes:
      train_windows : (n_train, window, n_assets) — scaled to ~[-1, 1] per asset
      test_windows  : (n_test,  window, n_assets)
      raw_returns   : (n_days,  n_assets) — unscaled log-returns
      scale         : (n_assets,)         — per-asset multiplier to recover units
      dates         : (n_days,)           — calendar dates aligned with raw_returns
      tickers       : list of ticker strings, in column order
    """
    train_windows: np.ndarray
    test_windows:  np.ndarray
    scale:         np.ndarray
    raw_returns:   np.ndarray
    dates:         pd.DatetimeIndex
    tickers:       list[str]
    window_size:   int

    @property
    def n_assets(self) -> int:
        return self.raw_returns.shape[1]

    def unscale(self, scaled_windows: np.ndarray) -> np.ndarray:
        """Convert scaled windows back to log-return units. Broadcasts over assets."""
        return scaled_windows * self.scale  # shape (..., window, n_assets) * (n_assets,)

    def flatten(self, windows: np.ndarray) -> np.ndarray:
        """(n, window, n_assets) -> (n, window * n_assets) for the GAN."""
        n = windows.shape[0]
        return windows.reshape(n, -1)

    def unflatten(self, flat: np.ndarray) -> np.ndarray:
        """(n, window * n_assets) -> (n, window, n_assets) for evaluation."""
        n = flat.shape[0]
        return flat.reshape(n, self.window_size, self.n_assets)


def download_prices(
    tickers:    list[str],
    start:      str = "2005-01-01",
    end:        str = "2025-01-01",
    cache_path: str | pathlib.Path | None = None,
) -> pd.DataFrame:
    """Download daily closes for one or more tickers and align by trading date.

    Returns a DataFrame with one column per ticker. Rows where ANY ticker is
    missing are dropped — important for multi-asset alignment.
    """
    cache_path = pathlib.Path(cache_path) if cache_path else None
    if cache_path and cache_path.exists():
        cached = pd.read_pickle(cache_path)
        # Use cache only if it has all requested tickers; otherwise re-download.
        if set(tickers).issubset(set(cached.columns)):
            return cached[tickers].dropna()

    import yfinance as yf
    raw = yf.download(tickers, start=start, end=end, progress=False, auto_adjust=True)
    # yfinance returns a multi-index for multi-ticker; for single, a flat one.
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"].copy()
    else:
        prices = raw[["Close"]].copy()
        prices.columns = tickers
    prices = prices[tickers].dropna()
    if prices.empty:
        raise RuntimeError(
            f"No aligned data for tickers={tickers}. Check connectivity / date range."
        )

    if cache_path:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        prices.to_pickle(cache_path)

    return prices


def log_returns(prices: pd.DataFrame) -> tuple[np.ndarray, pd.DatetimeIndex]:
    """Daily log-returns from a prices DataFrame; returns (values, dates)."""
    lr = np.log(prices / prices.shift(1)).dropna()
    return lr.values.astype(np.float32), lr.index


def make_windows(series: np.ndarray, window: int) -> np.ndarray:
    """Slice a 1-D or 2-D series into overlapping windows of length `window`.

    If `series` has shape (T,), returns (T-window+1, window).
    If `series` has shape (T, n_assets), returns (T-window+1, window, n_assets).
    """
    if len(series) < window:
        raise ValueError(f"Series of length {len(series)} too short for window {window}.")
    return np.stack([series[i:i + window] for i in range(len(series) - window + 1)])


def prepare_smi_data(
    tickers:     list[str] | str = "^SSMI",
    window:      int            = 20,
    train_frac:  float          = 0.8,
    scale_sigma: float          = 4.0,
    start:       str            = "2005-01-01",
    end:         str            = "2025-01-01",
    cache_path:  str | pathlib.Path | None = None,
) -> SMIData:
    """End-to-end multi-asset pipeline: download → align → log-returns → scale → split → window.

    `tickers` may be a single string (univariate) or a list (multivariate).
    Scaling is per-asset (each column divided by its own 4*sigma).
    """
    if isinstance(tickers, str):
        tickers = [tickers]

    prices         = download_prices(tickers, start=start, end=end, cache_path=cache_path)
    returns, dates = log_returns(prices)            # (T, n_assets), DatetimeIndex of length T

    # Per-asset scale factors so each column lands roughly in [-1, 1].
    scale  = scale_sigma * returns.std(axis=0)      # shape (n_assets,)
    scaled = returns / scale                        # broadcasts

    split = int(train_frac * len(scaled))
    train = scaled[:split]
    test  = scaled[split:]

    return SMIData(
        train_windows=make_windows(train, window),  # (n_train, window, n_assets)
        test_windows=make_windows(test, window),
        scale=scale,
        raw_returns=returns,
        dates=dates,
        tickers=tickers,
        window_size=window,
    )
