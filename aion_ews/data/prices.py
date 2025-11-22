from __future__ import annotations

from typing import List, Tuple

import logging
from time import perf_counter

import numpy as np
import pandas as pd
import yfinance as yf

from ..utils.time import make_6h_grid


def _allocate_daily_volume_to_6h(vols_daily: pd.DataFrame, grid: pd.DatetimeIndex) -> pd.DataFrame:
    # Forward-fill daily volumes to 6H grid then divide by number of bins per calendar day
    vol6 = vols_daily.reindex(grid, method="ffill")
    counts = pd.Series(1, index=grid).groupby(grid.normalize()).transform("sum")
    vol6 = vol6.div(counts, axis=0)
    return vol6


def fetch_prices_make_ai_index(
    ai_tickers: List[str],
    baseline_tickers: List[str],
    start_date: str | None,
    end_date: str | None,
    resample: str = "6H",
) -> Tuple[pd.DataFrame, pd.Series]:
    logger = logging.getLogger(__name__)
    t0 = perf_counter()
    tickers = list(dict.fromkeys(list(ai_tickers) + list(baseline_tickers)))
    if not tickers:
        logger.warning("prices: no tickers provided")
        return pd.DataFrame(), pd.Series(dtype=float)

    df = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )
    logger.info(
        "prices: downloaded data for %d tickers, multiindex=%s (%.2fs)",
        len(tickers), isinstance(df.columns, pd.MultiIndex), perf_counter()-t0,
    )

    # Handle both single-ticker and multi-ticker structures
    closes = {}
    vols = {}
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            sub = df[t]
            if not isinstance(sub, pd.DataFrame) or sub.empty:
                continue
            if "Close" in sub:
                closes[t] = sub["Close"]
            elif "Adj Close" in sub:
                closes[t] = sub["Adj Close"]
            if "Volume" in sub:
                vols[t] = sub["Volume"]
    else:
        # single-ticker
        if "Close" in df:
            closes[tickers[0]] = df["Close"]
        elif "Adj Close" in df:
            closes[tickers[0]] = df["Adj Close"]
        if "Volume" in df:
            vols[tickers[0]] = df["Volume"]

    closes = pd.DataFrame(closes).sort_index()
    vols = pd.DataFrame(vols).reindex_like(closes)

    if closes.empty:
        grid = make_6h_grid(start_date, end_date)
        return pd.DataFrame(index=grid), pd.Series(index=grid, dtype=float)

    grid = pd.date_range(start=closes.index.min(), end=closes.index.max(), freq=resample.lower())
    close6 = closes.reindex(grid).ffill()
    vol6 = _allocate_daily_volume_to_6h(vols, grid)

    # AI equal-weight index
    ai_close = close6[ai_tickers].mean(axis=1, skipna=True)
    ai_index = 100.0 * (ai_close / ai_close.iloc[0]) if ai_close.notna().any() else pd.Series(index=grid, dtype=float)

    # Flatten columns
    close6 = close6.add_prefix("close_")
    vol6 = vol6.add_prefix("volume_")
    prices_df = pd.concat([close6, vol6], axis=1)

    prices_df.to_csv("data/intermediate/prices_6h.csv")
    ai_index.to_csv("data/intermediate/ai_index_6h.csv", header=["AI_INDEX"])
    logger.info(
        "prices: close6=%s, vol6=%s, ai_index=%d saved (%.2fs)",
        close6.shape, vol6.shape, len(ai_index), perf_counter()-t0,
    )

    return prices_df, ai_index
