from __future__ import annotations

from typing import List, Tuple

import logging
from time import perf_counter

import numpy as np
import pandas as pd


BIN_PER_DAY = 4  # 6H grid
WIN7 = 7 * BIN_PER_DAY
WIN30 = 30 * BIN_PER_DAY


def _align(df: pd.DataFrame | pd.Series, idx: pd.DatetimeIndex, fill: float | None = None) -> pd.DataFrame:
    if df is None or len(df) == 0:
        return pd.DataFrame(index=idx)
    df = pd.DataFrame(df).reindex(idx)
    if fill is None:
        return df.ffill()
    return df.fillna(fill)


def _pct_change(s: pd.Series) -> pd.Series:
    return s.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def _log_ret(s: pd.Series) -> pd.Series:
    return np.log(s).diff().replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_features(
    prices_df: pd.DataFrame,
    ai_index: pd.Series,
    reddit_df: pd.DataFrame,
    news_df: pd.DataFrame,
    gdelt_df: pd.DataFrame,
    hnews_df: pd.DataFrame | None = None,
    arxiv_df: pd.DataFrame | None = None,
    resample: str = "6H",
) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    t0 = perf_counter()
    idx = ai_index.index

    prices_df = _align(prices_df, idx)
    reddit_df = _align(reddit_df, idx, fill=0.0)
    news_df = _align(news_df, idx, fill=0.0)
    gdelt_df = _align(gdelt_df, idx, fill=0.0)
    if hnews_df is None:
        hnews_df = pd.DataFrame(index=idx)
    else:
        hnews_df = _align(hnews_df, idx, fill=0.0)
    if arxiv_df is None:
        arxiv_df = pd.DataFrame(index=idx)
    else:
        arxiv_df = _align(arxiv_df, idx, fill=0.0)

    feats = pd.DataFrame(index=idx)

    # AI index derived
    if isinstance(ai_index, pd.DataFrame):
        ai = ai_index.iloc[:, 0].reindex(idx)
    else:
        ai = pd.Series(ai_index).reindex(idx)
    ai = ai.astype(float)
    feats["ai_ret_6h"] = _log_ret(ai)
    feats["ai_ret_1d"] = feats["ai_ret_6h"].rolling(BIN_PER_DAY).sum().fillna(0.0)
    feats["ai_vol_7d"] = feats["ai_ret_6h"].rolling(WIN7).std().fillna(0.0)
    feats["ai_vol_30d"] = feats["ai_ret_6h"].rolling(WIN30).std().fillna(0.0)

    # Price returns for baselines if present
    for base in ["QQQ", "SPY"]:
        ccol = f"close_{base}"
        if ccol in prices_df.columns:
            r = _log_ret(prices_df[ccol])
            feats[f"ret_{base}_6h"] = r
            feats[f"corr_ai_{base}_30d"] = (
                feats["ai_ret_6h"].rolling(WIN30).corr(r).fillna(0.0)
            )

    # Volume spike features (z-score over 30 days), averaged across tickers
    vol_cols = [c for c in prices_df.columns if c.startswith("volume_")]
    if vol_cols:
        vol = prices_df[vol_cols]
        mu = vol.rolling(WIN30, min_periods=10).mean()
        sd = vol.rolling(WIN30, min_periods=10).std().replace(0, np.nan)
        z = (vol - mu) / sd
        feats["volume_spike_mean"] = z.mean(axis=1).fillna(0.0)
        feats["volume_spike_max"] = z.max(axis=1).fillna(0.0)

    # Reddit features
    if not reddit_df.empty:
        for col in ["reddit_posts", "reddit_comments", "reddit_mentions", "reddit_sent_mean", "reddit_velocity"]:
            if col in reddit_df:
                feats[col] = reddit_df[col].astype(float)
        if "reddit_sent_mean" in feats:
            feats["reddit_sent_diff"] = feats["reddit_sent_mean"].diff().fillna(0.0)
        if "reddit_mentions" in feats:
            feats["reddit_mentions_z"] = (
                (feats["reddit_mentions"] - feats["reddit_mentions"].rolling(WIN30).mean())
                / (feats["reddit_mentions"].rolling(WIN30).std().replace(0, np.nan))
            ).fillna(0.0)

    # News features
    if not news_df.empty:
        for col in ["news_count", "news_sent_mean"]:
            if col in news_df:
                feats[col] = news_df[col].astype(float)
        if "news_sent_mean" in feats:
            feats["news_sent_diff"] = feats["news_sent_mean"].diff().fillna(0.0)

    # Hacker News features
    if not hnews_df.empty:
        for col in ["hn_count", "hn_sent_mean", "hn_mentions"]:
            if col in hnews_df:
                feats[col] = hnews_df[col].astype(float)
        if "hn_sent_mean" in feats:
            feats["hn_sent_diff"] = feats["hn_sent_mean"].diff().fillna(0.0)

    # arXiv features
    if not arxiv_df.empty:
        for col in ["arxiv_count", "arxiv_title_sent_mean"]:
            if col in arxiv_df:
                feats[col] = arxiv_df[col].astype(float)

    # GDELT features
    if not gdelt_df.empty:
        for col in ["gdelt_count", "gdelt_sent_mean"]:
            if col in gdelt_df:
                feats[col] = gdelt_df[col].astype(float)
        if "gdelt_sent_mean" in feats:
            feats["gdelt_sent_diff"] = feats["gdelt_sent_mean"].diff().fillna(0.0)

    # Momentum/overheat indicators
    feats["ai_mom_7d"] = feats["ai_ret_6h"].rolling(WIN7).sum().fillna(0.0)
    feats["ai_mom_30d"] = feats["ai_ret_6h"].rolling(WIN30).sum().fillna(0.0)

    # Clean up
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    feats.to_csv("data/intermediate/features_6h.csv")
    logger.info(
        "features: built features shape=%s (cols=%d) saved (%.2fs)",
        feats.shape, feats.shape[1], perf_counter()-t0,
    )
    return feats
