from __future__ import annotations

from typing import Dict, Iterable, List

import logging
from time import perf_counter

import numpy as np
import pandas as pd


def _forward_min(series: pd.Series, steps: int) -> pd.Series:
    if steps <= 0:
        return pd.Series(index=series.index, dtype=float)
    rev = series.iloc[::-1]
    roll = rev.rolling(window=steps, min_periods=1).min()
    fwd_min = roll.iloc[::-1].shift(-1)
    return fwd_min


essentials = ["AI_INDEX"]


def make_labels(
    ai_index: pd.Series,
    horizons_days: Iterable[int] = (1, 3, 7),
    drop_threshold: float = 0.07,
) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    t0 = perf_counter()
    s = ai_index.rename("AI_INDEX").astype(float)
    s = s.sort_index()
    out: Dict[str, pd.Series] = {}

    for h in horizons_days:
        steps = int(h * 4)  # 6H -> 4 bins per day
        fwd_min = _forward_min(s, steps)
        rel_min_ret = (fwd_min / s) - 1.0
        label = (rel_min_ret <= -abs(drop_threshold)).astype(int)
        out[f"label_crash_{h}d"] = label
        out[f"fwd_min_ret_{h}d"] = rel_min_ret

    df = pd.DataFrame(out, index=s.index)
    df.to_csv("data/intermediate/labels_6h.csv")
    try:
        label_cols = [c for c in df.columns if c.startswith("label_")]
        pos = {c: int(df[c].sum()) for c in label_cols}
        logger.info("labels: generated shape=%s positives=%s (%.2fs)", df.shape, pos, perf_counter()-t0)
    except Exception:
        pass
    return df
