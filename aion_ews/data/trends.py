from __future__ import annotations

from typing import List

import logging
from time import perf_counter, sleep
import random

import pandas as pd

try:
    from pytrends.request import TrendReq  # type: ignore
    from pytrends.exceptions import TooManyRequestsError  # type: ignore
except Exception:
    TrendReq = None  # type: ignore
    class TooManyRequestsError(Exception):
        pass

from ..utils.time import make_6h_grid


def fetch_trends(
    keywords: List[str],
    start_date: str | None,
    end_date: str | None,
    resample: str = "6H",
) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    t0 = perf_counter()
    grid = make_6h_grid(start_date, end_date)
    if not keywords:
        logger.info("trends: no keywords; returning empty frame")
        return pd.DataFrame(index=grid)
    if TrendReq is None:
        logger.warning("trends: pytrends not available; returning empty frame")
        return pd.DataFrame(index=grid)

    tf = f"{pd.to_datetime(grid.min()).date()} {pd.to_datetime(grid.max()).date()}"
    tr = TrendReq(hl="en-US", tz=0)

    def _fetch_batch_with_retries(kw: List[str], timeframe: str, max_retries: int = 8) -> pd.DataFrame:
        delay = 2.0
        for attempt in range(max_retries):
            try:
                tr.build_payload(kw_list=kw, timeframe=timeframe)
                dfb = tr.interest_over_time()
                return dfb
            except TooManyRequestsError:
                logger.warning("trends: 429 TooManyRequests for %s (attempt %d/%d); sleeping %.1fs", kw, attempt+1, max_retries, delay)
                sleep(delay + random.uniform(0, 0.5))
                delay *= 2.0
            except Exception:
                logger.exception("trends: unexpected error for batch %s", kw)
                break
        return pd.DataFrame()

    frames = []
    # Google Trends: be conservative, use batches of 3
    for i in range(0, len(keywords), 3):
        kw = keywords[i : i + 5]
        kw = kw[:3]
        df = _fetch_batch_with_retries(kw, tf, max_retries=8)
        if df is None or df.empty:
            continue
        df = df.drop(columns=[c for c in df.columns if c.lower() == "isPartial".lower()], errors="ignore")
        frames.append(df)
        # throttle between batches to avoid 429
        sleep(2.0 + random.uniform(0, 1.0))

    if not frames:
        logger.info("trends: no frames collected; returning empty frame")
        return pd.DataFrame(index=grid)

    daily = pd.concat(frames, axis=1)
    daily.index = pd.to_datetime(daily.index, utc=True).tz_localize(None)

    # Forward-fill to 6H grid
    out = daily.reindex(grid, method="ffill")
    out.columns = [f"trends_{c}" for c in out.columns]
    out.to_csv("data/intermediate/trends_6h.csv")
    logger.info("trends: aggregated shape=%s saved (%.2fs)", out.shape, perf_counter()-t0)
    return out
