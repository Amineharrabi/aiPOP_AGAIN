from __future__ import annotations

from typing import List

import logging
from time import perf_counter, sleep

import pandas as pd

from ..utils.time import make_6h_grid
from ..utils.sentiment import score_texts


def fetch_arxiv(
    keywords: List[str],
    start_date: str | None,
    end_date: str | None,
    resample: str = "6H",
) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    t0 = perf_counter()
    grid = make_6h_grid(start_date, end_date)
    resample = (resample or "6h").lower()

    try:
        import arxiv  # type: ignore
    except Exception:
        logger.warning("arxiv: package not installed; returning empty frame")
        return pd.DataFrame(index=grid)

    # Build a simple query focusing on AI
    base_query = '("artificial intelligence" OR AI OR "machine learning")'
    # we ignore tickers here; arXiv is research-heavy

    client = arxiv.Client(page_size=500, delay_seconds=2, num_retries=2)
    search = arxiv.Search(
        query=base_query,
        max_results=1000,
        sort_by=arxiv.SortCriterion.SubmittedDate,
    )

    start_ts = pd.Timestamp(grid.min(), tz="UTC")
    end_ts = pd.Timestamp(grid.max(), tz="UTC")

    rows = []
    try:
        for result in client.results(search):
            # published is timezone-aware
            ts = pd.Timestamp(result.published).tz_convert("UTC")
            if ts < start_ts or ts > end_ts:
                continue
            title = (result.title or "").strip()
            summary = (result.summary or "").strip()
            txt = f"{title} {summary}".strip()
            rows.append({
                "ts": ts.tz_localize(None),
                "title": title,
                "text": txt,
            })
    except Exception:
        logger.warning("arxiv: error during fetch; returning what we have", exc_info=True)

    if not rows:
        logger.info("arxiv: no rows collected; returning empty frame")
        return pd.DataFrame(index=grid)

    df = pd.DataFrame(rows)
    df["bin"] = df["ts"].dt.floor(resample)
    try:
        df["sent"] = pd.Series(score_texts(df["title"].tolist()), index=df.index)
    except Exception:
        df["sent"] = 0.0

    agg = (
        df.groupby("bin")
        .agg(
            arxiv_count=("title", "count"),
            arxiv_title_sent_mean=("sent", "mean"),
        )
        .reindex(grid, fill_value=0)
    )

    agg.to_csv("data/intermediate/arxiv_6h.csv")
    logger.info("arxiv: aggregated shape=%s saved (%.2fs)", agg.shape, perf_counter() - t0)
    return agg
