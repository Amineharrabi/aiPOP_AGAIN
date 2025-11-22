from __future__ import annotations

from typing import List

import logging
from time import perf_counter

import feedparser
import pandas as pd
from urllib.parse import urlencode

from ..utils.sentiment import score_texts
from ..utils.time import make_6h_grid


def _rss_urls_for_ticker(t: str) -> List[str]:
    # limited history via Google News; fetch recent window only
    q = f"{t} stock when:90d"
    params = {
        "q": q,
        "hl": "en-US",
        "gl": "US",
        "ceid": "US:en",
    }
    url = "https://news.google.com/rss/search?" + urlencode(params)
    return [url]


def fetch_google_news(
    tickers: List[str],
    start_date: str | None,
    end_date: str | None,
    resample: str = "6H",
) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    t0 = perf_counter()
    grid = make_6h_grid(start_date, end_date)
    resample = (resample or "6h").lower()
    if not tickers:
        logger.info("news: no tickers provided; returning empty frame")
        return pd.DataFrame(index=grid)

    rows = []
    for t in tickers:
        for url in _rss_urls_for_ticker(t):
            try:
                feed = feedparser.parse(url)
            except Exception:
                logger.exception("news: feed parse error for ticker=%s", t)
                continue
            for e in feed.entries:
                title = getattr(e, "title", "")
                published = getattr(e, "published", None) or getattr(e, "updated", None)
                if not published:
                    continue
                ts = pd.to_datetime(published, utc=True, errors="coerce")
                if pd.isna(ts):
                    continue
                ts = ts.tz_localize(None)
                rows.append({"ts": ts, "title": title})

    if not rows:
        logger.info("news: no rows collected; returning empty frame")
        return pd.DataFrame(index=grid)

    df = pd.DataFrame(rows)
    df["bin"] = df["ts"].dt.floor(resample)

    try:
        df["sent"] = pd.Series(score_texts(df["title"].tolist()), index=df.index)
    except Exception:
        df["sent"] = 0.0

    agg = df.groupby("bin").agg(
        news_count=("title", "count"),
        news_sent_mean=("sent", "mean"),
    ).reindex(grid, fill_value=0)

    agg.to_csv("data/intermediate/news_6h.csv")
    logger.info("news: aggregated shape=%s saved (%.2fs)", agg.shape, perf_counter()-t0)
    return agg
