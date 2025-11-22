from __future__ import annotations

from typing import List, Dict, Any

import logging
from time import perf_counter, sleep
import requests
import pandas as pd
import re

from ..utils.time import make_6h_grid
from ..utils.sentiment import score_texts


ALGOLIA_URL = "https://hn.algolia.com/api/v1/search_by_date"


def _clean_text(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def _mention_count(text: str, tickers: List[str]) -> int:
    if not text:
        return 0
    pattern = r"\b(?:" + "|".join(re.escape(t) for t in tickers) + r")\b"
    return len(re.findall(pattern, text.upper()))


def fetch_hackernews(
    tickers: List[str],
    start_date: str | None,
    end_date: str | None,
    resample: str = "6H",
) -> pd.DataFrame:
    logger = logging.getLogger(__name__)
    t0 = perf_counter()
    grid = make_6h_grid(start_date, end_date)
    resample = (resample or "6h").lower()

    base_queries = ["AI", "Artificial Intelligence"]
    queries = list(dict.fromkeys(base_queries + list(tickers)))  # unique, preserve order

    rows: List[Dict[str, Any]] = []
    session = requests.Session()

    start_epoch = int(pd.Timestamp(grid.min()).timestamp())
    end_epoch = int(pd.Timestamp(grid.max()).timestamp())

    for q in queries:
        page = 0
        while True:
            params = {
                "query": q,
                "tags": "story",
                "numericFilters": f"created_at_i>={start_epoch},created_at_i<={end_epoch}",
                "page": page,
                "hitsPerPage": 100,
            }
            try:
                r = session.get(ALGOLIA_URL, params=params, timeout=15)
                r.raise_for_status()
                data = r.json()
            except Exception:
                logger.warning("hackernews: request failed for q=%s page=%d", q, page, exc_info=True)
                break

            hits = data.get("hits", [])
            if not hits:
                break

            for h in hits:
                ts = int(h.get("created_at_i", 0) or 0)
                if ts < start_epoch or ts > end_epoch:
                    continue
                title = h.get("title") or h.get("story_title") or ""
                text = h.get("story_text") or ""
                txt = _clean_text(f"{title} {text}")
                rows.append(
                    {
                        "ts": pd.to_datetime(ts, unit="s", utc=True).tz_localize(None),
                        "title": title,
                        "text": txt,
                        "mentions": _mention_count(txt, tickers),
                    }
                )
            page += 1
            if page >= 20:
                break
            sleep(0.2)

    if not rows:
        logger.info("hackernews: no rows collected; returning empty frame")
        return pd.DataFrame(index=grid)

    df = pd.DataFrame(rows)
    df["bin"] = df["ts"].dt.floor(resample)

    try:
        df["sent"] = pd.Series(score_texts(df["text"].tolist()), index=df.index)
    except Exception:
        df["sent"] = 0.0

    agg = (
        df.groupby("bin")
        .agg(
            hn_count=("title", "count"),
            hn_sent_mean=("sent", "mean"),
            hn_mentions=("mentions", "sum"),
        )
        .reindex(grid, fill_value=0)
    )

    agg.to_csv("data/intermediate/hackernews_6h.csv")
    logger.info("hackernews: aggregated shape=%s saved (%.2fs)", agg.shape, perf_counter() - t0)
    return agg
