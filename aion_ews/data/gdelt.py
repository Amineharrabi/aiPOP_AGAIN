from __future__ import annotations

from typing import List

import logging
from time import perf_counter

import pandas as pd
from datetime import datetime, timedelta

try:
    from gdeltdoc import GdeltDoc, Filters
except Exception:
    GdeltDoc = None
    Filters = None

from ..utils.sentiment import score_texts
from ..utils.time import make_6h_grid


def fetch_gdelt(
    keywords: List[str],
    start_date: str | None,
    end_date: str | None,
    resample: str = "6H",
) -> pd.DataFrame:
    """Fetch GDELT news articles and aggregate by time bin"""
    logger = logging.getLogger(__name__)
    t0 = perf_counter()
    grid = make_6h_grid(start_date, end_date)
    resample = (resample or "6h").lower()
    
    if not keywords:
        logger.info("gdelt: no keywords provided; returning empty frame")
        return pd.DataFrame(index=grid)
    
    if GdeltDoc is None:
        logger.warning("gdelt: gdeltdoc not available; returning empty frame")
        return pd.DataFrame(index=grid)
    
    try:
        gd = GdeltDoc()
        
        # Parse date strings to datetime
        if isinstance(start_date, str):
            start_dt = pd.to_datetime(start_date)
        else:
            start_dt = start_date or (datetime.now() - timedelta(days=180))
        
        if isinstance(end_date, str):
            end_dt = pd.to_datetime(end_date)
        else:
            end_dt = end_date or datetime.now()
        
        logger.info(f"gdelt: fetching from {start_dt.date()} to {end_dt.date()}")
        
        # Create filters using gdeltdoc
        filters = Filters(
            keyword=keywords,
            start_date=start_dt.strftime('%Y-%m-%d'),
            end_date=end_dt.strftime('%Y-%m-%d'),
            num_records=250
        )
        logger.info(f"gdelt: calling article_search with keywords={keywords}")
        
        # Fetch articles
        df = gd.article_search(filters)
        logger.info(f"gdelt: article_search returned, type={type(df)}, empty={df.empty if df is not None else 'None'}")
        
        if df is None or df.empty:
            logger.info("gdelt: no articles retrieved; returning empty frame")
            return pd.DataFrame(index=grid)
        
        logger.info(f"gdelt: retrieved {len(df)} articles, columns: {df.columns.tolist()}")
        if len(df) > 0:
            first_row = df.iloc[0].to_dict()
            logger.info(f"gdelt: sample row: {first_row}")
            logger.info(f"gdelt: title type={type(first_row.get('title'))}, value={repr(first_row.get('title'))}")
            logger.info(f"gdelt: seendate type={type(first_row.get('seendate'))}, value={repr(first_row.get('seendate'))}")
        
        # Extract title and published date
        rows = []
        for idx, row in df.iterrows():
            try:
                # gdeltdoc returns: url, url_mobile, title, seendate, socialimage, domain, language, sourcecountry
                title = row['title'] if 'title' in row.index else ''
                seendate = row['seendate'] if 'seendate' in row.index else None
                
                logger.debug(f"gdelt row {idx}: title={repr(title)[:50]}, seendate={repr(seendate)}")
                
                # Skip if missing or null
                if pd.isna(title) or not title or pd.isna(seendate) or not seendate:
                    continue
                
                # Parse seendate - GDELT returns ISO 8601 format: 20250919T124500Z
                # Try ISO format first, then fall back to compact format
                seendate_str = str(seendate)
                
                # Handle ISO 8601 format with T and Z
                if 'T' in seendate_str:
                    # Remove the 'Z' if present and parse
                    seendate_clean = seendate_str.replace('Z', '')
                    ts = pd.to_datetime(seendate_clean, format='%Y%m%dT%H%M%S', errors='coerce')
                else:
                    # Fall back to compact format without T
                    ts = pd.to_datetime(seendate_str, format='%Y%m%d%H%M%S', errors='coerce')
                
                if pd.isna(ts):
                    logger.debug(f"gdelt: failed to parse seendate: {seendate}")
                    continue
                
                # Ensure timezone-naive (GDELT times are UTC)
                if ts.tz is not None:
                    ts = ts.tz_localize(None)
                
                rows.append({"ts": ts, "title": str(title)})
            except Exception as e:
                logger.debug(f"gdelt: error parsing row {idx}: {e}")
                continue
        
        logger.info(f"gdelt: extracted {len(rows)} valid rows from {len(df)} articles")
        if not rows:
            logger.warning(f"gdelt: no valid rows extracted; returning empty frame")
            if len(df) > 0:
                logger.debug(f"gdelt: first row: {df.iloc[0].to_dict()}")
            return pd.DataFrame(index=grid)
        
        df = pd.DataFrame(rows)
        df["bin"] = df["ts"].dt.floor(resample)
        
        # Score sentiment
        try:
            df["sent"] = pd.Series(score_texts(df["title"].tolist()), index=df.index)
        except Exception:
            df["sent"] = 0.0
        
        # Aggregate by time bin
        agg = df.groupby("bin").agg(
            gdelt_count=("title", "count"),
            gdelt_sent_mean=("sent", "mean"),
        ).reindex(grid, fill_value=0)
        
        agg.to_csv("data/intermediate/gdelt_6h.csv")
        logger.info("gdelt: aggregated shape=%s saved (%.2fs)", agg.shape, perf_counter()-t0)
        return agg
        
    except Exception as e:
        logger.error(f"gdelt: error fetching articles: {e}", exc_info=True)
        return pd.DataFrame(index=grid)