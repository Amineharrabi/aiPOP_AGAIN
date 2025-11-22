from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd


def parse_date_or_none(s: Optional[str]) -> Optional[pd.Timestamp]:
    if s in (None, "", "null", "None"):
        return None
    # returns tz-aware UTC
    return pd.to_datetime(s, utc=True)


def to_utc_naive(idx: pd.DatetimeIndex) -> pd.DatetimeIndex:
    if idx.tz is not None:
        return idx.tz_convert("UTC").tz_localize(None)
    return idx


def make_6h_grid(start: Optional[str], end: Optional[str]) -> pd.DatetimeIndex:
    now_utc = pd.Timestamp.now(tz="UTC")
    start_ts = parse_date_or_none(start) or (now_utc - pd.Timedelta(days=365))
    end_ts = parse_date_or_none(end) or now_utc
    # Ensure tz-aware UTC
    if start_ts.tzinfo is None:
        start_ts = start_ts.tz_localize("UTC")
    else:
        start_ts = start_ts.tz_convert("UTC")
    if end_ts.tzinfo is None:
        end_ts = end_ts.tz_localize("UTC")
    else:
        end_ts = end_ts.tz_convert("UTC")
    grid = pd.date_range(start=start_ts, end=end_ts, freq="6h", tz="UTC").tz_localize(None)
    return grid
