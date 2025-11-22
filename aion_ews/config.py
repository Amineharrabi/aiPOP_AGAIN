import os
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import yaml


def _env_or_default(name: str, default: Optional[str]) -> Optional[str]:
    v = os.environ.get(name)
    return v if v not in (None, "") else default


def default_config() -> Dict[str, Any]:
    today = datetime.utcnow().date()
    start = _env_or_default("AION_START_DATE", None)
    end = _env_or_default("AION_END_DATE", None)

    return {
        "start_date": start or (today - timedelta(days=365 * 5)).isoformat(),
        "end_date": end,  # None -> up to today
        "resample": "6H",
        "ai_tickers": [
            "NVDA",
            "MSFT",
            "AMD",
            "AVGO",
            "GOOGL",
            "META",
            "TSLA",
            "PLTR",
            "SMCI",
            "INTC",
        ],
        "baseline_tickers": ["QQQ", "SPY"],
        "subreddits": [
            "stocks",
            "investing",
            "wallstreetbets",
            "technology",
            "MachineLearning",
            "ArtificialIntelligence",
            "Artificial",
        ],
        "crash": {
            "horizons_days": [1, 3, 7],
            "drop_threshold": 0.07,
            "y_window_days": 3,  # reserved for alternative definitions
        },
        "model": {
            "cv_splits": 5,
            "calibrate_method": "isotonic",
            "validation_frac": 0.2,
            "random_state": 42,
            "decision_threshold": 0.2,
        },
    }


def load_config(path: str = "aion_ews_config.yaml") -> Dict[str, Any]:
    cfg = default_config()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            override = yaml.safe_load(f) or {}
        # shallow merge first level and for nested 'crash'/'model'
        for k, v in override.items():
            if isinstance(v, dict) and k in ("crash", "model"):
                cfg[k].update(v)
            else:
                cfg[k] = v
    return cfg
