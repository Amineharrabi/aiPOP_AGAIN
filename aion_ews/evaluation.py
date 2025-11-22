from __future__ import annotations

from typing import Dict, List, Tuple

import logging
from time import perf_counter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib.dates as mdates


BIN_PER_DAY = 4  # 6H


def _median_lead_time(
    preds: pd.Series,
    labels: pd.Series,
    horizon_days: int,
    threshold: float = 0.2,
) -> float:
    if preds.empty or labels.empty:
        return float("nan")
    preds = preds.reindex(labels.index).fillna(0.0)
    lead_times: List[pd.Timedelta] = []
    lookback = horizon_days * BIN_PER_DAY

    pos_idx = labels.index[labels == 1]
    for t in pos_idx:
        # Find earliest alert crossing within previous horizon window
        start_i = max(0, labels.index.get_loc(t) - lookback)
        window_idx = labels.index[start_i : labels.index.get_loc(t)]
        if len(window_idx) == 0:
            continue
        pw = preds.loc[window_idx]
        hits = pw[pw >= threshold]
        if not hits.empty:
            t_alert = hits.index[0]
            lead_times.append(t - t_alert)
    if not lead_times:
        return float("nan")
    # median in hours
    return float(np.median([lt / np.timedelta64(1, "h") for lt in lead_times]))


def _metrics_for_horizon(preds: pd.Series, labels: pd.Series) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score, brier_score_loss, precision_score, recall_score

    preds = preds.reindex(labels.index).fillna(0.0)
    y = labels.values.astype(int)
    p = preds.values.astype(float)
    out = {}
    try:
        if len(np.unique(y)) > 1:
            out["auc"] = float(roc_auc_score(y, p))
    except Exception:
        pass
    try:
        out["brier"] = float(brier_score_loss(y, p))
    except Exception:
        pass
    yhat = (p >= 0.2).astype(int)
    try:
        out["precision@0.5"] = float(precision_score(y, yhat, zero_division=0))
        out["recall@0.5"] = float(recall_score(y, yhat, zero_division=0))
    except Exception:
        pass
    return out


def make_backtest_plots_tiled(
    ai_index: pd.Series,
    results: Dict[str, any],
    out_root: str = "outputs/tiles",
) -> str:
    logger = logging.getLogger(__name__)
    t0 = perf_counter()
    preds_df: pd.DataFrame = results.get("preds_df", pd.DataFrame())

    def _as_dt_index(idx: pd.Index) -> pd.DatetimeIndex:
        try:
            if not isinstance(idx, pd.DatetimeIndex):
                idx = pd.to_datetime(idx)
            if getattr(idx, "tz", None) is not None:
                try:
                    idx = idx.tz_convert("UTC").tz_localize(None)
                except Exception:
                    idx = idx.tz_localize(None)
        except Exception:
            pass
        return pd.DatetimeIndex(idx)

    ai_index = ai_index.copy()
    ai_index.index = _as_dt_index(ai_index.index)
    if isinstance(preds_df, pd.DataFrame) and not preds_df.empty:
        preds_df = preds_df.copy()
        preds_df.index = _as_dt_index(preds_df.index)

    ai_idx = ai_index.index
    p_idx = preds_df.index if isinstance(preds_df, pd.DataFrame) else pd.DatetimeIndex([])
    inter_idx = ai_idx.intersection(p_idx)

    if len(inter_idx) > 0:
        base_all = ai_index.reindex(inter_idx).dropna()
        preds_all = preds_df.reindex(inter_idx) if not preds_df.empty else preds_df
        x_index = inter_idx
    else:
        base_all = ai_index.dropna()
        x_index = ai_idx
        if isinstance(preds_df, pd.DataFrame) and not preds_df.empty:
            try:
                preds_all = preds_df.reindex(x_index, method="ffill", limit=1)
            except Exception:
                preds_all = preds_df.reindex(x_index)
        else:
            preds_all = preds_df

    def _horizon_key(c: str) -> Tuple[int, str]:
        for d in [1, 3, 7, 14, 30]:
            if f"{d}d" in c:
                return (d, c)
        return (999, c)

    prob_cols: List[str] = []
    if isinstance(preds_all, pd.DataFrame) and not preds_all.empty:
        prob_cols = [c for c in preds_all.columns if isinstance(c, str) and c.startswith("p_")]
        prob_cols = sorted(prob_cols, key=_horizon_key)

    os.makedirs(out_root, exist_ok=True)

    years = sorted(set(x_index.year))
    color_map = {1: "#d62728", 3: "#ff7f0e", 7: "#2ca02c"}
    fallback_colors = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    def _plot_slice(xi: pd.DatetimeIndex, base: pd.Series, preds: pd.DataFrame, save_path: str, xfmt: str):
        if len(xi) == 0:
            return
        fig, ax1 = plt.subplots(figsize=(12, 6))
        b = base.reindex(xi).dropna()
        if not b.empty:
            bnorm = b / b.iloc[0] * 100.0
            ax1.plot(bnorm.index, bnorm.values, color="#1f77b4", label="AI Index (norm)")
            ax1.set_ylabel("AI index (=100 start)", color="#1f77b4")
            ax1.tick_params(axis="y", labelcolor="#1f77b4")
        ax2 = ax1.twinx()
        fb_i = 0
        for col in prob_cols:
            hk = _horizon_key(col)[0]
            color = color_map.get(hk)
            if color is None:
                color = fallback_colors[min(fb_i, len(fallback_colors)-1)]
                fb_i = min(fb_i + 1, len(fallback_colors)-1)
            if isinstance(preds, pd.DataFrame) and col in preds.columns:
                y = preds[col].reindex(xi).values
                ax2.step(xi, y, where="post", color=color, alpha=0.85, linewidth=1.6, label=col)
        ax2.set_ylabel("Crash probability")
        ax2.set_ylim(0, 1)
        if xfmt == "months":
            ax1.xaxis.set_major_locator(mdates.MonthLocator())
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
        elif xfmt == "days":
            ax1.xaxis.set_major_locator(mdates.DayLocator(interval=2))
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%d"))
        lines1, labels1 = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")
        ax1.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)

    for y in years:
        ydir = os.path.join(out_root, f"{y}")
        os.makedirs(ydir, exist_ok=True)
        ymask = x_index.year == y
        y_idx = x_index[ymask]
        y_preds = preds_all if isinstance(preds_all, pd.DataFrame) else pd.DataFrame(index=y_idx)
        y_path = os.path.join(ydir, f"{y}_year.png")
        _plot_slice(y_idx, base_all, y_preds, y_path, xfmt="months")
        for m in range(1, 13):
            mmask = ymask & (x_index.month == m)
            m_idx = x_index[mmask]
            if len(m_idx) == 0:
                continue
            m_preds = preds_all if isinstance(preds_all, pd.DataFrame) else pd.DataFrame(index=m_idx)
            m_path = os.path.join(ydir, f"{y}-{m:02d}.png")
            _plot_slice(m_idx, base_all, m_preds, m_path, xfmt="days")

    logger.info("evaluation: tiled plots saved to %s for years=%s in %.2fs", out_root, years, perf_counter()-t0)
    return out_root


def make_backtest_plot(
    ai_index: pd.Series,
    results: Dict[str, any],
    save_path: str = "outputs/backtest_latest.png",
) -> str:
    logger = logging.getLogger(__name__)
    t0 = perf_counter()
    preds_df: pd.DataFrame = results.get("preds_df", pd.DataFrame())
    labels_df: pd.DataFrame = results.get("labels_df", pd.DataFrame())

    def _as_dt_index(idx: pd.Index) -> pd.DatetimeIndex:
        try:
            if not isinstance(idx, pd.DatetimeIndex):
                idx = pd.to_datetime(idx)
            if getattr(idx, "tz", None) is not None:
                try:
                    idx = idx.tz_convert("UTC").tz_localize(None)
                except Exception:
                    idx = idx.tz_localize(None)
        except Exception:
            pass
        return pd.DatetimeIndex(idx)

    ai_index = ai_index.copy()
    ai_index.index = _as_dt_index(ai_index.index)
    if isinstance(preds_df, pd.DataFrame) and not preds_df.empty:
        preds_df = preds_df.copy()
        preds_df.index = _as_dt_index(preds_df.index)

    ai_idx = ai_index.index
    p_idx = preds_df.index if isinstance(preds_df, pd.DataFrame) else pd.DatetimeIndex([])
    inter_idx = ai_idx.intersection(p_idx)

    try:
        logger.debug(
            "evaluation: ai_index len=%d freq=%s | preds len=%d freq=%s | intersection=%d",
            len(ai_idx), pd.infer_freq(ai_idx), len(p_idx), pd.infer_freq(p_idx), len(inter_idx),
        )
    except Exception:
        pass

    if len(inter_idx) > 0:
        base = ai_index.reindex(inter_idx).dropna()
        preds_plot = preds_df.reindex(inter_idx) if not preds_df.empty else preds_df
        x_index = inter_idx
        align_mode = "intersection"
    else:
        base = ai_index.dropna()
        x_index = ai_idx
        if isinstance(preds_df, pd.DataFrame) and not preds_df.empty:
            try:
                preds_plot = preds_df.reindex(x_index, method="ffill", limit=1)
            except Exception:
                preds_plot = preds_df.reindex(x_index)
        else:
            preds_plot = preds_df
        align_mode = "ffill_to_ai_index"

    try:
        logger.debug(
            "evaluation: align_mode=%s | base_len=%d | preds_plot_shape=%s",
            align_mode, len(base), getattr(preds_plot, "shape", None),
        )
    except Exception:
        pass

    def _horizon_key(c: str) -> Tuple[int, str]:
        for d in [1, 3, 7, 14, 30]:
            if f"{d}d" in c:
                return (d, c)
        return (999, c)

    prob_cols: List[str] = []
    if isinstance(preds_df, pd.DataFrame) and not preds_df.empty:
        prob_cols = [c for c in preds_df.columns if isinstance(c, str) and c.startswith("p_")]
        prob_cols = sorted(prob_cols, key=_horizon_key)

    fig, ax1 = plt.subplots(figsize=(12, 6))

    if not base.empty:
        base_norm = base / base.iloc[0] * 100.0
        ax1.plot(base_norm.index, base_norm.values, color="#1f77b4", label="AI Index (norm)")
        ax1.set_ylabel("AI index (=100 start)", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    color_map = {1: "#d62728", 3: "#ff7f0e", 7: "#2ca02c"}
    fallback_colors = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    fb_i = 0
    for col in prob_cols:
        hk = _horizon_key(col)[0]
        color = color_map.get(hk)
        if color is None:
            color = fallback_colors[min(fb_i, len(fallback_colors)-1)]
            fb_i = min(fb_i + 1, len(fallback_colors)-1)
        try:
            y = preds_plot[col].values if col in preds_plot.columns else None
        except Exception:
            y = None
        if y is None:
            continue
        ax2.step(x_index, y, where="post", color=color, alpha=0.85, linewidth=1.6, label=col)
    ax2.set_ylabel("Crash probability")
    ax2.set_ylim(0, 1)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

    ax1.set_title("AION-EWS Backtest: AI Index vs Crash Probabilities")
    ax1.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    logger.info(
        "evaluation: backtest plot saved to %s (preds=%s, labels=%s, align=%s) in %.2fs",
        save_path, getattr(preds_plot, "shape", None), getattr(labels_df, "shape", None), align_mode, perf_counter()-t0,
    )
    return save_path


def summarize_run_json(results: Dict[str, any], timestamp_iso: str) -> Dict[str, any]:
    logger = logging.getLogger(__name__)
    t0 = perf_counter()
    preds_df: pd.DataFrame = results.get("preds_df", pd.DataFrame())
    labels_df: pd.DataFrame = results.get("labels_df", pd.DataFrame())
    metrics_all: Dict[str, Dict[str, float]] = results.get("metrics", {})
    top_drivers: List[Tuple[str, float]] = results.get("top_drivers", [])

    last = preds_df.dropna(how="all").iloc[-1] if not preds_df.empty else pd.Series(dtype=float)

    def _get(name: str) -> float:
        return float(last.get(name, np.nan)) if isinstance(last, pd.Series) else float("nan")

    out = {
        "timestamp": timestamp_iso,
        "crash_prob_1d": float(np.clip(_get("p_crash_1d"), 0, 1)) if "p_crash_1d" in preds_df.columns else float("nan"),
        "crash_prob_3d": float(np.clip(_get("p_crash_3d"), 0, 1)) if "p_crash_3d" in preds_df.columns else float("nan"),
        "crash_prob_7d": float(np.clip(_get("p_crash_7d"), 0, 1)) if "p_crash_7d" in preds_df.columns else float("nan"),
        "top_drivers": [(str(f), float(v)) for f, v in top_drivers],
        "calibration": {},
    }

    # Decide threshold for downstream evaluation/alerting
    try:
        thr = float(next((m.get("threshold") for m in metrics_all.values() if isinstance(m, dict) and "threshold" in m), 0.2))
    except Exception:
        thr = 0.2

    # Add calibration/metrics and median lead time
    for tag, m in metrics_all.items():
        horizon = tag.replace("crash_", "")
        out["calibration"][horizon] = {
            k: (float(v) if v == v else None) for k, v in m.items()
        }
        # median lead time
        label_col = f"label_{tag}"
        p_col = f"p_{tag}"
        if p_col in preds_df.columns and label_col in labels_df.columns:
            med_lt = _median_lead_time(preds_df[p_col], labels_df[label_col], int(horizon.replace("d", "")), threshold=thr)
            out["calibration"][horizon][f"median_lead_time_hours@{thr}"] = med_lt if med_lt == med_lt else None

    # Alert line (cinematic) if any horizon exceeds threshold
    try:
        threshold = thr
        probs = {
            "1d": out.get("crash_prob_1d"),
            "3d": out.get("crash_prob_3d"),
            "7d": out.get("crash_prob_7d"),
        }
        # pick max horizon by probability
        best_h, best_p = max(((h, p) for h, p in probs.items() if p == p), key=lambda x: x[1], default=(None, None))
        if best_p is not None and best_p >= threshold:
            pct = int(round(best_p * 100))
            out["alert_line"] = f"AION-EWS: Storm warning — {pct}% crash risk in next {best_h}."
    except Exception:
        pass

    return out
