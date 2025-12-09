from __future__ import annotations

import logging
import os
from time import perf_counter
from typing import Dict, List, Tuple, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.metrics import (
    roc_auc_score,
    brier_score_loss,
    precision_score,
    recall_score,
)

BIN_PER_DAY = 4  # 6-hour bins per day


def _as_dt_index(idx: pd.Index) -> pd.DatetimeIndex:
    """Normalize any index to a naive UTC DatetimeIndex."""
    try:
        if not isinstance(idx, pd.DatetimeIndex):
            idx = pd.to_datetime(idx)
        if getattr(idx, "tz", None) is not None:
            idx = idx.tz_convert("UTC").tz_localize(None)
    except Exception:
        pass
    return pd.DatetimeIndex(idx)


def _median_lead_time(
    preds: pd.Series,
    labels: pd.Series,
    horizon_days: int,
    threshold: float = 0.2,
) -> float:
    if preds.empty or labels.empty:
        return float("nan")

    # Align with labels as master
    preds = preds.reindex(labels.index, fill_value=0.0)
    pos_times = labels.index[labels == 1]
    if pos_times.empty:
        return float("nan")

    lookback_steps = horizon_days * BIN_PER_DAY
    lead_deltas = []

    for t in pos_times:
        t_loc = labels.index.get_loc(t)
        start_loc = max(0, t_loc - lookback_steps)
        window_idx = labels.index[start_loc:t_loc]

        if window_idx.empty:
            continue

        window_preds = preds.loc[window_idx]
        hit_idx = window_preds[window_preds >= threshold].index
        if not hit_idx.empty:
            alert_time = hit_idx[0]
            lead_deltas.append(t - alert_time)

    if not lead_deltas:
        return float("nan")

    # Convert to hours using numpy timedelta arithmetic
    lead_hours = np.array(lead_deltas) / np.timedelta64(1, "h")
    return float(np.median(lead_hours))


def _metrics_for_horizon(preds: pd.Series, labels: pd.Series) -> Dict[str, float]:
    preds = preds.reindex(labels.index, fill_value=0.0)
    y = labels.values.astype(int)
    p = preds.values.astype(float)
    out = {}

    # AUC only if both classes present
    if len(np.unique(y)) > 1:
        try:
            out["auc"] = float(roc_auc_score(y, p))
        except Exception:
            pass

    # Brier always safe
    try:
        out["brier"] = float(brier_score_loss(y, p))
    except Exception:
        pass

    # Fixed-threshold classification metrics
    yhat = (p >= 0.2).astype(int)
    try:
        out["precision@0.5"] = float(precision_score(y, yhat, zero_division=0))
        out["recall@0.5"] = float(recall_score(y, yhat, zero_division=0))
    except Exception:
        pass

    return out


def make_backtest_plots_tiled(
    ai_index: pd.Series,
    results: Dict[str, Any],
    out_root: str = "outputs/tiles",
) -> str:
    logger = logging.getLogger(__name__)
    t0 = perf_counter()
    preds_df = results.get("preds_df", pd.DataFrame())

    # Normalize datetime indices
    ai_index = ai_index.copy()
    ai_index.index = _as_dt_index(ai_index.index)
    if not preds_df.empty:
        preds_df = preds_df.copy()
        preds_df.index = _as_dt_index(preds_df.index)

    # Align data
    ai_idx, p_idx = ai_index.index, preds_df.index if not preds_df.empty else pd.DatetimeIndex([])
    inter_idx = ai_idx.intersection(p_idx)

    if len(inter_idx) > 0:
        base_all = ai_index.reindex(inter_idx).dropna()
        preds_all = preds_df.reindex(inter_idx)
        x_index = inter_idx
    else:
        base_all = ai_index.dropna()
        x_index = ai_idx
        if not preds_df.empty:
            try:
                preds_all = preds_df.reindex(x_index, method="ffill", limit=1)
            except Exception:
                preds_all = preds_df.reindex(x_index)
        else:
            preds_all = preds_df

    def _horizon_key(col: str) -> Tuple[int, str]:
        for d in [1, 3, 7, 14, 30]:
            if f"{d}d" in col:
                return (d, col)
        return (999, col)

    prob_cols = []
    if isinstance(preds_all, pd.DataFrame) and not preds_all.empty:
        prob_cols = sorted(
            [c for c in preds_all.columns if isinstance(c, str) and c.startswith("p_")],
            key=_horizon_key
        )

    os.makedirs(out_root, exist_ok=True)
    years = sorted(set(x_index.year))
    color_map = {1: "#d62728", 3: "#ff7f0e", 7: "#2ca02c"}
    fallback_colors = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    def _plot_slice(xi: pd.DatetimeIndex, base: pd.Series, preds: pd.DataFrame, save_path: str, xfmt: str):
        if xi.empty:
            return
        fig, ax1 = plt.subplots(figsize=(12, 6))
        b = base.reindex(xi).dropna()
        if not b.empty:
            bnorm = (b / b.iloc[0]) * 100.0
            ax1.plot(bnorm.index, bnorm.values, color="#1f77b4", label="AI Index (norm)")
            ax1.set_ylabel("AI index (=100 start)", color="#1f77b4")
            ax1.tick_params(axis="y", labelcolor="#1f77b4")

        ax2 = ax1.twinx()
        for i, col in enumerate(prob_cols):
            hk = _horizon_key(col)[0]
            color = color_map.get(hk, fallback_colors[min(i, len(fallback_colors) - 1)])
            if col in preds.columns:
                y_vals = preds[col].reindex(xi).values
                ax2.step(xi, y_vals, where="post", color=color, alpha=0.85, linewidth=1.6, label=col)

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
        ydir = os.path.join(out_root, str(y))
        os.makedirs(ydir, exist_ok=True)
        ymask = x_index.year == y
        y_idx = x_index[ymask]
        y_preds = preds_all if not preds_all.empty else pd.DataFrame(index=y_idx)
        _plot_slice(y_idx, base_all, y_preds, os.path.join(ydir, f"{y}_year.png"), "months")

        for m in range(1, 13):
            mmask = ymask & (x_index.month == m)
            m_idx = x_index[mmask]
            if m_idx.empty:
                continue
            m_preds = preds_all if not preds_all.empty else pd.DataFrame(index=m_idx)
            _plot_slice(m_idx, base_all, m_preds, os.path.join(ydir, f"{y}-{m:02d}.png"), "days")

    logger.info("evaluation: tiled plots saved to %s for years=%s in %.2fs", out_root, years, perf_counter() - t0)
    return out_root


def make_backtest_plot(
    ai_index: pd.Series,
    results: Dict[str, Any],
    save_path: str = "outputs/backtest_latest.png",
) -> str:
    logger = logging.getLogger(__name__)
    t0 = perf_counter()
    preds_df = results.get("preds_df", pd.DataFrame())
    labels_df = results.get("labels_df", pd.DataFrame())

    ai_index = ai_index.copy()
    ai_index.index = _as_dt_index(ai_index.index)
    if not preds_df.empty:
        preds_df = preds_df.copy()
        preds_df.index = _as_dt_index(preds_df.index)

    ai_idx = ai_index.index
    p_idx = preds_df.index if not preds_df.empty else pd.DatetimeIndex([])
    inter_idx = ai_idx.intersection(p_idx)

    if len(inter_idx) > 0:
        base = ai_index.reindex(inter_idx).dropna()
        preds_plot = preds_df.reindex(inter_idx)
        x_index = inter_idx
        align_mode = "intersection"
    else:
        base = ai_index.dropna()
        x_index = ai_idx
        if not preds_df.empty:
            try:
                preds_plot = preds_df.reindex(x_index, method="ffill", limit=1)
            except Exception:
                preds_plot = preds_df.reindex(x_index)
        else:
            preds_plot = preds_df
        align_mode = "ffill_to_ai_index"

    def _horizon_key(col: str) -> Tuple[int, str]:
        for d in [1, 3, 7, 14, 30]:
            if f"{d}d" in col:
                return (d, col)
        return (999, col)

    prob_cols = []
    if not preds_df.empty:
        prob_cols = sorted(
            [c for c in preds_df.columns if isinstance(c, str) and c.startswith("p_")],
            key=_horizon_key
        )

    fig, ax1 = plt.subplots(figsize=(12, 6))

    if not base.empty:
        base_norm = (base / base.iloc[0]) * 100.0
        ax1.plot(base_norm.index, base_norm.values, color="#1f77b4", label="AI Index (norm)")
        ax1.set_ylabel("AI index (=100 start)", color="#1f77b4")
        ax1.tick_params(axis="y", labelcolor="#1f77b4")

    ax2 = ax1.twinx()
    color_map = {1: "#d62728", 3: "#ff7f0e", 7: "#2ca02c"}
    fallback_colors = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

    for i, col in enumerate(prob_cols):
        hk = _horizon_key(col)[0]
        color = color_map.get(hk, fallback_colors[min(i, len(fallback_colors) - 1)])
        if col in preds_plot.columns:
            y_vals = preds_plot[col].values
            ax2.step(x_index, y_vals, where="post", color=color, alpha=0.85, linewidth=1.6, label=col)

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
        save_path,
        getattr(preds_plot, "shape", None),
        getattr(labels_df, "shape", None),
        align_mode,
        perf_counter() - t0,
    )
    return save_path


def summarize_run_json(results: Dict[str, Any], timestamp_iso: str) -> Dict[str, Any]:
    logger = logging.getLogger(__name__)
    t0 = perf_counter()
    preds_df = results.get("preds_df", pd.DataFrame())
    labels_df = results.get("labels_df", pd.DataFrame())
    metrics_all = results.get("metrics", {})
    top_drivers = results.get("top_drivers", [])

    last_row = preds_df.dropna(how="all").iloc[-1] if not preds_df.empty else pd.Series(dtype=float)

    def _safe_get(name: str) -> float:
        val = last_row.get(name, np.nan) if isinstance(last_row, pd.Series) else np.nan
        return float(np.clip(val, 0, 1)) if not np.isnan(val) else float("nan")

    out = {
        "timestamp": timestamp_iso,
        "crash_prob_1d": _safe_get("p_crash_1d") if "p_crash_1d" in preds_df.columns else float("nan"),
        "crash_prob_3d": _safe_get("p_crash_3d") if "p_crash_3d" in preds_df.columns else float("nan"),
        "crash_prob_7d": _safe_get("p_crash_7d") if "p_crash_7d" in preds_df.columns else float("nan"),
        "top_drivers": [(str(f), float(v)) for f, v in top_drivers],
        "calibration": {},
    }

    # Infer threshold
    thr = 0.2
    for m in metrics_all.values():
        if isinstance(m, dict) and "threshold" in m:
            try:
                thr = float(m["threshold"])
                break
            except (TypeError, ValueError):
                continue

    # Add metrics + lead times
    for tag, metric_dict in metrics_all.items():
        horizon = tag.replace("crash_", "")
        clean_metrics = {k: (float(v) if pd.notna(v) else None) for k, v in metric_dict.items()}
        out["calibration"][horizon] = clean_metrics

        p_col = f"p_{tag}"
        label_col = f"label_{tag}"
        if p_col in preds_df.columns and label_col in labels_df.columns:
            med_lt = _median_lead_time(
                preds_df[p_col], labels_df[label_col], int(horizon.rstrip("d")), threshold=thr
            )
            out["calibration"][horizon][f"median_lead_time_hours@{thr}"] = (
                float(med_lt) if pd.notna(med_lt) else None
            )

    # Generate alert line
    try:
        probs = {h: out.get(f"crash_prob_{h}d") for h in ["1", "3", "7"]}
        valid_probs = {h: p for h, p in probs.items() if pd.notna(p)}
        if valid_probs:
            best_h, best_p = max(valid_probs.items(), key=lambda x: x[1])
            if best_p >= thr:
                pct = int(round(best_p * 100))
                out["alert_line"] = f"AION-EWS: Storm warning — {pct}% crash risk in next {best_h}d."
    except Exception:
        pass

    return out
