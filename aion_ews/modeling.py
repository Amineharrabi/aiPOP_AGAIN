from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import logging
from time import perf_counter

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.isotonic import IsotonicRegression


@dataclass
class TrainedModel:
    pipeline: Pipeline
    calibrator: object
    features: List[str]


def _feature_columns(dataset: pd.DataFrame) -> List[str]:
    cols = [
        c
        for c in dataset.columns
        if not c.startswith("label_") and not c.startswith("fwd_min_ret_")
    ]
    return cols


class _ProbCalibrator:
    def __init__(self, pipe: Pipeline, method: str = "sigmoid"):
        self.pipe = pipe
        self.method = method
        if method == "isotonic":
            self.model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
            self._use_proba = True
        else:
            self.model = LogisticRegression(solver="lbfgs", max_iter=1000)
            self._use_proba = True

    def fit(self, X: np.ndarray, y: np.ndarray):
        p = self.pipe.predict_proba(X)[:, 1]
        if isinstance(self.model, IsotonicRegression):
            self.model.fit(p, y)
        else:
            self.model.fit(p.reshape(-1, 1), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p = self.pipe.predict_proba(X)[:, 1]
        if isinstance(self.model, IsotonicRegression):
            pc = self.model.predict(p)
        else:
            pc = self.model.predict_proba(p.reshape(-1, 1))[:, 1]
        pc = np.clip(pc, 0.0, 1.0)
        return np.vstack([1.0 - pc, pc]).T


def _fit_calibrated_logreg(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    random_state: int,
    calibrate_method: str = "sigmoid",
) -> Tuple[Pipeline, object]:
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "logreg",
                LogisticRegression(
                    penalty="l2",
                    C=1.0,
                    solver="lbfgs",
                    max_iter=200,
                    random_state=random_state,
                    n_jobs=None,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    pipe.fit(X_train, y_train)
    if calibrate_method == "none":
        return pipe, None
    # Custom calibration on validation slice without deprecated API
    calib = _ProbCalibrator(pipe, method=calibrate_method).fit(X_val, y_val)
    return pipe, calib


def _walk_forward_predictions(
    X: np.ndarray,
    y: np.ndarray,
    index: pd.DatetimeIndex,
    cfg: dict,
    feature_names: List[str],
    horizon_tag: str,
) -> Tuple[pd.Series, Dict[str, float], TrainedModel]:
    logger = logging.getLogger(__name__)
    t0 = perf_counter()
    n_splits = int(cfg["model"].get("cv_splits", 5))
    val_frac = float(cfg["model"].get("validation_frac", 0.2))
    random_state = int(cfg["model"].get("random_state", 42))

    tscv = TimeSeriesSplit(n_splits=n_splits)

    preds = pd.Series(index=index, dtype=float, name=f"p_{horizon_tag}")
    thr = float(cfg.get("model", {}).get("decision_threshold", 0.2))

    aucs: List[float] = []
    briers: List[float] = []
    precisions: List[float] = []
    recalls: List[float] = []

    last_model: TrainedModel | None = None

    for split_i, (train_idx, test_idx) in enumerate(tscv.split(X)):
        if len(train_idx) < 10 or len(test_idx) == 0:
            logger.debug("%s fold=%d skipped: small train/test sizes", horizon_tag, split_i)
            continue
        val_size = max(1, int(len(train_idx) * val_frac))
        tr_idx = train_idx[:-val_size]
        va_idx = train_idx[-val_size:]
        if len(tr_idx) < 5 or len(va_idx) < 1:
            logger.debug("%s fold=%d skipped: small tr/val", horizon_tag, split_i)
            continue
        # Require both classes in train and validation for stable fit/calibration
        if len(np.unique(y[tr_idx])) < 2 or len(np.unique(y[va_idx])) < 2:
            logger.debug("%s fold=%d skipped: single-class tr/val", horizon_tag, split_i)
            continue

        pipe, calib = _fit_calibrated_logreg(
            X[tr_idx], y[tr_idx], X[va_idx], y[va_idx],
            random_state=random_state,
            calibrate_method=cfg["model"].get("calibrate_method", "sigmoid"),
        )
        if calib is not None:
            proba = calib.predict_proba(X[test_idx])[:, 1]
        else:
            proba = pipe.predict_proba(X[test_idx])[:, 1]
        preds.iloc[test_idx] = proba

        # Metrics for this fold
        y_true = y[test_idx]
        if len(np.unique(y_true)) > 1:
            try:
                aucs.append(roc_auc_score(y_true, proba))
            except Exception:
                pass
        try:
            briers.append(brier_score_loss(y_true, proba))
        except Exception:
            pass
        y_hat = (proba >= thr).astype(int)
        try:
            precisions.append(precision_score(y_true, y_hat, zero_division=0))
            recalls.append(recall_score(y_true, y_hat, zero_division=0))
        except Exception:
            pass

        # Keep the last trained model
        last_model = TrainedModel(pipeline=pipe, calibrator=calib, features=feature_names)
        logger.info(
            "%s fold=%d: tr=%d va=%d te=%d done",
            horizon_tag, split_i, len(tr_idx), len(va_idx), len(test_idx)
        )

    metrics = {
        "auc": float(np.nanmean(aucs)) if aucs else float("nan"),
        "brier": float(np.nanmean(briers)) if briers else float("nan"),
        f"precision@{thr}": float(np.nanmean(precisions)) if precisions else float("nan"),
        f"recall@{thr}": float(np.nanmean(recalls)) if recalls else float("nan"),
        "threshold": thr,
    }

    if last_model is None:
        # Robust fallback
        unique = np.unique(y)
        if len(unique) < 2:
            # No positive events (or all ones). Use constant prevalence.
            prevalence = float(np.mean(y))
            proba = np.full(shape=len(y), fill_value=prevalence, dtype=float)
            preds = pd.Series(proba, index=index, name=f"p_{horizon_tag}")
            try:
                metrics["brier"] = float(brier_score_loss(y, proba))
            except Exception:
                pass
            logger.warning("%s: dataset single-class (prevalence=%.4f); using constant probabilities", horizon_tag, prevalence)
            logger.info("%s: completed in %.2fs", horizon_tag, perf_counter()-t0)
            return preds, metrics, None  # no model to save
        else:
            # Try to find a time-consistent split with both classes in train and val
            val_size = max(1, int(len(X) * val_frac))
            fitted = False
            for shift in range(0, len(X) - val_size):
                tr_idx = np.arange(0, len(X) - val_size - shift)
                va_idx = np.arange(len(X) - val_size - shift, len(X) - shift)
                if len(tr_idx) < 5 or len(va_idx) < 1:
                    continue
                if len(np.unique(y[tr_idx])) < 2 or len(np.unique(y[va_idx])) < 2:
                    continue
                pipe, calib = _fit_calibrated_logreg(
                    X[tr_idx], y[tr_idx], X[va_idx], y[va_idx],
                    random_state=random_state,
                    calibrate_method=cfg["model"].get("calibrate_method", "sigmoid"),
                )
                if calib is not None:
                    proba = calib.predict_proba(X)[:, 1]
                else:
                    proba = pipe.predict_proba(X)[:, 1]
                preds = pd.Series(proba, index=index, name=f"p_{horizon_tag}")
                last_model = TrainedModel(pipeline=pipe, calibrator=calib, features=feature_names)
                try:
                    metrics["auc"] = float(roc_auc_score(y, proba))
                    metrics["brier"] = float(brier_score_loss(y, proba))
                    metrics[f"precision@{thr}"] = float(precision_score(y, (proba >= thr).astype(int), zero_division=0))
                    metrics[f"recall@{thr}"] = float(recall_score(y, (proba >= thr).astype(int), zero_division=0))
                    metrics["threshold"] = thr
                except Exception:
                    pass
                fitted = True
                break
            if not fitted:
                # As a last resort, return constant prevalence
                prevalence = float(np.mean(y))
                proba = np.full(shape=len(y), fill_value=prevalence, dtype=float)
                preds = pd.Series(proba, index=index, name=f"p_{horizon_tag}")
                try:
                    metrics["brier"] = float(brier_score_loss(y, proba))
                except Exception:
                    pass
                logger.warning("%s: could not fit a valid split; using constant prevalence=%.4f", horizon_tag, prevalence)
                logger.info("%s: completed in %.2fs", horizon_tag, perf_counter()-t0)
                return preds, metrics, None

    logger.info("%s: completed in %.2fs (metrics=%s)", horizon_tag, perf_counter()-t0, metrics)
    return preds, metrics, last_model


def _compute_contributions(model: TrainedModel, x_last: np.ndarray) -> List[Tuple[str, float]]:
    try:
        # Standardized contributions in log-odds space
        scaler: StandardScaler = model.pipeline.named_steps["scaler"]
        logreg: LogisticRegression = model.pipeline.named_steps["logreg"]
        z = scaler.transform(x_last.reshape(1, -1))[0]
        coef = logreg.coef_.ravel()
        contrib = coef * z
        pairs = list(zip(model.features, contrib))
        pairs.sort(key=lambda p: abs(p[1]), reverse=True)
        return pairs
    except Exception:
        return []


def train_and_backtest(
    dataset: pd.DataFrame,
    horizons_days: Iterable[int],
    cfg: dict,
) -> Dict[str, any]:
    logger = logging.getLogger(__name__)
    t0 = perf_counter()
    dataset = dataset.sort_index()
    feature_names = _feature_columns(dataset)
    X = dataset[feature_names].values.astype(float)

    preds_cols = {}
    metrics_all: Dict[str, Dict[str, float]] = {}
    models: Dict[str, TrainedModel] = {}

    labels_df = dataset[[c for c in dataset.columns if c.startswith("label_crash_")]].copy()
    logger.info("modeling: dataset shape=%s, features=%d", dataset.shape, len(feature_names))

    for h in horizons_days:
        label_col = f"label_crash_{h}d"
        if label_col not in dataset:
            continue
        y = dataset[label_col].values.astype(int)
        tag = f"crash_{h}d"
        logger.info("modeling: horizon=%s positives=%d prevalence=%.4f", tag, int(y.sum()), float(y.mean()))
        p, metrics, model = _walk_forward_predictions(X, y, dataset.index, cfg, feature_names, tag)
        preds_cols[f"p_{tag}"] = p
        metrics_all[tag] = metrics
        models[tag] = model
        # Persist model if available
        if model is not None:
            try:
                path = f"models/model_{tag}.joblib"
                joblib.dump(model, path)
                logger.info("modeling: saved %s", path)
            except Exception:
                logger.exception("modeling: failed to save model %s", tag)

    preds_df = pd.DataFrame(preds_cols, index=dataset.index)
    logger.info("modeling: predictions shape=%s (%.2fs)", preds_df.shape, perf_counter()-t0)

    # Top drivers at the latest timestamp, choose horizon with max risk
    last_ts = preds_df.dropna(how="all").index.max() if not preds_df.empty else None
    top_drivers: List[Tuple[str, float]] = []
    last_probs = {}
    if last_ts is not None:
        last_row = preds_df.loc[last_ts]
        last_probs = {k: float(last_row[k]) for k in preds_df.columns}
        if len(last_row.dropna()) > 0:
            winner = last_row.idxmax()  # e.g., p_crash_3d
            model = models.get(winner.replace("p_", ""))
            if model is not None:
                x_last = dataset.loc[last_ts, feature_names].values.astype(float)
                top_drivers = _compute_contributions(model, x_last)[:8]
    logger.info("modeling: top_drivers=%s, last_probs=%s", top_drivers[:3], {k: round(v,3) for k,v in list(last_probs.items())[:3]})

    return {
        "preds_df": preds_df,
        "metrics": metrics_all,
        "models": models,
        "labels_df": labels_df,
        "top_drivers": top_drivers,
        "last_probs": last_probs,
    }
