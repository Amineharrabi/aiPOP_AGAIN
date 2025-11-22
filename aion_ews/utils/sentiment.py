from __future__ import annotations

from typing import Iterable, List
import os


def _try_hf_pipeline():
    try:
        from transformers import pipeline  # type: ignore

        model_id = os.environ.get(
            "AION_EWS_HF_MODEL",
            "sshleifer/tiny-distilbert-base-uncased-finetuned-sst-2-english",
        )
        return pipeline("sentiment-analysis", model=model_id)
    except Exception:
        return None


_VADER = None
_HF = None
_ALLOW_HF = os.environ.get("AION_EWS_USE_HF", "").lower() in ("1", "true", "yes", "on")


def _get_vader():
    global _VADER
    if _VADER is None:
        try:
            from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer  # type: ignore

            _VADER = SentimentIntensityAnalyzer()
        except Exception:
            _VADER = None
    return _VADER


def _get_hf():
    global _HF
    if not _ALLOW_HF:
        return None
    if _HF is None:
        _HF = _try_hf_pipeline()
    return _HF


def score_texts(texts: Iterable[str]) -> List[float]:
    texts = [t for t in texts if isinstance(t, str) and t.strip()]
    if not texts:
        return []

    # Prefer lightweight VADER to avoid large downloads
    vader = _get_vader()
    if vader is not None:
        try:
            return [vader.polarity_scores(t).get("compound", 0.0) for t in texts]
        except Exception:
            pass

    # Optionally use Hugging Face pipeline if allowed
    hf = _get_hf()
    if hf is not None:
        try:
            out = hf(texts, truncation=True, max_length=256)
            scores = [
                d.get("score", 0.0)
                * (1 if d.get("label", "POSITIVE") == "POSITIVE" else -1)
                for d in out
            ]
            return scores
        except Exception:
            pass

    # last resort neutral
    return [0.0 for _ in texts]
