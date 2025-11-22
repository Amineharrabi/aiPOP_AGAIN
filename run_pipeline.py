import json
import os
import logging
from time import perf_counter
from datetime import datetime

from aion_ews.config import load_config
from aion_ews.data.prices import fetch_prices_make_ai_index
from aion_ews.data.reddit import fetch_reddit_aggregate
from aion_ews.data.news import fetch_google_news
from aion_ews.data.gdelt import fetch_gdelt
from aion_ews.data.hackernews import fetch_hackernews
from aion_ews.data.arxiv import fetch_arxiv
from aion_ews.features import build_features
from aion_ews.labeling import make_labels
from aion_ews.modeling import train_and_backtest
from aion_ews.evaluation import make_backtest_plot, make_backtest_plots_tiled, summarize_run_json
from aion_ews.utils.io import ensure_dirs
from aion_ews.utils.logging import setup_logging


def main():
    log_path = setup_logging()
    logger = logging.getLogger("runner")
    cfg = load_config()
    ensure_dirs(["data/raw", "data/intermediate", "models", "outputs", "logs"])
    logger.info("Loaded config")

    # 1) Data collection
    t0 = perf_counter()
    logger.info("[1/5] Fetching prices and building AI index...")
    prices_df, ai_index = fetch_prices_make_ai_index(
        ai_tickers=cfg["ai_tickers"],
        baseline_tickers=cfg["baseline_tickers"],
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
        resample=cfg["resample"],
    )
    logger.info("Prices shape=%s, AI index points=%d (%.2fs)", prices_df.shape, len(ai_index), perf_counter()-t0)

    t0 = perf_counter()
    logger.info("[1/5] Fetching Reddit aggregates...")
    reddit_df = fetch_reddit_aggregate(
        subreddits=cfg["subreddits"],
        tickers=cfg["ai_tickers"] + cfg["baseline_tickers"],
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
        resample=cfg["resample"],
    )
    logger.info("Reddit shape=%s (%.2fs)", getattr(reddit_df, 'shape', None), perf_counter()-t0)

    t0 = perf_counter()
    logger.info("[1/5] Fetching Google News RSS...")
    news_df = fetch_google_news(
        tickers=cfg["ai_tickers"],
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
        resample=cfg["resample"],
    )
    logger.info("News shape=%s (%.2fs)", getattr(news_df, 'shape', None), perf_counter()-t0)

    t0 = perf_counter()
    logger.info("[1/5] Fetching Hacker News...")
    hnews_df = fetch_hackernews(
        tickers=cfg["ai_tickers"] + cfg["baseline_tickers"],
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
        resample=cfg["resample"],
    )
    logger.info("HackerNews shape=%s (%.2fs)", getattr(hnews_df, 'shape', None), perf_counter()-t0)

    t0 = perf_counter()
    logger.info("[1/5] Fetching arXiv...")
    arxiv_df = fetch_arxiv(
        keywords=["artificial intelligence", "AI", "machine learning"],
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
        resample=cfg["resample"],
    )
    logger.info("arXiv shape=%s (%.2fs)", getattr(arxiv_df, 'shape', None), perf_counter()-t0)

    t0 = perf_counter()
    logger.info("[1/5] Fetching GDELT news...")
    gdelt_df = fetch_gdelt(
        keywords=cfg["ai_tickers"] + ["Ai", "Artificial Intelligence"],
        start_date=cfg["start_date"],
        end_date=cfg["end_date"],
        resample=cfg["resample"],
    )
    logger.info("GDELT shape=%s (%.2fs)", getattr(gdelt_df, 'shape', None), perf_counter()-t0)

    # 2) Labeling on AI index
    t0 = perf_counter()
    logger.info("[2/5] Generating labels...")
    labels = make_labels(
        ai_index,
        horizons_days=cfg["crash"]["horizons_days"],
        drop_threshold=cfg["crash"]["drop_threshold"],
    )
    logger.info("Labels shape=%s (%.2fs)", labels.shape, perf_counter()-t0)

    # 3) Features
    t0 = perf_counter()
    logger.info("[3/5] Building features...")
    features = build_features(
        gdelt_df=gdelt_df,
        prices_df=prices_df,
        ai_index=ai_index,
        reddit_df=reddit_df,
        news_df=news_df,
        hnews_df=hnews_df,
        arxiv_df=arxiv_df,
        resample=cfg["resample"],
    )
    logger.info("Features shape=%s, columns=%d (%.2fs)", features.shape, features.shape[1], perf_counter()-t0)

    # Combine for modeling
    dataset = features.join(labels, how="inner").dropna()
    dataset.to_csv("data/intermediate/features_labels.csv")
    logger.info("Dataset shape after join=%s", dataset.shape)

    # 4) Modeling & Backtest
    t0 = perf_counter()
    logger.info("[4/5] Training and backtesting models...")
    results = train_and_backtest(
        dataset=dataset,
        horizons_days=cfg["crash"]["horizons_days"],
        cfg=cfg,
    )
    logger.info("Modeling done (%.2fs)", perf_counter()-t0)

    # 5) Evaluation
    t0 = perf_counter()
    logger.info("[5/5] Creating backtest plot and saving outputs...")
    fig_path = make_backtest_plot(ai_index, results, save_path="outputs/backtest_latest.png")
    tiles_dir = make_backtest_plots_tiled(ai_index, results, out_root="outputs/tiles")
    csv_out = os.path.join("outputs", "predictions_latest.csv")
    dataset.join(results["preds_df"], how="left").to_csv(csv_out, index_label="timestamp")

    # JSON summary for dashboard ingestion
    now_ts = datetime.utcnow().replace(microsecond=0).isoformat() + "Z"
    json_summary = summarize_run_json(results, now_ts)
    json_out = os.path.join("outputs", "run_latest.json")
    with open(json_out, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)
    logger.info("Saved outputs: csv=%s, fig=%s, tiles=%s, json=%s (%.2fs)", csv_out, fig_path, tiles_dir, json_out, perf_counter()-t0)

    # Short human-readable summary
    summary_path = os.path.join("outputs", "summary_latest.txt")
    lines = []
    lines.append(f"Run timestamp (UTC): {now_ts}")
    for tag, m in results.get("metrics", {}).items():
        thr = m.get("threshold", 0.2)
        prec_key = f"precision@{thr}"
        rec_key = f"recall@{thr}"
        lines.append(
            f"{tag}: AUC={m.get('auc', float('nan')):.3f}  Brier={m.get('brier', float('nan')):.3f}  "
            f"Precision@{thr}={m.get(prec_key, float('nan')):.3f}  Recall@{thr}={m.get(rec_key, float('nan')):.3f}"
        )
    last_probs = results.get("last_probs", {})
    if last_probs:
        lines.append("Latest crash probabilities:")
        for k, v in sorted(last_probs.items()):
            lines.append(f" - {k} = {v:.3f}")
    if results.get("top_drivers"):
        lines.append("Top driver features (log-odds contributions):")
        for feat, contrib in results["top_drivers"]:
            lines.append(f" - {feat}: {contrib:.3f}")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    print("Saved:")
    print(" -", csv_out)
    print(" -", fig_path)
    print(" -", json_out)
    print(" -", tiles_dir)
    print(" -", summary_path)


if __name__ == "__main__":
    main()
