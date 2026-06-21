"""Evaluate the custom SFT-BE checkpoint on AllNLI pair-class."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from similarity_search.app.similarity_engine import SFTBE_CHECKPOINT_PATH, load_sftbe_model
from similarity_search.models.minilm_baseline import (
    pair_scores,
    retrieval_metrics,
    save_predictions,
)
from similarity_search.models.tfidf_baseline import (
    POSITIVE_LABEL,
    choose_threshold,
    load_split,
    pair_metrics,
    validate_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="data/processed/allnli/pair-class",
        help="Directory containing dev/test parquet or CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/sftbe_checkpoint",
        help="Directory for metrics and predictions.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--retrieval-pool-size",
        type=int,
        default=20,
        help="One relevant premise plus contradiction distractors per query.",
    )
    parser.add_argument(
        "--max-retrieval-queries",
        type=int,
        default=1_000,
        help="Maximum entailment queries per split; use 0 for all.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = {split: load_split(input_dir, split) for split in ("dev", "test")}
    for split, frame in frames.items():
        validate_frame(frame, split)

    model = load_sftbe_model()
    dev_scores = pair_scores(model, frames["dev"], args.batch_size)
    test_scores = pair_scores(model, frames["test"], args.batch_size)
    dev_targets = (frames["dev"]["label_name"] == POSITIVE_LABEL).to_numpy()
    test_targets = (frames["test"]["label_name"] == POSITIVE_LABEL).to_numpy()
    threshold, best_dev_f1 = choose_threshold(dev_targets, dev_scores)

    metrics: dict[str, Any] = {
        "task": "entailment-as-semantic-similarity",
        "positive_label": POSITIVE_LABEL,
        "model": {
            "name": "SFT-BE checkpoint",
            "checkpoint_path": str(SFTBE_CHECKPOINT_PATH),
            "embedding_dimension": int(model.model.hidden_size),
            "trained_in_project": True,
        },
        "threshold_selection": {
            "split": "dev",
            "threshold": threshold,
            "best_f1": best_dev_f1,
        },
        "pair_classification": {
            "dev": pair_metrics(dev_targets, dev_scores, threshold),
            "test": pair_metrics(test_targets, test_scores, threshold),
        },
        "retrieval": {
            "dev": retrieval_metrics(
                model,
                frames["dev"],
                args.batch_size,
                args.retrieval_pool_size,
                args.max_retrieval_queries,
                args.seed,
            ),
            "test": retrieval_metrics(
                model,
                frames["test"],
                args.batch_size,
                args.retrieval_pool_size,
                args.max_retrieval_queries,
                args.seed + 1,
            ),
        },
    }

    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    save_predictions(frames["dev"], dev_scores, threshold, output_dir / "dev_predictions.csv")
    save_predictions(frames["test"], test_scores, threshold, output_dir / "test_predictions.csv")

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics -> {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()

