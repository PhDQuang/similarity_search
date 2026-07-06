"""Train and evaluate TF-IDF on the fixed full AllNLI 70/15/15 split."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from similarity_search.models.evaluation import (
    POSITIVE_LABEL,
    binary_confusion,
    entailment_targets,
    evaluate_pair_splits,
    evaluate_retrieval_splits,
    evaluate_test_sample_performance,
    load_splits,
    save_json,
    save_pair_predictions,
    validate_pair_class_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/processed/allnli_70_15_15_clean/pair-class")
    parser.add_argument("--output-dir", default="outputs/tfidf_baseline")
    parser.add_argument("--model-dir", default="models/tfidf_baseline")
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--retrieval-pool-size", type=int, default=20)
    parser.add_argument("--test-sample-size", type=int, default=5_000)
    parser.add_argument(
        "--max-retrieval-queries",
        type=int,
        default=0,
        help="0 means all entailment queries in val/test.",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def fit_vectorizer(
    train: pd.DataFrame,
    max_features: int,
    min_df: int,
    ngram_max: int,
) -> TfidfVectorizer:
    corpus = pd.concat([train["premise_clean"], train["hypothesis_clean"]], ignore_index=True).fillna("")
    vectorizer = TfidfVectorizer(
        lowercase=False,
        stop_words="english",
        ngram_range=(1, ngram_max),
        max_features=max_features,
        min_df=min_df,
        sublinear_tf=True,
        norm="l2",
    )
    vectorizer.fit(corpus)
    return vectorizer


def pair_scores(vectorizer: TfidfVectorizer, frame: pd.DataFrame) -> np.ndarray:
    left = vectorizer.transform(frame["premise_clean"].fillna("").astype(str))
    right = vectorizer.transform(frame["hypothesis_clean"].fillna("").astype(str))
    return np.asarray(left.multiply(right).sum(axis=1)).ravel()


def retrieval_pair_scores(vectorizer: TfidfVectorizer, frame: pd.DataFrame) -> np.ndarray:
    left = vectorizer.transform(frame["text_a"].fillna("").astype(str))
    right = vectorizer.transform(frame["text_b"].fillna("").astype(str))
    return np.asarray(left.multiply(right).sum(axis=1)).ravel()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    frames = load_splits(args.input_dir)
    for split, frame in frames.items():
        validate_pair_class_frame(frame, split)

    vectorizer = fit_vectorizer(
        frames["train"],
        max_features=args.max_features,
        min_df=args.min_df,
        ngram_max=args.ngram_max,
    )
    val_scores = pair_scores(vectorizer, frames["val"])
    test_scores = pair_scores(vectorizer, frames["test"])
    threshold, pair_report = evaluate_pair_splits(frames["val"], frames["test"], val_scores, test_scores)
    retrieval = evaluate_retrieval_splits(
        frames,
        score_pairs=lambda retrieval_frame: retrieval_pair_scores(vectorizer, retrieval_frame),
        pool_size=args.retrieval_pool_size,
        max_queries=args.max_retrieval_queries,
        seed=args.seed,
    )
    test5k_frame, test5k_scores, test5k_report = evaluate_test_sample_performance(
        frames["test"],
        score_pairs=lambda sample_frame: pair_scores(vectorizer, sample_frame),
        threshold=threshold,
        sample_size=args.test_sample_size,
        seed=args.seed,
    )

    metrics: dict[str, Any] = {
        "task": "entailment-as-semantic-similarity",
        "fixed_dataset": "AllNLI pair-class full 70/15/15",
        "positive_label": POSITIVE_LABEL,
        "model": {
            "name": "TF-IDF baseline",
            "trained_in_project": True,
            "vocabulary_size": int(len(vectorizer.vocabulary_)),
            "ngram_range": [1, args.ngram_max],
            "max_features": args.max_features,
            "min_df": args.min_df,
            "lowercase_done_in_dataset": True,
            "stop_words": "english",
        },
        **pair_report,
        "retrieval": retrieval,
        "test_sample_performance": test5k_report,
    }

    joblib.dump(vectorizer, model_dir / "vectorizer.joblib")
    save_json(metrics["model"], model_dir / "config.json")
    save_json(metrics, output_dir / "metrics.json")
    save_pair_predictions(frames["val"], val_scores, threshold, "tfidf_cosine", output_dir / "val_predictions.csv")
    save_pair_predictions(frames["test"], test_scores, threshold, "tfidf_cosine", output_dir / "test_predictions.csv")
    save_json(test5k_report, output_dir / "test5k_performance.json")
    save_pair_predictions(
        test5k_frame,
        test5k_scores,
        threshold,
        "tfidf_cosine",
        output_dir / "test5k_predictions.csv",
    )
    binary_confusion(entailment_targets(frames["test"]), test_scores, threshold).to_csv(
        output_dir / "binary_confusion_matrix.csv"
    )
    print(metrics)
    print(f"Saved TF-IDF model -> {model_dir}")
    print(f"Saved TF-IDF outputs -> {output_dir}")


if __name__ == "__main__":
    main()


