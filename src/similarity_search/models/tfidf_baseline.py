"""Train and evaluate a TF-IDF semantic similarity baseline on AllNLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

POSITIVE_LABEL = "entailment"
NEGATIVE_RETRIEVAL_LABEL = "contradiction"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="data/processed/allnli/pair-class",
        help="Directory containing train/dev/test parquet or CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/tfidf_baseline",
        help="Directory for metrics, predictions, and reports.",
    )
    parser.add_argument(
        "--model-dir",
        default="models/tfidf_baseline",
        help="Directory for the fitted vectorizer and configuration.",
    )
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--ngram-max", type=int, default=2, choices=(1, 2, 3))
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


def load_split(input_dir: Path, split: str) -> pd.DataFrame:
    parquet_path = input_dir / f"{split}.parquet"
    csv_path = input_dir / f"{split}.csv"
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing {split}.parquet or {split}.csv under {input_dir}")


def validate_frame(frame: pd.DataFrame, split: str) -> None:
    required = {"premise_clean", "hypothesis_clean", "label_name"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{split} is missing required columns: {', '.join(missing)}")


def fit_vectorizer(
    train: pd.DataFrame,
    max_features: int,
    min_df: int,
    ngram_max: int,
) -> TfidfVectorizer:
    corpus = pd.concat(
        [train["premise_clean"], train["hypothesis_clean"]],
        ignore_index=True,
    ).fillna("")
    vectorizer = TfidfVectorizer(
        lowercase=True,
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
    premise_vectors = vectorizer.transform(frame["premise_clean"].fillna(""))
    hypothesis_vectors = vectorizer.transform(frame["hypothesis_clean"].fillna(""))
    return np.asarray(premise_vectors.multiply(hypothesis_vectors).sum(axis=1)).ravel()


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.0, 0.0
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1], 1e-12
    )
    best_index = int(np.argmax(f1))
    return float(thresholds[best_index]), float(f1[best_index])


def pair_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = scores >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="binary",
        zero_division=0,
    )
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "mean_positive_score": float(scores[y_true].mean()),
        "mean_negative_score": float(scores[~y_true].mean()),
    }


def retrieval_metrics(
    vectorizer: TfidfVectorizer,
    frame: pd.DataFrame,
    pool_size: int,
    max_queries: int,
    seed: int,
) -> dict[str, float | int]:
    if pool_size < 2:
        raise ValueError("retrieval-pool-size must be at least 2")

    positives = frame[frame["label_name"] == POSITIVE_LABEL].reset_index(drop=True)
    negatives = frame[frame["label_name"] == NEGATIVE_RETRIEVAL_LABEL].reset_index(drop=True)
    if positives.empty or negatives.empty:
        raise ValueError("Retrieval evaluation requires entailment and contradiction rows")

    rng = np.random.default_rng(seed)
    if max_queries > 0 and len(positives) > max_queries:
        selected = rng.choice(len(positives), size=max_queries, replace=False)
        positives = positives.iloc[selected].reset_index(drop=True)

    query_vectors = vectorizer.transform(positives["hypothesis_clean"].fillna(""))
    positive_vectors = vectorizer.transform(positives["premise_clean"].fillna(""))
    negative_vectors = vectorizer.transform(negatives["premise_clean"].fillna(""))
    replace = len(negatives) < pool_size - 1

    ranks: list[int] = []
    for query_index in range(len(positives)):
        distractor_indices = rng.choice(
            len(negatives),
            size=pool_size - 1,
            replace=replace,
        )
        distractor_scores = negative_vectors[distractor_indices].dot(
            query_vectors[query_index].T
        ).toarray().ravel()
        positive_score = float(
            positive_vectors[query_index].dot(query_vectors[query_index].T).toarray()[0, 0]
        )

        scores = np.concatenate(([positive_score], distractor_scores))
        permutation = rng.permutation(pool_size)
        shuffled_scores = scores[permutation]
        relevant_position = int(np.flatnonzero(permutation == 0)[0])
        ranked_positions = np.argsort(-shuffled_scores, kind="stable")
        rank = int(np.flatnonzero(ranked_positions == relevant_position)[0]) + 1
        ranks.append(rank)

    rank_array = np.asarray(ranks)
    hit_at_1 = float(np.mean(rank_array <= 1))
    hit_at_5 = float(np.mean(rank_array <= min(5, pool_size)))
    return {
        "queries": int(len(rank_array)),
        "candidate_pool_size": int(pool_size),
        "precision_at_1": hit_at_1,
        "precision_at_5": float(hit_at_5 / min(5, pool_size)),
        "recall_at_5": hit_at_5,
        "mrr": float(np.mean(1.0 / rank_array)),
        "mean_rank": float(np.mean(rank_array)),
    }


def save_predictions(
    frame: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    path: Path,
) -> None:
    predictions = frame[
        ["premise", "hypothesis", "label_name"]
    ].copy()
    predictions["tfidf_cosine"] = scores
    predictions["actual_similar"] = predictions["label_name"] == POSITIVE_LABEL
    predictions["predicted_similar"] = scores >= threshold
    predictions.to_csv(path, index=False)


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    frames = {split: load_split(input_dir, split) for split in ("train", "dev", "test")}
    for split, frame in frames.items():
        validate_frame(frame, split)

    vectorizer = fit_vectorizer(
        frames["train"],
        max_features=args.max_features,
        min_df=args.min_df,
        ngram_max=args.ngram_max,
    )

    dev_scores = pair_scores(vectorizer, frames["dev"])
    test_scores = pair_scores(vectorizer, frames["test"])
    dev_targets = (frames["dev"]["label_name"] == POSITIVE_LABEL).to_numpy()
    test_targets = (frames["test"]["label_name"] == POSITIVE_LABEL).to_numpy()
    threshold, best_dev_f1 = choose_threshold(dev_targets, dev_scores)

    metrics: dict[str, Any] = {
        "task": "entailment-as-semantic-similarity",
        "positive_label": POSITIVE_LABEL,
        "vectorizer": {
            "vocabulary_size": len(vectorizer.vocabulary_),
            "ngram_range": [1, args.ngram_max],
            "max_features": args.max_features,
            "min_df": args.min_df,
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
                vectorizer,
                frames["dev"],
                args.retrieval_pool_size,
                args.max_retrieval_queries,
                args.seed,
            ),
            "test": retrieval_metrics(
                vectorizer,
                frames["test"],
                args.retrieval_pool_size,
                args.max_retrieval_queries,
                args.seed + 1,
            ),
        },
    }

    joblib.dump(vectorizer, model_dir / "vectorizer.joblib")
    (model_dir / "config.json").write_text(
        json.dumps(metrics["vectorizer"], indent=2),
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    save_predictions(frames["dev"], dev_scores, threshold, output_dir / "dev_predictions.csv")
    save_predictions(frames["test"], test_scores, threshold, output_dir / "test_predictions.csv")

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics -> {output_dir / 'metrics.json'}")
    print(f"Saved vectorizer -> {model_dir / 'vectorizer.joblib'}")


if __name__ == "__main__":
    main()
