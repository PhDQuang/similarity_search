"""Shared evaluation utilities so every fixed model uses the same protocol."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)

LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}
SCORE_BY_LABEL = {"entailment": 1.0, "neutral": 0.5, "contradiction": 0.0}
POSITIVE_LABEL = "entailment"
SPLITS = ("train", "val", "test")


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def save_json(data: dict[str, Any], path: str | Path) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(json_safe(data), indent=2), encoding="utf-8")


def load_split(input_dir: str | Path, split: str) -> pd.DataFrame:
    base = Path(input_dir) / split
    parquet_path = base.with_suffix(".parquet")
    csv_path = base.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    raise FileNotFoundError(f"Missing {split}.parquet or {split}.csv under {input_dir}")


def load_splits(input_dir: str | Path, splits: tuple[str, ...] = SPLITS) -> dict[str, pd.DataFrame]:
    return {split: load_split(input_dir, split) for split in splits}


def validate_pair_class_frame(frame: pd.DataFrame, split: str) -> None:
    required = {"premise_clean", "hypothesis_clean", "label_name"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{split} is missing required columns: {', '.join(missing)}")


def entailment_targets(frame: pd.DataFrame) -> np.ndarray:
    return (frame["label_name"].astype(str) == POSITIVE_LABEL).to_numpy()


def score_targets(frame: pd.DataFrame) -> np.ndarray:
    return frame["label_name"].map(SCORE_BY_LABEL).astype(float).to_numpy()


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best_index = int(np.argmax(f1))
    return float(thresholds[best_index]), float(f1[best_index])


def binary_pair_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    predictions = scores >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        predictions,
        average="binary",
        zero_division=0,
    )
    metrics = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, predictions)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "average_precision": float(average_precision_score(y_true, scores)),
        "mean_positive_score": float(scores[y_true].mean()) if y_true.any() else 0.0,
        "mean_negative_score": float(scores[~y_true].mean()) if (~y_true).any() else 0.0,
    }
    if len(np.unique(y_true)) > 1:
        metrics["roc_auc"] = float(roc_auc_score(y_true, scores))
    else:
        metrics["roc_auc"] = 0.0
    return metrics


def binary_confusion(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> pd.DataFrame:
    matrix = confusion_matrix(y_true, scores >= threshold, labels=[True, False])
    return pd.DataFrame(
        matrix,
        index=["actual_entailment", "actual_not_entailment"],
        columns=["pred_entailment", "pred_not_entailment"],
    )


def save_pair_predictions(
    frame: pd.DataFrame,
    scores: np.ndarray,
    threshold: float,
    score_column: str,
    path: str | Path,
) -> None:
    output = frame[["premise", "hypothesis", "label_name"]].copy()
    output[score_column] = scores
    output["actual_similar"] = output["label_name"] == POSITIVE_LABEL
    output["predicted_similar"] = scores >= threshold
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output.to_csv(output_path, index=False)


def build_retrieval_pairs(
    frame: pd.DataFrame,
    pool_size: int,
    max_queries: int,
    seed: int,
) -> pd.DataFrame:
    """Create one-positive retrieval pools from the same pair-class split.

    Query = hypothesis from an entailment row.
    Relevant candidate = paired premise from the same row.
    Distractors = premises from neutral/contradiction rows in the same split.
    """
    if pool_size < 2:
        raise ValueError("pool_size must be at least 2")

    positives = frame[frame["label_name"] == POSITIVE_LABEL].reset_index(drop=True)
    negatives = frame[frame["label_name"] != POSITIVE_LABEL].reset_index(drop=True)
    if positives.empty or negatives.empty:
        raise ValueError("Retrieval evaluation requires entailment and non-entailment rows")

    rng = np.random.default_rng(seed)
    if max_queries > 0 and len(positives) > max_queries:
        selected = rng.choice(len(positives), size=max_queries, replace=False)
        positives = positives.iloc[selected].reset_index(drop=True)

    replace = len(negatives) < pool_size - 1
    rows: list[dict[str, Any]] = []
    for query_id, row in positives.iterrows():
        query = str(row["hypothesis_clean"])
        candidate_texts = [str(row["premise_clean"])]
        negative_indices = rng.choice(len(negatives), size=pool_size - 1, replace=replace)
        candidate_texts.extend(negatives.iloc[negative_indices]["premise_clean"].astype(str).tolist())
        permutation = rng.permutation(pool_size)
        relevant_slot = int(np.flatnonzero(permutation == 0)[0])
        for slot, candidate_index in enumerate(permutation):
            rows.append(
                {
                    "query_id": int(query_id),
                    "candidate_slot": int(slot),
                    "relevant_slot": relevant_slot,
                    "text_a": candidate_texts[int(candidate_index)],
                    "text_b": query,
                    "is_relevant": int(candidate_index) == 0,
                }
            )
    return pd.DataFrame(rows)


def retrieval_metrics_from_scores(
    retrieval_frame: pd.DataFrame,
    scores: np.ndarray,
    pool_size: int,
) -> dict[str, float | int]:
    if len(retrieval_frame) != len(scores):
        raise ValueError(f"retrieval_frame rows and scores mismatch: {len(retrieval_frame)} != {len(scores)}")
    frame = retrieval_frame.copy()
    frame["score"] = scores
    ranks: list[int] = []
    for _, group in frame.groupby("query_id", sort=False):
        ranked = group.sort_values("score", ascending=False).reset_index(drop=True)
        relevant_positions = np.flatnonzero(ranked["is_relevant"].to_numpy(dtype=bool))
        if len(relevant_positions) != 1:
            raise ValueError("Each query must have exactly one relevant candidate")
        ranks.append(int(relevant_positions[0]) + 1)

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


def evaluate_pair_splits(
    val_frame: pd.DataFrame,
    test_frame: pd.DataFrame,
    val_scores: np.ndarray,
    test_scores: np.ndarray,
) -> tuple[float, dict[str, Any]]:
    val_targets = entailment_targets(val_frame)
    test_targets = entailment_targets(test_frame)
    threshold, best_val_f1 = choose_threshold(val_targets, val_scores)
    return threshold, {
        "threshold_selection": {
            "split": "val",
            "threshold": threshold,
            "best_f1": best_val_f1,
        },
        "pair_classification": {
            "val": binary_pair_metrics(val_targets, val_scores, threshold),
            "test": binary_pair_metrics(test_targets, test_scores, threshold),
        },
    }


ScorePairsFn = Callable[[pd.DataFrame], np.ndarray]


def evaluate_retrieval_splits(
    frames: dict[str, pd.DataFrame],
    score_pairs: ScorePairsFn,
    pool_size: int,
    max_queries: int,
    seed: int,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for offset, split in enumerate(("val", "test")):
        retrieval_frame = build_retrieval_pairs(
            frames[split],
            pool_size=pool_size,
            max_queries=max_queries,
            seed=seed + offset,
        )
        scores = score_pairs(retrieval_frame)
        metrics[split] = retrieval_metrics_from_scores(retrieval_frame, scores, pool_size)
    return metrics

