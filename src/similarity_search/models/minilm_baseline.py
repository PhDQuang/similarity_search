"""Evaluate a pretrained MiniLM SentenceTransformer on AllNLI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from similarity_search.models.tfidf_baseline import (
    NEGATIVE_RETRIEVAL_LABEL,
    POSITIVE_LABEL,
    choose_threshold,
    load_split,
    pair_metrics,
    validate_frame,
)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="data/processed/allnli/pair-class",
        help="Directory containing dev/test parquet or CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/minilm_baseline",
        help="Directory for metrics and predictions.",
    )
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument(
        "--trained-in-project",
        action="store_true",
        help="Mark the evaluated model as fine-tuned by this project.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        default=None,
        help="SentenceTransformer device, for example cpu or cuda.",
    )
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


def encode_texts(
    model: Any,
    texts: pd.Series,
    batch_size: int,
) -> np.ndarray:
    return model.encode(
        texts.fillna("").astype(str).tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def pair_scores(
    model: Any,
    frame: pd.DataFrame,
    batch_size: int,
) -> np.ndarray:
    premise_embeddings = encode_texts(model, frame["premise_clean"], batch_size)
    hypothesis_embeddings = encode_texts(model, frame["hypothesis_clean"], batch_size)
    return np.sum(premise_embeddings * hypothesis_embeddings, axis=1)


def retrieval_metrics(
    model: Any,
    frame: pd.DataFrame,
    batch_size: int,
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

    query_embeddings = encode_texts(model, positives["hypothesis_clean"], batch_size)
    positive_embeddings = encode_texts(model, positives["premise_clean"], batch_size)
    negative_embeddings = encode_texts(model, negatives["premise_clean"], batch_size)
    replace = len(negatives) < pool_size - 1

    ranks: list[int] = []
    for query_index in range(len(positives)):
        distractor_indices = rng.choice(
            len(negatives),
            size=pool_size - 1,
            replace=replace,
        )
        distractor_scores = (
            negative_embeddings[distractor_indices] @ query_embeddings[query_index]
        )
        positive_score = float(
            positive_embeddings[query_index] @ query_embeddings[query_index]
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
    predictions = frame[["premise", "hypothesis", "label_name"]].copy()
    predictions["minilm_cosine"] = scores
    predictions["actual_similar"] = predictions["label_name"] == POSITIVE_LABEL
    predictions["predicted_similar"] = scores >= threshold
    predictions.to_csv(path, index=False)


def main() -> None:
    from sentence_transformers import SentenceTransformer

    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = {split: load_split(input_dir, split) for split in ("dev", "test")}
    for split, frame in frames.items():
        validate_frame(frame, split)

    model = SentenceTransformer(args.model_name, device=args.device)
    dev_scores = pair_scores(model, frames["dev"], args.batch_size)
    test_scores = pair_scores(model, frames["test"], args.batch_size)
    dev_targets = (frames["dev"]["label_name"] == POSITIVE_LABEL).to_numpy()
    test_targets = (frames["test"]["label_name"] == POSITIVE_LABEL).to_numpy()
    threshold, best_dev_f1 = choose_threshold(dev_targets, dev_scores)

    metrics: dict[str, Any] = {
        "task": "entailment-as-semantic-similarity",
        "positive_label": POSITIVE_LABEL,
        "model": {
            "name": args.model_name,
            "device": str(model.device),
            "embedding_dimension": int(model.get_embedding_dimension()),
            "trained_in_project": args.trained_in_project,
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
