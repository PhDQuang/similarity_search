"""Evaluate SFT-BE checkpoint + Cross-Encoder on AllNLI pair-class."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from similarity_search.models.minilm_baseline import pair_scores
from similarity_search.models.tfidf_baseline import (
    POSITIVE_LABEL,
    choose_threshold,
    load_split,
    pair_metrics,
    validate_frame,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SFTBE_CHECKPOINT_PATH = PROJECT_ROOT / "models" / "sftbe_checkpoint" / "stage0_final.pt"
CROSS_ENCODER_DIR = PROJECT_ROOT / "models" / "allnli-cross-encoder-nli" / "final"
LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="data/processed/allnli/pair-class",
        help="Directory containing dev/test parquet or CSV files.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs/sftbe_cross_encoder",
        help="Directory for hybrid metrics and predictions.",
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--cross-batch-size", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.55, help="Cross-Encoder score weight.")
    parser.add_argument("--retrieval-pool-size", type=int, default=20)
    parser.add_argument("--max-retrieval-queries", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def softmax(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def load_sftbe_model() -> Any:
    import torch

    from similarity_search.sftbe.config import DATA_CONFIG, MODEL_CONFIG, get_device
    from similarity_search.sftbe.dataset import get_tokenizer
    from similarity_search.sftbe.model import create_sftbe_model

    if not SFTBE_CHECKPOINT_PATH.exists():
        raise FileNotFoundError(f"SFT-BE checkpoint not found: {SFTBE_CHECKPOINT_PATH}")

    class SFTBEEmbedder:
        def __init__(self) -> None:
            self.torch = torch
            self.max_length = MODEL_CONFIG["max_seq_length"]
            self.device = get_device()
            self.tokenizer = get_tokenizer(DATA_CONFIG["tokenizer_name"])
            self.model = create_sftbe_model(MODEL_CONFIG).to(self.device)
            state = torch.load(SFTBE_CHECKPOINT_PATH, map_location=self.device, weights_only=True)
            self.model.load_state_dict(state.get("model_state_dict", state))
            self.model.eval()

        def encode(
            self,
            texts: list[str],
            batch_size: int = 64,
            convert_to_numpy: bool = True,
            normalize_embeddings: bool = True,
            show_progress_bar: bool = False,
        ) -> np.ndarray:
            _ = show_progress_bar
            embeddings: list[np.ndarray] = []
            for start in range(0, len(texts), batch_size):
                batch = texts[start : start + batch_size]
                encoded = self.tokenizer(
                    batch,
                    truncation=True,
                    padding=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                input_ids = encoded["input_ids"].to(self.device)
                attention_mask = encoded["attention_mask"].to(self.device)
                with self.torch.no_grad():
                    batch_embeddings = self.model(input_ids, attention_mask)
                    if normalize_embeddings:
                        batch_embeddings = self.torch.nn.functional.normalize(
                            batch_embeddings,
                            p=2,
                            dim=1,
                        )
                embeddings.append(batch_embeddings.detach().cpu().numpy())
            result = np.vstack(embeddings) if embeddings else np.empty((0, self.model.hidden_size), dtype=np.float32)
            if convert_to_numpy:
                return result
            return result

    return SFTBEEmbedder()


def load_cross_encoder_components() -> tuple[Any, Any, str]:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    if not (CROSS_ENCODER_DIR / "model.safetensors").exists():
        raise FileNotFoundError(f"Cross-Encoder model not found: {CROSS_ENCODER_DIR}")
    tokenizer = AutoTokenizer.from_pretrained(str(CROSS_ENCODER_DIR))
    model = AutoModelForSequenceClassification.from_pretrained(str(CROSS_ENCODER_DIR))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    return tokenizer, model, device


def cross_encoder_predict(pairs: list[tuple[str, str]], batch_size: int = 32) -> list[dict[str, float | str]]:
    import torch

    if not pairs:
        return []
    tokenizer, model, device = load_cross_encoder_components()
    outputs: list[dict[str, float | str]] = []
    for start in range(0, len(pairs), batch_size):
        batch = pairs[start : start + batch_size]
        encoded = tokenizer(
            [item[0] for item in batch],
            [item[1] for item in batch],
            truncation=True,
            padding=True,
            max_length=128,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits.detach().cpu().numpy()
        probabilities = softmax(logits)
        labels = probabilities.argmax(axis=1)
        for probs, label in zip(probabilities, labels):
            outputs.append(
                {
                    "entailment_probability": float(probs[0]),
                    "neutral_probability": float(probs[1]),
                    "contradiction_probability": float(probs[2]),
                    "nli_label": LABEL_NAMES[int(label)],
                }
            )
    return outputs


def cross_entailment_scores(frame: pd.DataFrame, batch_size: int) -> np.ndarray:
    outputs = cross_encoder_predict(
        list(zip(frame["premise_clean"].astype(str), frame["hypothesis_clean"].astype(str))),
        batch_size=batch_size,
    )
    return np.asarray([float(item["entailment_probability"]) for item in outputs])


def hybrid_scores(sftbe_scores: np.ndarray, entailment_scores: np.ndarray, alpha: float) -> np.ndarray:
    return alpha * entailment_scores + (1.0 - alpha) * sftbe_scores


def save_hybrid_predictions(
    frame: pd.DataFrame,
    scores: np.ndarray,
    sftbe_scores: np.ndarray,
    cross_scores: np.ndarray,
    threshold: float,
    path: Path,
) -> None:
    predictions = frame[["premise", "hypothesis", "label_name"]].copy()
    predictions["sftbe_cosine"] = sftbe_scores
    predictions["cross_entailment_probability"] = cross_scores
    predictions["hybrid_score"] = scores
    predictions["actual_similar"] = predictions["label_name"] == POSITIVE_LABEL
    predictions["predicted_similar"] = scores >= threshold
    predictions.to_csv(path, index=False)


def retrieval_metrics(
    model: Any,
    frame: pd.DataFrame,
    batch_size: int,
    cross_batch_size: int,
    alpha: float,
    pool_size: int,
    max_queries: int,
    seed: int,
) -> dict[str, float | int]:
    if pool_size < 2:
        raise ValueError("retrieval-pool-size must be at least 2")

    positives = frame[frame["label_name"] == POSITIVE_LABEL].reset_index(drop=True)
    negatives = frame[frame["label_name"] != POSITIVE_LABEL].reset_index(drop=True)
    if positives.empty or negatives.empty:
        raise ValueError("Retrieval evaluation requires entailment and non-entailment rows")

    rng = np.random.default_rng(seed)
    if max_queries > 0 and len(positives) > max_queries:
        selected = rng.choice(len(positives), size=max_queries, replace=False)
        positives = positives.iloc[selected].reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    replace = len(negatives) < pool_size - 1
    for query_index, row in positives.iterrows():
        query = str(row["hypothesis_clean"])
        candidate_texts = [str(row["premise_clean"])]
        distractor_indices = rng.choice(len(negatives), size=pool_size - 1, replace=replace)
        candidate_texts.extend(negatives.iloc[distractor_indices]["premise_clean"].astype(str).tolist())
        permutation = rng.permutation(pool_size)
        relevant_position = int(np.flatnonzero(permutation == 0)[0])
        for slot, candidate_index in enumerate(permutation):
            rows.append(
                {
                    "query_id": int(query_index),
                    "candidate_slot": int(slot),
                    "relevant_slot": relevant_position,
                    "premise_clean": candidate_texts[int(candidate_index)],
                    "hypothesis_clean": query,
                }
            )

    eval_frame = pd.DataFrame(rows)
    sftbe = pair_scores(model, eval_frame, batch_size)
    cross = cross_entailment_scores(eval_frame, cross_batch_size)
    eval_frame["hybrid_score"] = hybrid_scores(sftbe, cross, alpha)

    ranks: list[int] = []
    for _, group in eval_frame.groupby("query_id", sort=False):
        ranked = group.sort_values("hybrid_score", ascending=False).reset_index(drop=True)
        relevant_slot = int(group["relevant_slot"].iloc[0])
        rank = int(np.flatnonzero(ranked["candidate_slot"].to_numpy() == relevant_slot)[0]) + 1
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


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = {split: load_split(input_dir, split) for split in ("dev", "test")}
    for split, frame in frames.items():
        validate_frame(frame, split)

    model = load_sftbe_model()
    dev_sftbe = pair_scores(model, frames["dev"], args.batch_size)
    test_sftbe = pair_scores(model, frames["test"], args.batch_size)
    dev_cross = cross_entailment_scores(frames["dev"], args.cross_batch_size)
    test_cross = cross_entailment_scores(frames["test"], args.cross_batch_size)
    dev_scores = hybrid_scores(dev_sftbe, dev_cross, args.alpha)
    test_scores = hybrid_scores(test_sftbe, test_cross, args.alpha)

    dev_targets = (frames["dev"]["label_name"] == POSITIVE_LABEL).to_numpy()
    test_targets = (frames["test"]["label_name"] == POSITIVE_LABEL).to_numpy()
    threshold, best_dev_f1 = choose_threshold(dev_targets, dev_scores)

    metrics: dict[str, Any] = {
        "task": "entailment-as-semantic-similarity",
        "positive_label": POSITIVE_LABEL,
        "model": {
            "name": "SFT-BE checkpoint + Cross-Encoder",
            "retriever": "SFT-BE checkpoint",
            "reranker": "Cross-Encoder NLI",
            "alpha_cross_encoder": args.alpha,
            "score_formula": "alpha * entailment_probability + (1 - alpha) * sftbe_cosine",
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
                args.cross_batch_size,
                args.alpha,
                args.retrieval_pool_size,
                args.max_retrieval_queries,
                args.seed,
            ),
            "test": retrieval_metrics(
                model,
                frames["test"],
                args.batch_size,
                args.cross_batch_size,
                args.alpha,
                args.retrieval_pool_size,
                args.max_retrieval_queries,
                args.seed + 1,
            ),
        },
    }

    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    save_hybrid_predictions(
        frames["dev"],
        dev_scores,
        dev_sftbe,
        dev_cross,
        threshold,
        output_dir / "dev_predictions.csv",
    )
    save_hybrid_predictions(
        frames["test"],
        test_scores,
        test_sftbe,
        test_cross,
        threshold,
        output_dir / "test_predictions.csv",
    )
    pd.DataFrame(
        {
            "split": ["dev", "test"],
            "mean_sftbe_score": [float(dev_sftbe.mean()), float(test_sftbe.mean())],
            "mean_cross_entailment": [float(dev_cross.mean()), float(test_cross.mean())],
            "mean_hybrid_score": [float(dev_scores.mean()), float(test_scores.mean())],
        }
    ).to_csv(output_dir / "score_summary.csv", index=False)

    print(json.dumps(metrics, indent=2))
    print(f"Saved metrics -> {output_dir / 'metrics.json'}")


if __name__ == "__main__":
    main()
