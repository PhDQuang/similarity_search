"""Build the final model comparison table used in the report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tfidf-metrics", default="outputs/tfidf_baseline/metrics.json")
    parser.add_argument("--minilm-metrics", default="outputs/minilm_baseline/metrics.json")
    parser.add_argument(
        "--finetuned-table",
        default="outputs/finetuned_minilm/model_comparison.csv",
    )
    parser.add_argument(
        "--cross-metadata",
        default="outputs/cross_encoder_outputs/cross_encoder_training_metadata.json",
    )
    parser.add_argument("--sftbe-metrics", default="outputs/sftbe_checkpoint/metrics.json")
    parser.add_argument(
        "--output",
        default="outputs/tables/final_model_summary.csv",
    )
    return parser.parse_args()


def read_json(path: str | Path) -> dict[str, Any]:
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Missing required metrics file: {file_path}")
    return json.loads(file_path.read_text(encoding="utf-8"))


def row_from_pair_retrieval(model: str, metrics: dict[str, Any], notes: str) -> dict[str, Any]:
    pair = metrics["pair_classification"]["test"]
    retrieval = metrics["retrieval"]["test"]
    return {
        "model": model,
        "threshold": pair.get("threshold", metrics.get("threshold_selection", {}).get("threshold")),
        "accuracy": pair.get("accuracy"),
        "precision": pair.get("precision"),
        "recall": pair.get("recall"),
        "f1": pair.get("f1"),
        "roc_auc": pair.get("roc_auc"),
        "average_precision": pair.get("average_precision"),
        "precision_at_1": retrieval.get("precision_at_1"),
        "recall_at_5": retrieval.get("recall_at_5"),
        "mrr": retrieval.get("mrr"),
        "mean_rank": retrieval.get("mean_rank"),
        "notes": notes,
    }


def row_from_finetuned_table(path: str | Path) -> dict[str, Any]:
    table = pd.read_csv(path)
    candidates = table[table["model"].str.contains("Fine-tuned", case=False, na=False)]
    if candidates.empty:
        candidates = table[table["model"].str.contains("MiniLM", case=False, na=False)].tail(1)
    if candidates.empty:
        raise ValueError(f"No fine-tuned MiniLM row found in {path}")
    row = candidates.iloc[0].to_dict()
    return {
        "model": "Fine-tuned MiniLM",
        "threshold": row.get("threshold"),
        "accuracy": row.get("accuracy"),
        "precision": row.get("precision"),
        "recall": row.get("recall"),
        "f1": row.get("f1"),
        "roc_auc": row.get("roc_auc"),
        "average_precision": row.get("average_precision"),
        "precision_at_1": row.get("precision_at_1"),
        "recall_at_5": row.get("recall_at_5"),
        "mrr": row.get("mrr"),
        "mean_rank": row.get("mean_rank"),
        "notes": "MiniLM fine-tuned on AllNLI pair-score.",
    }


def cross_encoder_rows(metadata: dict[str, Any]) -> list[dict[str, Any]]:
    pair = metadata["test_binary_similarity"]
    rerank = metadata["retrieval_rerank"]
    cross_row = {
        "model": "Cross-Encoder NLI",
        "threshold": pair.get("threshold"),
        "accuracy": pair.get("accuracy"),
        "precision": pair.get("precision"),
        "recall": pair.get("recall"),
        "f1": pair.get("f1"),
        "roc_auc": pair.get("roc_auc"),
        "average_precision": pair.get("average_precision"),
        "precision_at_1": None,
        "recall_at_5": None,
        "mrr": None,
        "mean_rank": None,
        "notes": "Pairwise NLI classifier; retrieval columns omitted for full-corpus search cost.",
    }
    hybrid_row = {
        "model": "Fine-tuned MiniLM + Cross-Encoder",
        "threshold": pair.get("threshold"),
        "accuracy": pair.get("accuracy"),
        "precision": pair.get("precision"),
        "recall": pair.get("recall"),
        "f1": pair.get("f1"),
        "roc_auc": pair.get("roc_auc"),
        "average_precision": pair.get("average_precision"),
        "precision_at_1": rerank.get("precision_at_1"),
        "recall_at_5": rerank.get("recall_at_5"),
        "mrr": rerank.get("mrr"),
        "mean_rank": rerank.get("mean_rank"),
        "notes": "Fine-tuned MiniLM retrieves candidates, then Cross-Encoder reranks them.",
    }
    return [cross_row, hybrid_row]


def main() -> None:
    args = parse_args()
    rows = [
        row_from_pair_retrieval(
            "TF-IDF",
            read_json(args.tfidf_metrics),
            "Sparse lexical baseline with cosine similarity.",
        ),
        row_from_pair_retrieval(
            "Pretrained MiniLM",
            read_json(args.minilm_metrics),
            "SentenceTransformer all-MiniLM-L6-v2 without project fine-tuning.",
        ),
        row_from_finetuned_table(args.finetuned_table),
    ]
    rows.extend(cross_encoder_rows(read_json(args.cross_metadata)))
    rows.append(
        row_from_pair_retrieval(
            "SFT-BE checkpoint",
            read_json(args.sftbe_metrics),
            "Custom shallow factorized Transformer bi-encoder checkpoint.",
        )
    )

    columns = [
        "model",
        "threshold",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "average_precision",
        "precision_at_1",
        "recall_at_5",
        "mrr",
        "mean_rank",
        "notes",
    ]
    table = pd.DataFrame(rows, columns=columns)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(table.to_string(index=False))
    print(f"Saved final summary -> {output_path}")


if __name__ == "__main__":
    main()
