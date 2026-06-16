"""Create a compact comparison table from baseline metric files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tfidf-metrics",
        default="outputs/tfidf_baseline/metrics.json",
    )
    parser.add_argument(
        "--minilm-metrics",
        default="outputs/minilm_baseline/metrics.json",
    )
    parser.add_argument(
        "--fine-tuned-metrics",
        default="outputs/finetuned_minilm/metrics.json",
        help="Optional fine-tuned MiniLM metrics file.",
    )
    parser.add_argument(
        "--output",
        default="outputs/tables/model_comparison.csv",
    )
    return parser.parse_args()


def load_metrics(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def comparison_row(name: str, metrics: dict) -> dict[str, float | str]:
    pair = metrics["pair_classification"]["test"]
    retrieval = metrics["retrieval"]["test"]
    return {
        "model": name,
        "accuracy": pair["accuracy"],
        "precision": pair["precision"],
        "recall": pair["recall"],
        "f1": pair["f1"],
        "roc_auc": pair["roc_auc"],
        "average_precision": pair["average_precision"],
        "precision_at_1": retrieval["precision_at_1"],
        "recall_at_5": retrieval["recall_at_5"],
        "mrr": retrieval["mrr"],
        "mean_rank": retrieval["mean_rank"],
    }


def main() -> None:
    args = parse_args()
    tfidf = load_metrics(Path(args.tfidf_metrics))
    minilm = load_metrics(Path(args.minilm_metrics))
    rows = [
        comparison_row("TF-IDF", tfidf),
        comparison_row("Pretrained MiniLM", minilm),
    ]
    fine_tuned_path = Path(args.fine_tuned_metrics)
    if fine_tuned_path.exists():
        rows.append(
            comparison_row(
                "Fine-tuned MiniLM",
                load_metrics(fine_tuned_path),
            )
        )
    table = pd.DataFrame(rows)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(table.to_string(index=False))
    print(f"Saved comparison -> {output_path}")


if __name__ == "__main__":
    main()
