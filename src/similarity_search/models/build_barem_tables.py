"""Build final comparison tables and a barem artifact manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd

from similarity_search.models.evaluation import save_json


MODEL_OUTPUTS = {
    "TF-IDF": "tfidf_baseline",
    "Fine-tuned MiniLM": "finetuned_minilm",
    "Cross-Encoder NLI": "cross_encoder_outputs",
    "SFT-BE AllNLI": "sftbe_checkpoint",
}

MODEL_ARTIFACTS = {
    "TF-IDF": "models/tfidf_baseline",
    "Fine-tuned MiniLM": "models/allnli-minilm-biencoder/final",
    "Cross-Encoder NLI": "models/allnli-cross-encoder-nli/final",
    "SFT-BE AllNLI": "models/sftbe_checkpoint/stage1_allnli_final.pt",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--output-table-dir", default="outputs/tables")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def model_row(name: str, metrics: dict[str, Any]) -> dict[str, Any]:
    test_pair = metrics.get("pair_classification", {}).get("test", {})
    test_retrieval = metrics.get("retrieval", {}).get("test", {})
    nli_test = metrics.get("nli", {}).get("test", {})
    model = metrics.get("model", {})
    threshold = metrics.get("threshold_selection", {}).get("threshold", test_pair.get("threshold"))
    return {
        "model": name,
        "fixed_dataset": metrics.get("fixed_dataset", ""),
        "base_or_source_model": model.get("base_model", model.get("source_checkpoint", "")),
        "loss": model.get("loss", ""),
        "threshold": threshold,
        "accuracy": test_pair.get("accuracy"),
        "precision": test_pair.get("precision"),
        "recall": test_pair.get("recall"),
        "f1": test_pair.get("f1"),
        "roc_auc": test_pair.get("roc_auc"),
        "average_precision": test_pair.get("average_precision"),
        "precision_at_1": test_retrieval.get("precision_at_1"),
        "precision_at_5": test_retrieval.get("precision_at_5"),
        "recall_at_5": test_retrieval.get("recall_at_5"),
        "mrr": test_retrieval.get("mrr"),
        "mean_rank": test_retrieval.get("mean_rank"),
        "nli_accuracy": nli_test.get("test_accuracy"),
        "nli_macro_f1": nli_test.get("test_macro_f1"),
        "notes": "All rows use the same fixed AllNLI pair-class 70/15/15 split.",
    }


def training_row(name: str, output_dir: Path) -> dict[str, Any] | None:
    metadata = read_json(output_dir / "training_metadata.json")
    metrics = read_json(output_dir / "metrics.json")
    if metadata is None and metrics is None:
        return None
    metadata = metadata or {}
    metrics = metrics or {}
    model = metrics.get("model", {})
    return {
        "model": name,
        "fixed_dataset": metadata.get("fixed_dataset", metrics.get("fixed_dataset", "")),
        "train_rows": metadata.get("train_rows"),
        "val_rows": metadata.get("val_rows"),
        "test_rows": metadata.get("test_rows"),
        "loss": metadata.get("loss", model.get("loss", "")),
        "epochs": metadata.get("epochs"),
        "batch_size": metadata.get("batch_size"),
        "gradient_accumulation_steps": metadata.get("gradient_accumulation_steps"),
        "learning_rate": metadata.get("learning_rate"),
        "warmup_ratio": metadata.get("warmup_ratio"),
        "gpu_or_device": metadata.get("gpu", metadata.get("device", "")),
        "artifact": MODEL_ARTIFACTS.get(name, metadata.get("final_model_dir", metadata.get("final_checkpoint", ""))),
    }


def manifest(outputs_dir: Path) -> dict[str, Any]:
    return {
        "dataset": {
            "metadata": "data/processed/allnli_70_15_15_clean/pair-class/metadata.json",
            "row_counts": str(outputs_dir / "tables/dataset_quality/allnli_70_15_15/pair-class/row_counts.csv"),
            "label_distribution": str(outputs_dir / "tables/dataset_quality/allnli_70_15_15/pair-class/label_distribution.csv"),
            "numeric_stats": str(outputs_dir / "tables/dataset_quality/allnli_70_15_15/pair-class/clean_validation_summary.csv"),
            "examples": str(outputs_dir / "tables/dataset_quality/allnli_70_15_15/pair-class/duplicate_examples_fix_clean.csv"),
            "figures": [
                str(outputs_dir / "figures/dataset_quality/allnli_70_15_15/pair-class/label_distribution.png"),
                str(outputs_dir / "figures/dataset_quality/allnli_70_15_15/pair-class/token_length_distribution.png"),
            ],
        },
        "models": {
            name: {
                "metrics": str(outputs_dir / folder / "metrics.json"),
                "training_metadata": str(outputs_dir / folder / "training_metadata.json"),
                "test_predictions": str(outputs_dir / folder / "test_predictions.csv"),
            }
            for name, folder in MODEL_OUTPUTS.items()
        },
        "final_tables": {
            "model_comparison": str(outputs_dir / "tables/final_model_summary.csv"),
            "training_process": str(outputs_dir / "tables/training_process_summary.csv"),
        },
    }


def main() -> None:
    args = parse_args()
    outputs_dir = Path(args.outputs_dir)
    table_dir = Path(args.output_table_dir)
    table_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    training_rows = []
    for name, folder in MODEL_OUTPUTS.items():
        output_dir = outputs_dir / folder
        metrics = read_json(output_dir / "metrics.json")
        if metrics is not None:
            rows.append(model_row(name, metrics))
        train = training_row(name, output_dir)
        if train is not None:
            training_rows.append(train)

    if rows:
        pd.DataFrame(rows).to_csv(table_dir / "final_model_summary.csv", index=False)
    if training_rows:
        pd.DataFrame(training_rows).to_csv(table_dir / "training_process_summary.csv", index=False)
    save_json(manifest(outputs_dir), outputs_dir / "reports" / "barem_artifact_manifest.json")
    print(f"Saved tables -> {table_dir}")
    print(f"Saved manifest -> {outputs_dir / 'reports' / 'barem_artifact_manifest.json'}")


if __name__ == "__main__":
    main()



