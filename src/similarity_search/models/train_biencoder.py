"""Fine-tune a MiniLM bi-encoder on the AllNLI pair subset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import torch
from datasets import Dataset, load_dataset
from sentence_transformers import (
    SentenceTransformer,
    SentenceTransformerTrainer,
    SentenceTransformerTrainingArguments,
)
from sentence_transformers.sentence_transformer.evaluation import (
    EmbeddingSimilarityEvaluator,
)
from sentence_transformers.sentence_transformer.losses import (
    MultipleNegativesRankingLoss,
)
from sentence_transformers.sentence_transformer.training_args import BatchSamplers

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_DATASET = "sentence-transformers/all-nli"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=DEFAULT_MODEL)
    parser.add_argument("--dataset-name", default=DEFAULT_DATASET)
    parser.add_argument(
        "--output-dir",
        default="models/allnli-minilm-biencoder",
        help="Root directory for checkpoints, final model, and training metadata.",
    )
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=1_000)
    parser.add_argument("--save-steps", type=int, default=1_000)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=0,
        help="Optional sampled train size; 0 uses the full train split.",
    )
    parser.add_argument(
        "--max-eval-samples",
        type=int,
        default=10_000,
        help="Sample size from pair-score/dev for evaluator; 0 uses all rows.",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allow-cpu",
        action="store_true",
        help="Allow training without CUDA. Intended only for tiny smoke tests.",
    )
    parser.add_argument(
        "--push-to-hub",
        action="store_true",
        help="Push the final model using the HF_TOKEN environment variable.",
    )
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--hub-private-repo", action="store_true")
    return parser.parse_args()


def sample_dataset(dataset: Dataset, max_samples: int, seed: int) -> Dataset:
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


def prepare_pair_dataset(dataset: Dataset) -> Dataset:
    required = {"anchor", "positive"}
    missing = sorted(required - set(dataset.column_names))
    if missing:
        raise ValueError(f"AllNLI pair dataset is missing: {', '.join(missing)}")

    removable = [column for column in dataset.column_names if column not in required]
    if removable:
        dataset = dataset.remove_columns(removable)
    return dataset.filter(
        lambda row: bool(str(row["anchor"]).strip())
        and bool(str(row["positive"]).strip())
    )


def build_evaluator(
    dataset_name: str,
    max_samples: int,
    batch_size: int,
    seed: int,
) -> tuple[EmbeddingSimilarityEvaluator, int]:
    dataset = load_dataset(dataset_name, "pair-score", split="dev")
    dataset = sample_dataset(dataset, max_samples, seed)
    required = {"sentence1", "sentence2", "score"}
    missing = sorted(required - set(dataset.column_names))
    if missing:
        raise ValueError(f"AllNLI pair-score dataset is missing: {', '.join(missing)}")

    evaluator = EmbeddingSimilarityEvaluator(
        sentences1=list(dataset["sentence1"]),
        sentences2=list(dataset["sentence2"]),
        scores=[float(score) for score in dataset["score"]],
        batch_size=batch_size,
        main_similarity="cosine",
        name="allnli-pair-score-dev",
        show_progress_bar=True,
    )
    return evaluator, len(dataset)


def precision_flags() -> tuple[bool, bool]:
    if not torch.cuda.is_available():
        return False, False
    bf16_supported = bool(
        hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported()
    )
    return (not bf16_supported), bf16_supported


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA GPU not found. In Kaggle, select Settings > Accelerator > GPU, "
            "then restart the session."
        )
    if args.push_to_hub and not args.hub_model_id:
        raise ValueError("--hub-model-id is required with --push-to-hub")

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    final_dir = output_dir / "final"
    output_dir.mkdir(parents=True, exist_ok=True)

    train_dataset = load_dataset(args.dataset_name, "pair", split="train")
    train_dataset = prepare_pair_dataset(train_dataset)
    train_dataset = sample_dataset(
        train_dataset,
        args.max_train_samples,
        args.seed,
    )
    evaluator, eval_rows = build_evaluator(
        args.dataset_name,
        args.max_eval_samples,
        args.eval_batch_size,
        args.seed,
    )

    model = SentenceTransformer(args.model_name)
    model.max_seq_length = args.max_seq_length
    loss = MultipleNegativesRankingLoss(model)
    fp16, bf16 = precision_flags()

    training_args = SentenceTransformerTrainingArguments(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        fp16=fp16,
        bf16=bf16,
        batch_sampler=BatchSamplers.NO_DUPLICATES,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = SentenceTransformerTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        loss=loss,
        evaluator=evaluator,
    )
    trainer.train()
    model.save_pretrained(str(final_dir))
    final_evaluation = evaluator(model, output_path=str(output_dir))

    metadata: dict[str, Any] = {
        "base_model": args.model_name,
        "dataset": args.dataset_name,
        "train_subset": "pair",
        "train_rows": len(train_dataset),
        "evaluation_subset": "pair-score/dev",
        "evaluation_rows": eval_rows,
        "loss": "MultipleNegativesRankingLoss",
        "epochs": args.num_train_epochs,
        "batch_size": args.batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "effective_batch_size_per_process": (
            args.batch_size * args.gradient_accumulation_steps
        ),
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "max_seq_length": args.max_seq_length,
        "fp16": fp16,
        "bf16": bf16,
        "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "final_model_dir": str(final_dir),
        "final_evaluation": json_safe(final_evaluation),
    }
    (output_dir / "training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    if args.push_to_hub:
        model.push_to_hub(
            args.hub_model_id,
            private=args.hub_private_repo,
            token=os.environ.get("HF_TOKEN"),
        )

    print(json.dumps(metadata, indent=2))
    print(f"Saved final model -> {final_dir}")


if __name__ == "__main__":
    main()
