"""Fine-tune MiniLM from pretrained weights on the fixed AllNLI 70/15/15 split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset

from similarity_search_fix.models.evaluation import (
    POSITIVE_LABEL,
    SCORE_BY_LABEL,
    binary_confusion,
    entailment_targets,
    evaluate_pair_splits,
    evaluate_retrieval_splits,
    evaluate_test_sample_performance,
    json_safe,
    load_splits,
    save_json,
    save_pair_predictions,
    validate_pair_class_frame,
)

DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="fix/data/processed/allnli_70_15_15/pair-class")
    parser.add_argument("--output-dir", default="fix/outputs/minilm_finetuned")
    parser.add_argument("--model-dir", default="fix/models/allnli_70_15_15_minilm")
    parser.add_argument("--base-model", default=DEFAULT_MODEL)
    parser.add_argument("--num-train-epochs", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--max-seq-length", type=int, default=128)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=1_000)
    parser.add_argument("--save-steps", type=int, default=1_000)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument(
        "--metric-for-best-model",
        default="eval_fixed-allnli-val_spearman_cosine",
        help="Metric emitted by EmbeddingSimilarityEvaluator for early stopping.",
    )
    parser.add_argument(
        "--trainer-eval-samples",
        type=int,
        default=20_000,
        help="Evaluator rows during training. Final metrics still use full val/test. 0 means full val.",
    )
    parser.add_argument("--retrieval-pool-size", type=int, default=20)
    parser.add_argument("--max-retrieval-queries", type=int, default=0)
    parser.add_argument("--test-sample-size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def to_sentence_transformer_dataset(frame: pd.DataFrame) -> Dataset:
    data = pd.DataFrame(
        {
            "sentence1": frame["premise_clean"].astype(str),
            "sentence2": frame["hypothesis_clean"].astype(str),
            "score": frame["label_name"].map(SCORE_BY_LABEL).astype(float),
        }
    )
    return Dataset.from_pandas(data, preserve_index=False)


def build_evaluator(frame: pd.DataFrame, batch_size: int, max_samples: int, seed: int) -> Any:
    try:
        from sentence_transformers.evaluation import EmbeddingSimilarityEvaluator
    except ImportError:
        from sentence_transformers.sentence_transformer.evaluation import EmbeddingSimilarityEvaluator

    eval_frame = frame
    if max_samples > 0 and len(eval_frame) > max_samples:
        eval_frame = eval_frame.sample(n=max_samples, random_state=seed).reset_index(drop=True)
    return EmbeddingSimilarityEvaluator(
        sentences1=eval_frame["premise_clean"].astype(str).tolist(),
        sentences2=eval_frame["hypothesis_clean"].astype(str).tolist(),
        scores=eval_frame["label_name"].map(SCORE_BY_LABEL).astype(float).tolist(),
        batch_size=batch_size,
        main_similarity="cosine",
        name="fixed-allnli-val",
        show_progress_bar=True,
    )


def precision_flags() -> tuple[bool, bool]:
    if not torch.cuda.is_available():
        return False, False
    bf16_supported = bool(hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
    return (not bf16_supported), bf16_supported


def encode_texts(model: Any, texts: pd.Series, batch_size: int) -> np.ndarray:
    return model.encode(
        texts.fillna("").astype(str).tolist(),
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )


def embedding_pair_scores(model: Any, frame: pd.DataFrame, batch_size: int, left_col: str, right_col: str) -> np.ndarray:
    left_embeddings = encode_texts(model, frame[left_col], batch_size)
    right_embeddings = encode_texts(model, frame[right_col], batch_size)
    return np.sum(left_embeddings * right_embeddings, axis=1)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA GPU not found. In Kaggle, enable GPU before running this notebook.")

    from sentence_transformers import (
        SentenceTransformer,
        SentenceTransformerTrainer,
        SentenceTransformerTrainingArguments,
    )
    from transformers import EarlyStoppingCallback

    try:
        from sentence_transformers.losses import CosineSimilarityLoss
    except ImportError:
        from sentence_transformers.sentence_transformer.losses import CosineSimilarityLoss

    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    checkpoint_dir = model_dir / "checkpoints"
    final_dir = model_dir / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    frames = load_splits(args.input_dir)
    for split, frame in frames.items():
        validate_pair_class_frame(frame, split)

    train_dataset = to_sentence_transformer_dataset(frames["train"])
    evaluator = build_evaluator(
        frames["val"],
        batch_size=args.eval_batch_size,
        max_samples=args.trainer_eval_samples,
        seed=args.seed,
    )

    model = SentenceTransformer(args.base_model)
    model.max_seq_length = args.max_seq_length
    loss = CosineSimilarityLoss(model)
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
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model=args.metric_for_best_model,
        greater_is_better=True,
        logging_strategy="steps",
        logging_steps=args.logging_steps,
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
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        ],
    )
    train_result = trainer.train()
    model.save_pretrained(str(final_dir))
    final_eval = evaluator(model, output_path=str(output_dir))

    val_scores = embedding_pair_scores(model, frames["val"], args.eval_batch_size, "premise_clean", "hypothesis_clean")
    test_scores = embedding_pair_scores(model, frames["test"], args.eval_batch_size, "premise_clean", "hypothesis_clean")
    threshold, pair_report = evaluate_pair_splits(frames["val"], frames["test"], val_scores, test_scores)
    retrieval = evaluate_retrieval_splits(
        frames,
        score_pairs=lambda retrieval_frame: embedding_pair_scores(
            model,
            retrieval_frame,
            args.eval_batch_size,
            "text_a",
            "text_b",
        ),
        pool_size=args.retrieval_pool_size,
        max_queries=args.max_retrieval_queries,
        seed=args.seed,
    )
    test5k_frame, test5k_scores, test5k_report = evaluate_test_sample_performance(
        frames["test"],
        score_pairs=lambda sample_frame: embedding_pair_scores(
            model,
            sample_frame,
            args.eval_batch_size,
            "premise_clean",
            "hypothesis_clean",
        ),
        threshold=threshold,
        sample_size=args.test_sample_size,
        seed=args.seed,
    )

    metrics: dict[str, Any] = {
        "task": "entailment-as-semantic-similarity",
        "fixed_dataset": "AllNLI pair-class full 70/15/15",
        "positive_label": POSITIVE_LABEL,
        "model": {
            "name": "Fine-tuned MiniLM",
            "base_model": args.base_model,
            "trained_in_project": True,
            "embedding_dimension": int(model.get_sentence_embedding_dimension()),
            "loss": "CosineSimilarityLoss",
            "score_mapping": SCORE_BY_LABEL,
        },
        **pair_report,
        "retrieval": retrieval,
        "test_sample_performance": test5k_report,
    }
    metadata = {
        "base_model": args.base_model,
        "fixed_dataset": "AllNLI pair-class full 70/15/15",
        "train_rows": len(frames["train"]),
        "val_rows": len(frames["val"]),
        "test_rows": len(frames["test"]),
        "loss": "CosineSimilarityLoss",
        "epochs": args.num_train_epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "max_seq_length": args.max_seq_length,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_threshold": args.early_stopping_threshold,
        "metric_for_best_model": args.metric_for_best_model,
        "fp16": fp16,
        "bf16": bf16,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_result": json_safe(getattr(train_result, "metrics", {})),
        "trainer_eval": json_safe(final_eval),
        "final_model_dir": str(final_dir),
    }

    save_json(metrics, output_dir / "metrics.json")
    save_json(metadata, output_dir / "training_metadata.json")
    save_pair_predictions(frames["val"], val_scores, threshold, "minilm_cosine", output_dir / "val_predictions.csv")
    save_pair_predictions(frames["test"], test_scores, threshold, "minilm_cosine", output_dir / "test_predictions.csv")
    save_json(test5k_report, output_dir / "test5k_performance.json")
    save_pair_predictions(
        test5k_frame,
        test5k_scores,
        threshold,
        "minilm_cosine",
        output_dir / "test5k_predictions.csv",
    )
    binary_confusion(entailment_targets(frames["test"]), test_scores, threshold).to_csv(
        output_dir / "binary_confusion_matrix.csv"
    )
    print(json.dumps(json_safe(metadata), indent=2))
    print(f"Saved MiniLM model -> {final_dir}")
    print(f"Saved MiniLM outputs -> {output_dir}")


if __name__ == "__main__":
    main()

