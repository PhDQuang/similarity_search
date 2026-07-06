"""Fine-tune a Cross-Encoder NLI model on the fixed AllNLI 70/15/15 split."""

from __future__ import annotations

import argparse
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

from similarity_search.models.evaluation import (
    ID2LABEL,
    LABEL2ID,
    POSITIVE_LABEL,
    binary_confusion,
    binary_pair_metrics,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/processed/allnli_70_15_15_clean/pair-class")
    parser.add_argument("--output-dir", default="outputs/cross_encoder_outputs")
    parser.add_argument("--model-dir", default="models/allnli-cross-encoder-nli")
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--num-train-epochs", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=1_000)
    parser.add_argument("--save-steps", type=int, default=1_000)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--early-stopping-threshold", type=float, default=0.0)
    parser.add_argument(
        "--trainer-eval-samples",
        type=int,
        default=20_000,
        help="Rows for evaluation during training. Final val/test metrics use full splits. 0 means full val.",
    )
    parser.add_argument("--retrieval-pool-size", type=int, default=20)
    parser.add_argument("--max-retrieval-queries", type=int, default=0)
    parser.add_argument("--test-sample-size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


def make_training_args(**kwargs: Any) -> TrainingArguments:
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    return TrainingArguments(**kwargs)


def trainer_tokenizer_kwargs(tokenizer: Any) -> dict[str, Any]:
    """Support both old Trainer(tokenizer=...) and new Trainer(processing_class=...)."""
    signature = inspect.signature(Trainer.__init__)
    if "processing_class" in signature.parameters:
        return {"processing_class": tokenizer}
    if "tokenizer" in signature.parameters:
        return {"tokenizer": tokenizer}
    return {}


def to_dataset(frame: pd.DataFrame) -> Dataset:
    data = pd.DataFrame(
        {
            "text_a": frame["premise_clean"].astype(str),
            "text_b": frame["hypothesis_clean"].astype(str),
            "labels": frame["label_name"].map(LABEL2ID).astype(int),
        }
    )
    return Dataset.from_pandas(data, preserve_index=False)


def sample_dataset(dataset: Dataset, max_samples: int, seed: int) -> Dataset:
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


def tokenize_dataset(dataset: Dataset | DatasetDict, tokenizer: Any, max_length: int) -> Dataset | DatasetDict:
    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(batch["text_a"], batch["text_b"], truncation=True, max_length=max_length)

    return dataset.map(tokenize_batch, batched=True, remove_columns=["text_a", "text_b"])


def softmax_np(logits: np.ndarray) -> np.ndarray:
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


def nli_metrics(labels: np.ndarray, logits: np.ndarray, prefix: str) -> dict[str, float]:
    probs = softmax_np(logits)
    preds = probs.argmax(axis=-1)
    return {
        f"{prefix}_accuracy": float(accuracy_score(labels, preds)),
        f"{prefix}_macro_f1": float(f1_score(labels, preds, average="macro")),
        f"{prefix}_entailment_f1": float(
            f1_score(labels == LABEL2ID["entailment"], preds == LABEL2ID["entailment"])
        ),
    }


def cross_encoder_scores(
    model: Any,
    tokenizer: Any,
    device: str,
    frame: pd.DataFrame,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    scores: list[np.ndarray] = []
    left = frame["text_a" if "text_a" in frame.columns else "premise_clean"].astype(str).tolist()
    right = frame["text_b" if "text_b" in frame.columns else "hypothesis_clean"].astype(str).tolist()
    for start in range(0, len(frame), batch_size):
        end = start + batch_size
        encoded = tokenizer(
            left[start:end],
            right[start:end],
            truncation=True,
            padding=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = model(**encoded).logits.detach().cpu().numpy()
        scores.append(softmax_np(logits)[:, LABEL2ID["entailment"]])
    return np.concatenate(scores) if scores else np.asarray([], dtype=float)


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError("CUDA GPU not found. In Kaggle, enable GPU before running this notebook.")

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    checkpoint_dir = model_dir / "checkpoints"
    final_dir = model_dir / "final"
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    frames = load_splits(args.input_dir)
    for split, frame in frames.items():
        validate_pair_class_frame(frame, split)

    raw_dataset = DatasetDict(
        {
            "train": to_dataset(frames["train"]),
            "val": to_dataset(frames["val"]),
            "test": to_dataset(frames["test"]),
        }
    )
    trainer_eval = sample_dataset(raw_dataset["val"], args.trainer_eval_samples, args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenized_train = tokenize_dataset(raw_dataset["train"], tokenizer, args.max_length)
    tokenized_val_eval = tokenize_dataset(trainer_eval, tokenizer, args.max_length)
    tokenized_val_full = tokenize_dataset(raw_dataset["val"], tokenizer, args.max_length)
    tokenized_test_full = tokenize_dataset(raw_dataset["test"], tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        return nli_metrics(np.asarray(labels), np.asarray(logits), "eval")

    fp16 = bool(torch.cuda.is_available())
    training_args = make_training_args(
        output_dir=str(checkpoint_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        fp16=fp16,
        evaluation_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        logging_steps=args.logging_steps,
        load_best_model_at_end=True,
        metric_for_best_model="eval_macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_train,
        eval_dataset=tokenized_val_eval,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.early_stopping_patience,
                early_stopping_threshold=args.early_stopping_threshold,
            )
        ],
        **trainer_tokenizer_kwargs(tokenizer),
    )
    train_result = trainer.train()
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    val_prediction = trainer.predict(tokenized_val_full)
    test_prediction = trainer.predict(tokenized_test_full)
    val_logits = np.asarray(val_prediction.predictions)
    test_logits = np.asarray(test_prediction.predictions)
    val_labels = np.asarray(val_prediction.label_ids)
    test_labels = np.asarray(test_prediction.label_ids)
    val_scores = softmax_np(val_logits)[:, LABEL2ID["entailment"]]
    test_scores = softmax_np(test_logits)[:, LABEL2ID["entailment"]]

    threshold, pair_report = evaluate_pair_splits(frames["val"], frames["test"], val_scores, test_scores)
    retrieval = evaluate_retrieval_splits(
        frames,
        score_pairs=lambda retrieval_frame: cross_encoder_scores(
            model,
            tokenizer,
            str(training_args.device),
            retrieval_frame,
            args.eval_batch_size,
            args.max_length,
        ),
        pool_size=args.retrieval_pool_size,
        max_queries=args.max_retrieval_queries,
        seed=args.seed,
    )
    test5k_frame, test5k_scores, test5k_report = evaluate_test_sample_performance(
        frames["test"],
        score_pairs=lambda sample_frame: cross_encoder_scores(
            model,
            tokenizer,
            str(training_args.device),
            sample_frame,
            args.eval_batch_size,
            args.max_length,
        ),
        threshold=threshold,
        sample_size=args.test_sample_size,
        seed=args.seed,
    )

    test_preds = softmax_np(test_logits).argmax(axis=-1)
    report = classification_report(
        test_labels,
        test_preds,
        target_names=[ID2LABEL[index] for index in range(3)],
        output_dict=True,
        zero_division=0,
    )
    pd.DataFrame(report).T.to_csv(output_dir / "cross_encoder_classification_report.csv")
    pd.DataFrame(
        confusion_matrix(test_labels, test_preds),
        index=[ID2LABEL[index] for index in range(3)],
        columns=[ID2LABEL[index] for index in range(3)],
    ).to_csv(output_dir / "cross_encoder_confusion_matrix.csv")

    metrics: dict[str, Any] = {
        "task": "cross-encoder-nli-reranker-for-semantic-search",
        "fixed_dataset": "AllNLI pair-class full 70/15/15",
        "positive_label": POSITIVE_LABEL,
        "model": {
            "name": "Cross-Encoder NLI",
            "base_model": args.base_model,
            "trained_in_project": True,
            "loss": "CrossEntropyLoss",
            "num_labels": 3,
        },
        "nli": {
            "val": nli_metrics(val_labels, val_logits, "val"),
            "test": nli_metrics(test_labels, test_logits, "test"),
        },
        **pair_report,
        "retrieval": retrieval,
        "test_sample_performance": test5k_report,
    }
    metadata = {
        "base_model": args.base_model,
        "fixed_dataset": "AllNLI pair-class full 70/15/15",
        "label2id": LABEL2ID,
        "train_rows": len(frames["train"]),
        "val_rows": len(frames["val"]),
        "test_rows": len(frames["test"]),
        "epochs": args.num_train_epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_length": args.max_length,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_threshold": args.early_stopping_threshold,
        "fp16": fp16,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_result": json_safe(getattr(train_result, "metrics", {})),
        "final_model_dir": str(final_dir),
    }

    save_json(metrics, output_dir / "metrics.json")
    save_json(metadata, output_dir / "training_metadata.json")
    save_pair_predictions(frames["val"], val_scores, threshold, "entailment_probability", output_dir / "val_predictions.csv")
    save_pair_predictions(frames["test"], test_scores, threshold, "entailment_probability", output_dir / "test_predictions.csv")
    save_json(test5k_report, output_dir / "test5k_performance.json")
    save_pair_predictions(
        test5k_frame,
        test5k_scores,
        threshold,
        "entailment_probability",
        output_dir / "test5k_predictions.csv",
    )
    binary_confusion(entailment_targets(frames["test"]), test_scores, threshold).to_csv(
        output_dir / "binary_confusion_matrix.csv"
    )
    print(json.dumps(json_safe(metadata), indent=2))
    print(f"Saved Cross-Encoder model -> {final_dir}")
    print(f"Saved Cross-Encoder outputs -> {output_dir}")


if __name__ == "__main__":
    main()


