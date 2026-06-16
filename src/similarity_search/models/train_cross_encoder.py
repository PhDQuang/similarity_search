"""Train a Cross-Encoder NLI reranker for semantic similarity search."""

from __future__ import annotations

import argparse
import inspect
import json
import os
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from datasets import Dataset, DatasetDict, load_dataset
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_recall_fscore_support,
    roc_auc_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
    set_seed,
)

LABEL2ID = {"entailment": 0, "neutral": 1, "contradiction": 2}
ID2LABEL = {value: key for key, value in LABEL2ID.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset-name", default="sentence-transformers/all-nli")
    parser.add_argument("--train-dataset-config", default="pair-class")
    parser.add_argument("--benchmark-dataset-name", default="phdquang/allnli-pair-class-processed")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--base-model", default="distilbert-base-uncased")
    parser.add_argument("--output-dir", default="/kaggle/working/allnli-cross-encoder-nli")
    parser.add_argument("--result-dir", default="/kaggle/working/cross_encoder_outputs")
    parser.add_argument("--max-train-samples", type=int, default=300_000)
    parser.add_argument("--max-dev-samples", type=int, default=20_000)
    parser.add_argument("--max-test-samples", type=int, default=20_000)
    parser.add_argument("--num-train-epochs", type=float, default=1.0)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-length", type=int, default=128)
    parser.add_argument("--logging-steps", type=int, default=100)
    parser.add_argument("--eval-steps", type=int, default=1000)
    parser.add_argument("--save-steps", type=int, default=1000)
    parser.add_argument("--save-total-limit", type=int, default=2)
    parser.add_argument("--rerank-queries", type=int, default=1000)
    parser.add_argument("--rerank-pool-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-cpu", action="store_true")
    parser.add_argument("--push-to-hub", action="store_true")
    parser.add_argument("--hub-model-id", default=None)
    parser.add_argument("--hub-private-repo", action="store_true")
    return parser.parse_args()


def json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        return value.item()
    return value


def load_pair_class_dataset(dataset_name: str, dataset_config: str | None) -> DatasetDict:
    if dataset_config:
        return load_dataset(dataset_name, dataset_config)
    return load_dataset(dataset_name)


def get_split(dataset: DatasetDict, names: tuple[str, ...]) -> Dataset:
    for name in names:
        if name in dataset:
            return dataset[name]
    raise KeyError(f"Missing split. Tried {names}. Available: {list(dataset.keys())}")


def label_to_id(value: Any) -> int:
    if isinstance(value, str):
        value = value.strip().lower()
        if value in LABEL2ID:
            return LABEL2ID[value]
        if value.isdigit():
            return int(value)
    return int(value)


def choose_text_columns(dataset: Dataset) -> tuple[str, str]:
    columns = set(dataset.column_names)
    if {"premise_clean", "hypothesis_clean"}.issubset(columns):
        return "premise_clean", "hypothesis_clean"
    if {"premise", "hypothesis"}.issubset(columns):
        return "premise", "hypothesis"
    if {"sentence1", "sentence2"}.issubset(columns):
        return "sentence1", "sentence2"
    raise ValueError(f"Cannot infer text columns from: {dataset.column_names}")


def normalize_dataset(dataset: Dataset, text_a_col: str, text_b_col: str) -> Dataset:
    label_source = "label" if "label" in dataset.column_names else "label_name"

    def mapper(batch: dict[str, list[Any]]) -> dict[str, list[Any]]:
        return {
            "text_a": [str(value).strip() for value in batch[text_a_col]],
            "text_b": [str(value).strip() for value in batch[text_b_col]],
            "labels": [label_to_id(value) for value in batch[label_source]],
        }

    return dataset.map(mapper, batched=True, remove_columns=dataset.column_names)


def sample_dataset(dataset: Dataset, max_samples: int, seed: int) -> Dataset:
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    return dataset.shuffle(seed=seed).select(range(max_samples))


def softmax_np(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=-1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=-1, keepdims=True)


def make_training_args(**kwargs: Any) -> TrainingArguments:
    signature = inspect.signature(TrainingArguments.__init__)
    if "eval_strategy" in signature.parameters and "evaluation_strategy" in kwargs:
        kwargs["eval_strategy"] = kwargs.pop("evaluation_strategy")
    return TrainingArguments(**kwargs)


def choose_threshold(y_true: np.ndarray, scores: np.ndarray) -> tuple[float, float]:
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    if len(thresholds) == 0:
        return 0.5, 0.0
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(
        precision[:-1] + recall[:-1],
        1e-12,
    )
    best_index = int(np.argmax(f1))
    return float(thresholds[best_index]), float(f1[best_index])


def binary_similarity_metrics(
    labels: np.ndarray,
    entailment_probs: np.ndarray,
    threshold: float,
) -> dict[str, float]:
    y_true = labels == LABEL2ID["entailment"]
    y_pred = entailment_probs >= threshold
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="binary",
        zero_division=0,
    )
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(roc_auc_score(y_true, entailment_probs)),
        "average_precision": float(average_precision_score(y_true, entailment_probs)),
        "mean_positive_score": float(entailment_probs[y_true].mean()),
        "mean_negative_score": float(entailment_probs[~y_true].mean()),
    }


def build_dataset(args: argparse.Namespace) -> DatasetDict:
    raw = load_pair_class_dataset(args.train_dataset_name, args.train_dataset_config or None)
    train_raw = get_split(raw, ("train",))
    dev_raw = get_split(raw, ("dev", "validation"))
    test_raw = get_split(raw, ("test",))

    text_a_col, text_b_col = choose_text_columns(train_raw)
    train = normalize_dataset(train_raw, text_a_col, text_b_col)
    dev = normalize_dataset(dev_raw, text_a_col, text_b_col)
    test = normalize_dataset(test_raw, text_a_col, text_b_col)

    return DatasetDict(
        {
            "train": sample_dataset(train, args.max_train_samples, args.seed),
            "dev": sample_dataset(dev, args.max_dev_samples, args.seed),
            "test": sample_dataset(test, args.max_test_samples, args.seed),
        }
    )


def tokenize_dataset(dataset: DatasetDict, tokenizer: Any, max_length: int) -> DatasetDict:
    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text_a"],
            batch["text_b"],
            truncation=True,
            max_length=max_length,
        )

    return dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text_a", "text_b"],
    )


def predict_entailment_probabilities(
    trainer: Trainer,
    tokenizer: Any,
    frame: pd.DataFrame,
    max_length: int,
) -> np.ndarray:
    pair_dataset = Dataset.from_pandas(frame[["text_a", "text_b"]], preserve_index=False)

    def tokenize_batch(batch: dict[str, list[str]]) -> dict[str, Any]:
        return tokenizer(
            batch["text_a"],
            batch["text_b"],
            truncation=True,
            max_length=max_length,
        )

    pair_dataset = pair_dataset.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text_a", "text_b"],
    )
    predictions = trainer.predict(pair_dataset)
    probs = softmax_np(predictions.predictions)
    return probs[:, LABEL2ID["entailment"]]


def rerank_retrieval_metrics(
    trainer: Trainer,
    tokenizer: Any,
    frame: pd.DataFrame,
    max_length: int,
    max_queries: int,
    pool_size: int,
    seed: int,
) -> tuple[dict[str, float | int], pd.DataFrame]:
    rng = np.random.default_rng(seed)
    positives = frame[frame["labels"] == LABEL2ID["entailment"]].reset_index(drop=True)
    negatives = frame[frame["labels"] != LABEL2ID["entailment"]].reset_index(drop=True)
    if positives.empty or negatives.empty:
        raise ValueError("Retrieval evaluation requires entailment and non-entailment rows")
    if max_queries > 0 and len(positives) > max_queries:
        selected = rng.choice(len(positives), size=max_queries, replace=False)
        positives = positives.iloc[selected].reset_index(drop=True)

    rows: list[dict[str, Any]] = []
    candidate_slots: list[int] = []
    relevant_slots: list[int] = []
    replace = len(negatives) < pool_size - 1

    for query_id, row in positives.iterrows():
        query = row["text_b"]
        positive_candidate = row["text_a"]
        negative_indices = rng.choice(len(negatives), size=pool_size - 1, replace=replace)
        candidate_texts = [positive_candidate] + negatives.iloc[negative_indices]["text_a"].tolist()
        permutation = rng.permutation(pool_size)
        relevant_slot = int(np.flatnonzero(permutation == 0)[0])

        for slot, candidate_index in enumerate(permutation):
            rows.append(
                {
                    "query_id": query_id,
                    "text_a": candidate_texts[candidate_index],
                    "text_b": query,
                }
            )
            candidate_slots.append(slot)
            relevant_slots.append(relevant_slot)

    eval_frame = pd.DataFrame(rows)
    eval_frame["entailment_probability"] = predict_entailment_probabilities(
        trainer,
        tokenizer,
        eval_frame,
        max_length,
    )
    eval_frame["candidate_slot"] = candidate_slots
    eval_frame["relevant_slot"] = relevant_slots

    ranks: list[int] = []
    for _, group in eval_frame.groupby("query_id", sort=False):
        ranked = group.sort_values("entailment_probability", ascending=False).reset_index(drop=True)
        relevant_slot = int(group["relevant_slot"].iloc[0])
        rank = int(np.flatnonzero(ranked["candidate_slot"].to_numpy() == relevant_slot)[0]) + 1
        ranks.append(rank)

    rank_array = np.asarray(ranks)
    hit_at_1 = float(np.mean(rank_array <= 1))
    hit_at_5 = float(np.mean(rank_array <= min(5, pool_size)))
    metrics = {
        "queries": int(len(rank_array)),
        "candidate_pool_size": int(pool_size),
        "precision_at_1": hit_at_1,
        "precision_at_5": float(hit_at_5 / min(5, pool_size)),
        "recall_at_5": hit_at_5,
        "mrr": float(np.mean(1.0 / rank_array)),
        "mean_rank": float(np.mean(rank_array)),
    }
    return metrics, eval_frame


def benchmark_evaluation(
    args: argparse.Namespace,
    trainer: Trainer,
    tokenizer: Any,
    threshold: float,
) -> dict[str, Any] | None:
    if args.skip_benchmark:
        return None
    try:
        benchmark_raw = load_dataset(args.benchmark_dataset_name)
        benchmark_test_raw = get_split(benchmark_raw, ("test",))
        text_a_col, text_b_col = choose_text_columns(benchmark_test_raw)
        benchmark_test = normalize_dataset(benchmark_test_raw, text_a_col, text_b_col)
        benchmark_test = sample_dataset(benchmark_test, args.max_test_samples, args.seed)
        benchmark_dataset = DatasetDict({"test": benchmark_test})
        benchmark_tokenized = tokenize_dataset(benchmark_dataset, tokenizer, args.max_length)["test"]
        benchmark_nli = trainer.evaluate(benchmark_tokenized, metric_key_prefix="benchmark_test")
        benchmark_prediction = trainer.predict(benchmark_tokenized)
        probs = softmax_np(benchmark_prediction.predictions)
        labels = np.asarray(benchmark_prediction.label_ids)
        binary = binary_similarity_metrics(labels, probs[:, LABEL2ID["entailment"]], threshold)
        return {"nli": benchmark_nli, "binary_similarity": binary}
    except Exception as exc:
        return {"error": repr(exc)}


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available() and not args.allow_cpu:
        raise RuntimeError(
            "CUDA GPU not found. On Kaggle, set Settings > Accelerator > GPU."
        )
    if args.push_to_hub and not args.hub_model_id:
        raise ValueError("--hub-model-id is required with --push-to-hub")

    set_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    output_dir = Path(args.output_dir)
    checkpoint_dir = output_dir / "checkpoints"
    final_model_dir = output_dir / "final"
    result_dir = Path(args.result_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    dataset = build_dataset(args)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, use_fast=True)
    tokenized = tokenize_dataset(dataset, tokenizer, args.max_length)

    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        num_labels=3,
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        probs = softmax_np(logits)
        return {
            "accuracy": float(accuracy_score(labels, preds)),
            "macro_f1": float(f1_score(labels, preds, average="macro")),
            "entailment_f1": float(
                f1_score(
                    labels == LABEL2ID["entailment"],
                    preds == LABEL2ID["entailment"],
                )
            ),
            "entailment_roc_auc": float(
                roc_auc_score(labels == LABEL2ID["entailment"], probs[:, LABEL2ID["entailment"]])
            ),
        }

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
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        report_to="none",
        seed=args.seed,
        data_seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["dev"],
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics,
    )

    train_result = trainer.train()
    trainer.save_model(str(final_model_dir))
    tokenizer.save_pretrained(str(final_model_dir))

    dev_metrics = trainer.evaluate(tokenized["dev"], metric_key_prefix="dev")
    test_metrics = trainer.evaluate(tokenized["test"], metric_key_prefix="test")

    test_prediction = trainer.predict(tokenized["test"])
    test_logits = test_prediction.predictions
    test_labels = np.asarray(test_prediction.label_ids)
    test_probs = softmax_np(test_logits)
    test_preds = np.argmax(test_probs, axis=-1)

    report = classification_report(
        test_labels,
        test_preds,
        target_names=[ID2LABEL[index] for index in range(3)],
        output_dict=True,
        zero_division=0,
    )
    confusion = confusion_matrix(test_labels, test_preds)
    pd.DataFrame(report).T.to_csv(result_dir / "cross_encoder_classification_report.csv")
    pd.DataFrame(
        confusion,
        index=[ID2LABEL[index] for index in range(3)],
        columns=[ID2LABEL[index] for index in range(3)],
    ).to_csv(result_dir / "cross_encoder_confusion_matrix.csv")

    dev_prediction = trainer.predict(tokenized["dev"])
    dev_probs = softmax_np(dev_prediction.predictions)
    dev_labels = np.asarray(dev_prediction.label_ids)
    dev_entailment_probs = dev_probs[:, LABEL2ID["entailment"]]
    threshold, best_dev_f1 = choose_threshold(
        dev_labels == LABEL2ID["entailment"],
        dev_entailment_probs,
    )
    test_entailment_probs = test_probs[:, LABEL2ID["entailment"]]
    dev_binary_metrics = binary_similarity_metrics(dev_labels, dev_entailment_probs, threshold)
    test_binary_metrics = binary_similarity_metrics(test_labels, test_entailment_probs, threshold)

    test_frame = dataset["test"].to_pandas()
    test_frame["label_name"] = test_frame["labels"].map(ID2LABEL)
    test_frame["pred_label_id"] = test_preds
    test_frame["pred_label_name"] = test_frame["pred_label_id"].map(ID2LABEL)
    test_frame["entailment_probability"] = test_entailment_probs
    test_frame["predicted_similar"] = test_frame["entailment_probability"] >= threshold
    test_frame.to_csv(result_dir / "cross_encoder_test_predictions.csv", index=False)

    retrieval_metrics, retrieval_predictions = rerank_retrieval_metrics(
        trainer=trainer,
        tokenizer=tokenizer,
        frame=dataset["test"].to_pandas(),
        max_length=args.max_length,
        max_queries=args.rerank_queries,
        pool_size=args.rerank_pool_size,
        seed=args.seed,
    )
    retrieval_predictions.to_csv(result_dir / "cross_encoder_retrieval_predictions.csv", index=False)

    benchmark_metrics = benchmark_evaluation(args, trainer, tokenizer, threshold)

    metadata = {
        "task": "cross-encoder-nli-reranker-for-semantic-search",
        "train_dataset_name": args.train_dataset_name,
        "train_dataset_config": args.train_dataset_config,
        "benchmark_dataset_name": None if args.skip_benchmark else args.benchmark_dataset_name,
        "base_model": args.base_model,
        "label2id": LABEL2ID,
        "train_rows": len(dataset["train"]),
        "dev_rows": len(dataset["dev"]),
        "test_rows": len(dataset["test"]),
        "epochs": args.num_train_epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "max_length": args.max_length,
        "fp16": fp16,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "train_result": json_safe(train_result.metrics),
        "dev_nli_metrics": json_safe(dev_metrics),
        "test_nli_metrics": json_safe(test_metrics),
        "selected_similarity_threshold": threshold,
        "selected_similarity_threshold_dev_f1": best_dev_f1,
        "dev_binary_similarity": dev_binary_metrics,
        "test_binary_similarity": test_binary_metrics,
        "retrieval_rerank": retrieval_metrics,
        "benchmark_metrics": json_safe(benchmark_metrics),
        "final_model_dir": str(final_model_dir),
    }
    (result_dir / "cross_encoder_training_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))

    shutil.make_archive(str(output_dir), "zip", str(output_dir))
    shutil.make_archive(str(result_dir), "zip", str(result_dir))

    if args.push_to_hub:
        model.push_to_hub(
            args.hub_model_id,
            private=args.hub_private_repo,
            token=os.environ.get("HF_TOKEN"),
        )
        tokenizer.push_to_hub(
            args.hub_model_id,
            private=args.hub_private_repo,
            token=os.environ.get("HF_TOKEN"),
        )

    print(f"Saved final model -> {final_model_dir}")
    print(f"Saved outputs -> {result_dir}")


if __name__ == "__main__":
    main()
