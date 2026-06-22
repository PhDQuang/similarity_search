"""TF-IDF preprocessing ablation on the fixed AllNLI 70/15/15 split."""

from __future__ import annotations

import argparse
import re
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from similarity_search_fix.data.text_utils import normalize_text
from similarity_search_fix.models.evaluation import (
    POSITIVE_LABEL,
    choose_threshold,
    evaluate_retrieval_splits,
    load_splits,
    binary_pair_metrics,
    entailment_targets,
    validate_pair_class_frame,
)
from similarity_search_fix.models.train_tfidf import fit_vectorizer, pair_scores, retrieval_pair_scores

VariantFn = Callable[[pd.Series], pd.Series]
PUNCT_RE = re.compile(r"[^\w\s]")
DIGIT_RE = re.compile(r"\d+")
SPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="fix/data/processed/allnli_70_15_15/pair-class")
    parser.add_argument("--output", default="fix/outputs/tables/tfidf_preprocessing_ablation.csv")
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--retrieval-pool-size", type=int, default=20)
    parser.add_argument("--max-retrieval-queries", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def normalized(series: pd.Series) -> pd.Series:
    return series.map(normalize_text)


def remove_punctuation(series: pd.Series) -> pd.Series:
    return normalized(series).str.replace(PUNCT_RE, " ", regex=True).str.replace(SPACE_RE, " ", regex=True).str.strip()


def remove_punctuation_digits(series: pd.Series) -> pd.Series:
    return remove_punctuation(series).str.replace(DIGIT_RE, " ", regex=True).str.replace(SPACE_RE, " ", regex=True).str.strip()


def apply_variant(frame: pd.DataFrame, transform: VariantFn) -> pd.DataFrame:
    result = frame.copy()
    result["premise_clean"] = transform(result["premise"])
    result["hypothesis_clean"] = transform(result["hypothesis"])
    return result


def run_variant(
    name: str,
    transform: VariantFn,
    ngram_max: int,
    frames: dict[str, pd.DataFrame],
    args: argparse.Namespace,
) -> dict[str, float | int | str]:
    variant_frames = {split: apply_variant(frame, transform) for split, frame in frames.items()}
    vectorizer = fit_vectorizer(
        variant_frames["train"],
        max_features=args.max_features,
        min_df=args.min_df,
        ngram_max=ngram_max,
    )
    val_scores = pair_scores(vectorizer, variant_frames["val"])
    test_scores = pair_scores(vectorizer, variant_frames["test"])
    val_targets = entailment_targets(variant_frames["val"])
    test_targets = entailment_targets(variant_frames["test"])
    threshold, val_f1 = choose_threshold(val_targets, val_scores)
    pair = binary_pair_metrics(test_targets, test_scores, threshold)
    retrieval = evaluate_retrieval_splits(
        variant_frames,
        score_pairs=lambda retrieval_frame: retrieval_pair_scores(vectorizer, retrieval_frame),
        pool_size=args.retrieval_pool_size,
        max_queries=args.max_retrieval_queries,
        seed=args.seed,
    )["test"]
    return {
        "variant": name,
        "ngram_range": f"1-{ngram_max}",
        "vocabulary_size": len(vectorizer.vocabulary_),
        "val_threshold": threshold,
        "val_f1": val_f1,
        "test_accuracy": pair["accuracy"],
        "test_precision": pair["precision"],
        "test_recall": pair["recall"],
        "test_f1": pair["f1"],
        "test_roc_auc": pair["roc_auc"],
        "precision_at_1": retrieval["precision_at_1"],
        "recall_at_5": retrieval["recall_at_5"],
        "mrr": retrieval["mrr"],
        "positive_label": POSITIVE_LABEL,
    }


def main() -> None:
    args = parse_args()
    frames = load_splits(args.input_dir)
    for split, frame in frames.items():
        validate_pair_class_frame(frame, split)

    variants: list[tuple[str, VariantFn, int]] = [
        ("lowercase_normalized_unigram", normalized, 1),
        ("lowercase_normalized_bigram", normalized, 2),
        ("lowercase_no_punctuation_unigram", remove_punctuation, 1),
        ("lowercase_no_punctuation_bigram", remove_punctuation, 2),
        ("lowercase_no_punctuation_no_digits_bigram", remove_punctuation_digits, 2),
    ]
    rows = [run_variant(name, transform, ngram_max, frames, args) for name, transform, ngram_max in variants]
    table = pd.DataFrame(rows).sort_values("test_f1", ascending=False)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output, index=False)
    print(table.to_string(index=False))
    print(f"Saved ablation table -> {output}")


if __name__ == "__main__":
    main()

