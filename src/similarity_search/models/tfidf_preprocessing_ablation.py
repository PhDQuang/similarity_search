"""Run a small TF-IDF preprocessing ablation on AllNLI pair-class."""

from __future__ import annotations

import argparse
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from similarity_search.models.tfidf_baseline import (
    POSITIVE_LABEL,
    choose_threshold,
    fit_vectorizer,
    load_split,
    pair_metrics,
    pair_scores,
    retrieval_metrics,
    validate_frame,
)


VariantFn = Callable[[pd.Series], pd.Series]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="data/processed/allnli/pair-class",
        help="Directory containing train/dev/test parquet or CSV files.",
    )
    parser.add_argument(
        "--output",
        default="outputs/tables/tfidf_preprocessing_ablation.csv",
        help="CSV path for the ablation table.",
    )
    parser.add_argument("--max-features", type=int, default=50_000)
    parser.add_argument("--min-df", type=int, default=2)
    parser.add_argument("--retrieval-pool-size", type=int, default=20)
    parser.add_argument("--max-retrieval-queries", type=int, default=1_000)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def identity(series: pd.Series) -> pd.Series:
    return series.fillna("").astype(str)


def lowercase(series: pd.Series) -> pd.Series:
    return identity(series).str.lower()


def no_punctuation(series: pd.Series) -> pd.Series:
    return lowercase(series).str.replace(r"[^\w\s]", " ", regex=True).str.replace(
        r"\s+",
        " ",
        regex=True,
    )


def no_digits_or_punctuation(series: pd.Series) -> pd.Series:
    return no_punctuation(series).str.replace(r"\d+", " ", regex=True).str.replace(
        r"\s+",
        " ",
        regex=True,
    )


def apply_variant(frame: pd.DataFrame, transform: VariantFn) -> pd.DataFrame:
    result = frame.copy()
    result["premise_clean"] = transform(result["premise_clean"])
    result["hypothesis_clean"] = transform(result["hypothesis_clean"])
    return result


def run_variant(
    name: str,
    transform: VariantFn,
    frames: dict[str, pd.DataFrame],
    max_features: int,
    min_df: int,
    ngram_max: int,
    retrieval_pool_size: int,
    max_retrieval_queries: int,
    seed: int,
) -> dict[str, float | int | str]:
    variant_frames = {split: apply_variant(frame, transform) for split, frame in frames.items()}
    vectorizer = fit_vectorizer(
        variant_frames["train"],
        max_features=max_features,
        min_df=min_df,
        ngram_max=ngram_max,
    )
    dev_scores = pair_scores(vectorizer, variant_frames["dev"])
    test_scores = pair_scores(vectorizer, variant_frames["test"])
    dev_targets = (variant_frames["dev"]["label_name"] == POSITIVE_LABEL).to_numpy()
    test_targets = (variant_frames["test"]["label_name"] == POSITIVE_LABEL).to_numpy()
    threshold, dev_f1 = choose_threshold(dev_targets, dev_scores)
    pair = pair_metrics(test_targets, test_scores, threshold)
    retrieval = retrieval_metrics(
        vectorizer,
        variant_frames["test"],
        retrieval_pool_size,
        max_retrieval_queries,
        seed,
    )
    return {
        "variant": name,
        "ngram_range": f"1-{ngram_max}",
        "vocabulary_size": len(vectorizer.vocabulary_),
        "dev_threshold": threshold,
        "dev_f1": dev_f1,
        "test_accuracy": pair["accuracy"],
        "test_precision": pair["precision"],
        "test_recall": pair["recall"],
        "test_f1": pair["f1"],
        "test_roc_auc": pair["roc_auc"],
        "precision_at_1": retrieval["precision_at_1"],
        "recall_at_5": retrieval["recall_at_5"],
        "mrr": retrieval["mrr"],
        "mean_rank": retrieval["mean_rank"],
    }


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    frames = {split: load_split(input_dir, split) for split in ("train", "dev", "test")}
    for split, frame in frames.items():
        validate_frame(frame, split)

    variants: list[tuple[str, VariantFn, int]] = [
        ("original_clean_text_unigram", identity, 1),
        ("lowercase_unigram", lowercase, 1),
        ("lowercase_no_punctuation_unigram", no_punctuation, 1),
        ("lowercase_no_punctuation_no_digits_unigram", no_digits_or_punctuation, 1),
        ("lowercase_no_punctuation_bigram", no_punctuation, 2),
    ]
    rows = [
        run_variant(
            name,
            transform,
            frames,
            args.max_features,
            args.min_df,
            ngram_max,
            args.retrieval_pool_size,
            args.max_retrieval_queries,
            args.seed,
        )
        for name, transform, ngram_max in variants
    ]

    table = pd.DataFrame(rows).sort_values("test_f1", ascending=False)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output_path, index=False)
    print(table.to_string(index=False))
    print(f"Saved ablation table -> {output_path}")


if __name__ == "__main__":
    main()

