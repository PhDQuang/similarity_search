"""Download and prepare AllNLI subsets for later modeling and EDA."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any

from similarity_search.data.text_utils import (
    char_count,
    lexical_overlap,
    normalize_text,
    token_count,
)

if TYPE_CHECKING:
    import pandas as pd

DATASET_NAME = "sentence-transformers/all-nli"
LABEL_NAMES = {0: "entailment", 1: "neutral", 2: "contradiction"}
SPLITS = ("train", "dev", "test")
SUBSETS = ("pair-class", "pair-score", "pair", "triplet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subsets",
        nargs="+",
        default=["pair-class"],
        choices=SUBSETS,
        help="AllNLI subset(s) to prepare.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/allnli",
        help="Directory for processed subset files.",
    )
    parser.add_argument(
        "--cache-dir",
        default="data/raw/hf_cache",
        help="Hugging Face dataset cache directory.",
    )
    parser.add_argument(
        "--save-format",
        default="parquet",
        choices=("parquet", "csv"),
        help="Output file format.",
    )
    parser.add_argument(
        "--max-rows-per-split",
        type=int,
        default=None,
        help="Optional small sample size per split for quick experiments.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Seed used when sampling rows.",
    )
    parser.add_argument(
        "--skip-features",
        action="store_true",
        help="Only normalize columns; do not add length/overlap features.",
    )
    return parser.parse_args()


def load_subset(subset: str, cache_dir: str | None) -> Any:
    from datasets import load_dataset

    return load_dataset(DATASET_NAME, subset, cache_dir=cache_dir)


def maybe_sample_frame(df: pd.DataFrame, max_rows: int | None, seed: int) -> pd.DataFrame:
    if max_rows is None or len(df) <= max_rows:
        return df.reset_index(drop=True)
    return df.sample(n=max_rows, random_state=seed).reset_index(drop=True)


def add_pair_class_features(df: pd.DataFrame, skip_features: bool) -> pd.DataFrame:
    df = df.copy()
    df["premise_clean"] = df["premise"].map(normalize_text)
    df["hypothesis_clean"] = df["hypothesis"].map(normalize_text)
    df["label_name"] = df["label"].map(LABEL_NAMES)

    if not skip_features:
        df["premise_char_len"] = df["premise_clean"].map(char_count)
        df["hypothesis_char_len"] = df["hypothesis_clean"].map(char_count)
        df["premise_token_len"] = df["premise_clean"].map(token_count)
        df["hypothesis_token_len"] = df["hypothesis_clean"].map(token_count)
        df["lexical_overlap"] = [
            lexical_overlap(a, b) for a, b in zip(df["premise_clean"], df["hypothesis_clean"])
        ]
    return df


def add_pair_score_features(df: pd.DataFrame, skip_features: bool) -> pd.DataFrame:
    df = df.copy()
    df["sentence1_clean"] = df["sentence1"].map(normalize_text)
    df["sentence2_clean"] = df["sentence2"].map(normalize_text)
    df["score_label"] = df["score"].map({1.0: "entailment", 0.5: "neutral", 0.0: "contradiction"})

    if not skip_features:
        df["sentence1_char_len"] = df["sentence1_clean"].map(char_count)
        df["sentence2_char_len"] = df["sentence2_clean"].map(char_count)
        df["sentence1_token_len"] = df["sentence1_clean"].map(token_count)
        df["sentence2_token_len"] = df["sentence2_clean"].map(token_count)
        df["lexical_overlap"] = [
            lexical_overlap(a, b) for a, b in zip(df["sentence1_clean"], df["sentence2_clean"])
        ]
    return df


def add_pair_features(df: pd.DataFrame, skip_features: bool) -> pd.DataFrame:
    df = df.copy()
    df["anchor_clean"] = df["anchor"].map(normalize_text)
    df["positive_clean"] = df["positive"].map(normalize_text)

    if not skip_features:
        df["anchor_char_len"] = df["anchor_clean"].map(char_count)
        df["positive_char_len"] = df["positive_clean"].map(char_count)
        df["anchor_token_len"] = df["anchor_clean"].map(token_count)
        df["positive_token_len"] = df["positive_clean"].map(token_count)
        df["anchor_positive_overlap"] = [
            lexical_overlap(a, b) for a, b in zip(df["anchor_clean"], df["positive_clean"])
        ]
    return df


def add_triplet_features(df: pd.DataFrame, skip_features: bool) -> pd.DataFrame:
    df = df.copy()
    df["anchor_clean"] = df["anchor"].map(normalize_text)
    df["positive_clean"] = df["positive"].map(normalize_text)
    df["negative_clean"] = df["negative"].map(normalize_text)

    if not skip_features:
        for column in ("anchor", "positive", "negative"):
            clean_column = f"{column}_clean"
            df[f"{column}_char_len"] = df[clean_column].map(char_count)
            df[f"{column}_token_len"] = df[clean_column].map(token_count)
        df["anchor_positive_overlap"] = [
            lexical_overlap(a, b) for a, b in zip(df["anchor_clean"], df["positive_clean"])
        ]
        df["anchor_negative_overlap"] = [
            lexical_overlap(a, b) for a, b in zip(df["anchor_clean"], df["negative_clean"])
        ]
    return df


def prepare_frame(df: pd.DataFrame, subset: str, split: str, skip_features: bool) -> pd.DataFrame:
    if subset == "pair-class":
        df = add_pair_class_features(df, skip_features)
    elif subset == "pair-score":
        df = add_pair_score_features(df, skip_features)
    elif subset == "pair":
        df = add_pair_features(df, skip_features)
    elif subset == "triplet":
        df = add_triplet_features(df, skip_features)
    else:
        raise ValueError(f"Unsupported subset: {subset}")

    df.insert(0, "dataset_name", DATASET_NAME)
    df.insert(1, "subset", subset)
    df.insert(2, "split", split)
    return df


def save_frame(df: pd.DataFrame, output_path: Path, save_format: str) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if save_format == "parquet":
        path = output_path.with_suffix(".parquet")
        df.to_parquet(path, index=False)
        return path
    path = output_path.with_suffix(".csv")
    df.to_csv(path, index=False)
    return path


def split_summary(df: pd.DataFrame, subset: str, split: str, path: Path) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "subset": subset,
        "split": split,
        "rows": int(len(df)),
        "columns": list(df.columns),
        "path": str(path),
    }

    if subset == "pair-class" and "label_name" in df.columns:
        summary["label_counts"] = {
            str(label): int(count) for label, count in df["label_name"].value_counts().to_dict().items()
        }
    if subset == "pair-score" and "score" in df.columns:
        summary["score_counts"] = {
            str(score): int(count) for score, count in df["score"].value_counts().to_dict().items()
        }
    return summary


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    metadata: dict[str, Any] = {
        "dataset_name": DATASET_NAME,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "subsets": {},
        "max_rows_per_split": args.max_rows_per_split,
        "save_format": args.save_format,
    }

    for subset in args.subsets:
        print(f"Loading {DATASET_NAME}/{subset}...")
        dataset = load_subset(subset, args.cache_dir)
        metadata["subsets"][subset] = {}

        for split in SPLITS:
            if split not in dataset:
                print(f"Skipping missing split: {subset}/{split}")
                continue

            frame = dataset[split].to_pandas()
            frame = maybe_sample_frame(frame, args.max_rows_per_split, args.seed)
            frame = prepare_frame(frame, subset, split, args.skip_features)

            save_base = output_dir / subset / split
            saved_path = save_frame(frame, save_base, args.save_format)
            metadata["subsets"][subset][split] = split_summary(frame, subset, split, saved_path)
            print(f"Saved {subset}/{split}: {len(frame):,} rows -> {saved_path}")

    metadata_path = output_dir / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Saved metadata -> {metadata_path}")


if __name__ == "__main__":
    main()
