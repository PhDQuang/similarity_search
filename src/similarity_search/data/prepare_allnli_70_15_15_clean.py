"""Prepare cleaned full AllNLI pair-class with a shared 70/15/15 split."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from similarity_search.data.text_utils import (
    char_count,
    lexical_overlap,
    normalize_text,
    token_count,
)
from similarity_search.models.evaluation import ID2LABEL, LABEL2ID, SCORE_BY_LABEL, save_json

SOURCE_SPLITS = ("train", "dev", "test")
TARGET_SPLITS = ("train", "val", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="sentence-transformers/all-nli")
    parser.add_argument("--dataset-config", default="pair-class")
    parser.add_argument("--output-dir", default="data/processed/allnli_70_15_15_clean/pair-class")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def label_to_name(value: Any) -> str:
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in LABEL2ID:
            return lowered
        if lowered.isdigit():
            return ID2LABEL[int(lowered)]
    return ID2LABEL[int(value)]


def load_full_pair_class(dataset_name: str, dataset_config: str, cache_dir: str | None) -> pd.DataFrame:
    from datasets import load_dataset

    dataset = load_dataset(dataset_name, dataset_config, cache_dir=cache_dir)
    frames: list[pd.DataFrame] = []
    for split in SOURCE_SPLITS:
        if split not in dataset:
            continue
        frame = dataset[split].to_pandas()
        frame.insert(0, "source_split", split)
        frame.insert(1, "source_row_id", range(len(frame)))
        frames.append(frame)
    if not frames:
        raise ValueError(f"No source splits found in {dataset_name}/{dataset_config}")
    return pd.concat(frames, ignore_index=True)


def preprocess_frame(frame: pd.DataFrame, dataset_name: str, dataset_config: str) -> pd.DataFrame:
    required = {"premise", "hypothesis", "label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing)}")

    result = frame.copy()
    result["premise_clean"] = result["premise"].map(normalize_text)
    result["hypothesis_clean"] = result["hypothesis"].map(normalize_text)
    result["label_name"] = result["label"].map(label_to_name)
    result["similarity_score"] = result["label_name"].map(SCORE_BY_LABEL).astype(float)
    result["premise_char_len"] = result["premise_clean"].map(char_count)
    result["hypothesis_char_len"] = result["hypothesis_clean"].map(char_count)
    result["premise_token_len"] = result["premise_clean"].map(token_count)
    result["hypothesis_token_len"] = result["hypothesis_clean"].map(token_count)
    result["lexical_overlap"] = [
        lexical_overlap(a, b) for a, b in zip(result["premise_clean"], result["hypothesis_clean"])
    ]
    result.insert(0, "dataset_name", dataset_name)
    result.insert(1, "dataset_config", dataset_config)
    return result


def split_frame(
    frame: pd.DataFrame,
    seed: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
) -> pd.DataFrame:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_rows = len(shuffled)
    train_end = int(n_rows * train_ratio)
    val_end = train_end + int(n_rows * val_ratio)
    pieces = []
    for split, piece in {
        "train": shuffled.iloc[:train_end].copy(),
        "val": shuffled.iloc[train_end:val_end].copy(),
        "test": shuffled.iloc[val_end:].copy(),
    }.items():
        piece.insert(2, "split", split)
        piece["_audit_row_id"] = range(len(piece))
        pieces.append(piece)
    return pd.concat(pieces, ignore_index=True)


def clean_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    working = frame.copy()
    before_rows = len(working)
    working = working[
        working["premise_clean"].astype(bool) & working["hypothesis_clean"].astype(bool)
    ].copy()
    after_non_empty = len(working)

    pair_key = working["premise_clean"].astype(str) + " ||| " + working["hypothesis_clean"].astype(str)
    working["_pair_key"] = pair_key
    working["_pair_label_key"] = pair_key + " ||| label=" + working["label_name"].astype(str)

    label_counts = working.groupby("_pair_key")["label_name"].nunique()
    conflicting_keys = set(label_counts[label_counts > 1].index)
    working = working[~working["_pair_key"].isin(conflicting_keys)].copy()
    after_conflict_filter = len(working)

    split_rank = {"train": 0, "val": 1, "test": 2}
    working["_split_rank"] = working["split"].map(split_rank)
    working = (
        working.sort_values(["_split_rank", "_audit_row_id"])
        .drop_duplicates(subset=["_pair_label_key"], keep="first")
        .copy()
    )
    after_duplicate_filter = len(working)

    clean_keys_by_split = {
        split: set(group["_pair_key"].tolist())
        for split, group in working.groupby("split", sort=False)
    }
    overlap_keys = 0
    splits = list(clean_keys_by_split)
    for index, left in enumerate(splits):
        for right in splits[index + 1:]:
            overlap_keys += len(clean_keys_by_split[left] & clean_keys_by_split[right])

    audit_columns = [column for column in working.columns if column.startswith("_")]
    clean = working.drop(columns=audit_columns).reset_index(drop=True)
    summary = {
        "rows_before": int(before_rows),
        "rows_after_non_empty_filter": int(after_non_empty),
        "conflicting_pair_groups": int(len(conflicting_keys)),
        "rows_after_conflict_filter": int(after_conflict_filter),
        "rows_after_duplicate_filter": int(after_duplicate_filter),
        "removed_rows": int(before_rows - after_duplicate_filter),
        "clean_validation": {
            "duplicate_pair_groups": int(
                (clean["premise_clean"].astype(str) + " ||| " + clean["hypothesis_clean"].astype(str))
                .value_counts()
                .gt(1)
                .sum()
            ),
            "conflicting_pair_groups": 0,
            "pair_overlap_between_splits": int(overlap_keys),
        },
    }
    return clean, summary


def save_splits(frame: pd.DataFrame, output_dir: Path) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, str] = {}
    for split in TARGET_SPLITS:
        split_frame = frame[frame["split"] == split].reset_index(drop=True)
        path = output_dir / f"{split}.parquet"
        split_frame.to_parquet(path, index=False)
        paths[split] = str(path)
    return paths


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    if not args.force and all((output_dir / f"{split}.parquet").exists() for split in TARGET_SPLITS):
        print(f"Clean dataset already exists: {output_dir}")
        return

    raw = load_full_pair_class(args.dataset_name, args.dataset_config, args.cache_dir)
    processed = preprocess_frame(raw, args.dataset_name, args.dataset_config)
    split = split_frame(processed, args.seed, args.train_ratio, args.val_ratio, args.test_ratio)
    clean, clean_summary = clean_frame(split)
    saved_paths = save_splits(clean, output_dir)

    row_counts = {
        split_name: int((clean["split"] == split_name).sum())
        for split_name in TARGET_SPLITS
    }
    metadata = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "split_ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "saved_paths": saved_paths,
        "row_counts": row_counts,
        "label_mapping": LABEL2ID,
        "score_mapping": SCORE_BY_LABEL,
        **clean_summary,
    }
    save_json(metadata, output_dir / "metadata.json")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()

