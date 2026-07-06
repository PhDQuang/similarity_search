"""Prepare full AllNLI pair-class with one shared 70/15/15 split."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd

from similarity_search.data.text_utils import (
    char_count,
    lexical_overlap,
    normalize_text,
    token_count,
    top_words,
)
from similarity_search.models.evaluation import ID2LABEL, LABEL2ID, SCORE_BY_LABEL, save_json

SOURCE_SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", default="sentence-transformers/all-nli")
    parser.add_argument("--dataset-config", default="pair-class")
    parser.add_argument("--output-dir", default="data/processed/allnli_70_15_15/pair-class")
    parser.add_argument("--outputs-dir", default="outputs")
    parser.add_argument("--cache-dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-ratio", type=float, default=0.70)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--save-format", choices=("parquet", "csv"), default="parquet")
    parser.add_argument("--top-n-words", type=int, default=30)
    parser.add_argument("--top-word-max-rows", type=int, default=200_000)
    parser.add_argument("--max-plot-rows", type=int, default=200_000)
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
) -> dict[str, pd.DataFrame]:
    ratio_sum = train_ratio + val_ratio + test_ratio
    if abs(ratio_sum - 1.0) > 1e-8:
        raise ValueError(f"Split ratios must sum to 1.0, got {ratio_sum}")

    shuffled = frame.sample(frac=1.0, random_state=seed).reset_index(drop=True)
    n_rows = len(shuffled)
    train_end = int(n_rows * train_ratio)
    val_end = train_end + int(n_rows * val_ratio)
    splits = {
        "train": shuffled.iloc[:train_end].copy(),
        "val": shuffled.iloc[train_end:val_end].copy(),
        "test": shuffled.iloc[val_end:].copy(),
    }
    for split, split_frame_ in splits.items():
        split_frame_.insert(2, "split", split)
        split_frame_.reset_index(drop=True, inplace=True)
    return splits


def save_split(frame: pd.DataFrame, output_dir: Path, split: str, save_format: str) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    if save_format == "parquet":
        path = output_dir / f"{split}.parquet"
        frame.to_parquet(path, index=False)
        return path
    path = output_dir / f"{split}.csv"
    frame.to_csv(path, index=False)
    return path


def plot_bar(labels: list[str], values: list[float], title: str, path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(labels, values, color="#2f6f9f")
    plt.title(title)
    plt.xticks(rotation=25, ha="right")
    plt.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180)
    plt.close()


def save_eda_outputs(
    splits: dict[str, pd.DataFrame],
    outputs_dir: Path,
    top_n_words: int,
    top_word_max_rows: int,
    max_plot_rows: int,
) -> dict[str, Any]:
    table_dir = outputs_dir / "tables" / "allnli_70_15_15" / "pair-class"
    figure_dir = outputs_dir / "figures" / "allnli_70_15_15" / "pair-class"
    report_dir = outputs_dir / "reports" / "allnli_70_15_15" / "pair-class"
    table_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    row_counts = pd.DataFrame([{"split": split, "rows": len(frame)} for split, frame in splits.items()])
    row_counts.to_csv(table_dir / "row_counts.csv", index=False)
    plot_bar(row_counts["split"].tolist(), row_counts["rows"].tolist(), "Rows by fixed split", figure_dir / "rows_by_split.png")

    label_tables = []
    for split, frame in splits.items():
        counts = frame["label_name"].value_counts().rename_axis("label_name").reset_index(name="count")
        counts.insert(0, "split", split)
        counts["percent"] = counts["count"] / counts["count"].sum()
        label_tables.append(counts)
    label_distribution = pd.concat(label_tables, ignore_index=True)
    label_distribution.to_csv(table_dir / "label_distribution.csv", index=False)
    pivot = label_distribution.pivot(index="split", columns="label_name", values="count").fillna(0)
    pivot.plot(kind="bar", figsize=(10, 5), color=["#2f6f9f", "#d08c2f", "#7a4f9f"])
    plt.title("AllNLI 70/15/15 label distribution")
    plt.xlabel("Split")
    plt.ylabel("Rows")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figure_dir / "label_distribution.png", dpi=180)
    plt.close()

    numeric_stats = []
    for split, frame in splits.items():
        stats = frame.select_dtypes(include="number").describe().T.reset_index().rename(columns={"index": "feature"})
        stats.insert(0, "split", split)
        numeric_stats.append(stats)
    pd.concat(numeric_stats, ignore_index=True).to_csv(table_dir / "numeric_descriptive_stats.csv", index=False)

    train = splits["train"]
    plot_frame = train.sample(min(len(train), max_plot_rows), random_state=42)
    plt.figure(figsize=(10, 6))
    plt.hist(plot_frame["premise_token_len"], bins=50, alpha=0.45, label="premise")
    plt.hist(plot_frame["hypothesis_token_len"], bins=50, alpha=0.45, label="hypothesis")
    plt.title("Token length distribution")
    plt.xlabel("Token count")
    plt.ylabel("Rows")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figure_dir / "token_length_histogram.png", dpi=180)
    plt.close()

    groups = [group["lexical_overlap"].dropna() for _, group in plot_frame.groupby("label_name")]
    labels = [str(label) for label, _ in plot_frame.groupby("label_name")]
    plt.figure(figsize=(9, 5))
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.title("Lexical overlap by label")
    plt.xlabel("Label")
    plt.ylabel("Jaccard overlap")
    plt.tight_layout()
    plt.savefig(figure_dir / "lexical_overlap_by_label.png", dpi=180)
    plt.close()

    examples = (
        train.groupby("label_name", group_keys=False)
        .head(5)[["premise", "hypothesis", "label_name", "lexical_overlap"]]
        .reset_index(drop=True)
    )
    examples.to_csv(table_dir / "examples_by_label.csv", index=False)

    word_frame = train.head(top_word_max_rows)
    words = top_words(
        list(word_frame["premise_clean"]) + list(word_frame["hypothesis_clean"]),
        top_n=top_n_words,
    )
    top_word_table = pd.DataFrame(words, columns=["word", "count"])
    top_word_table.to_csv(table_dir / "top_words.csv", index=False)
    plot_bar(
        top_word_table["word"].tolist(),
        top_word_table["count"].tolist(),
        f"Top {top_n_words} words",
        figure_dir / "top_words.png",
    )

    report = {
        "row_counts_table": str(table_dir / "row_counts.csv"),
        "label_distribution_table": str(table_dir / "label_distribution.csv"),
        "numeric_descriptive_stats_table": str(table_dir / "numeric_descriptive_stats.csv"),
        "examples_table": str(table_dir / "examples_by_label.csv"),
        "figures": [
            str(figure_dir / "rows_by_split.png"),
            str(figure_dir / "label_distribution.png"),
            str(figure_dir / "token_length_histogram.png"),
            str(figure_dir / "lexical_overlap_by_label.png"),
            str(figure_dir / "top_words.png"),
        ],
    }
    save_json(report, report_dir / "eda_summary.json")
    return report


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    outputs_dir = Path(args.outputs_dir)

    raw = load_full_pair_class(args.dataset_name, args.dataset_config, args.cache_dir)
    before_rows = len(raw)
    processed = preprocess_frame(raw, args.dataset_name, args.dataset_config)
    processed = processed[
        processed["premise_clean"].astype(bool) & processed["hypothesis_clean"].astype(bool)
    ].reset_index(drop=True)
    splits = split_frame(
        processed,
        seed=args.seed,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
    )

    saved_paths = {
        split: str(save_split(frame, output_dir, split, args.save_format))
        for split, frame in splits.items()
    }
    eda_report = save_eda_outputs(
        splits,
        outputs_dir=outputs_dir,
        top_n_words=args.top_n_words,
        top_word_max_rows=args.top_word_max_rows,
        max_plot_rows=args.max_plot_rows,
    )

    metadata = {
        "dataset_name": args.dataset_name,
        "dataset_config": args.dataset_config,
        "source_splits": SOURCE_SPLITS,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "seed": args.seed,
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "rows_before_clean_filter": int(before_rows),
        "rows_after_clean_filter": int(len(processed)),
        "saved_paths": saved_paths,
        "label_mapping": LABEL2ID,
        "score_mapping": SCORE_BY_LABEL,
        "eda_report": eda_report,
    }
    save_json(metadata, output_dir / "metadata.json")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()



