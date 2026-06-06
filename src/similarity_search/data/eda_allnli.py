"""Create EDA tables and figures for processed AllNLI data."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd

from similarity_search.data.text_utils import top_words

SPLITS = ("train", "dev", "test")
SUBSETS = ("pair-class", "pair-score", "pair", "triplet")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="data/processed/allnli",
        help="Directory created by prepare_allnli.py.",
    )
    parser.add_argument(
        "--subset",
        default="pair-class",
        choices=SUBSETS,
        help="Prepared AllNLI subset to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Root directory for EDA outputs.",
    )
    parser.add_argument(
        "--top-n-words",
        type=int,
        default=30,
        help="Number of frequent words to save and plot.",
    )
    parser.add_argument(
        "--max-plot-rows",
        type=int,
        default=100_000,
        help="Max sampled rows for distribution plots.",
    )
    return parser.parse_args()


def load_split(input_dir: Path, subset: str, split: str) -> pd.DataFrame | None:
    base = input_dir / subset / split
    parquet_path = base.with_suffix(".parquet")
    csv_path = base.with_suffix(".csv")
    if parquet_path.exists():
        return pd.read_parquet(parquet_path)
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


def ensure_dirs(output_dir: Path, subset: str) -> tuple[Path, Path, Path]:
    table_dir = output_dir / "tables" / "allnli" / subset
    figure_dir = output_dir / "figures" / "allnli" / subset
    report_dir = output_dir / "reports" / "allnli" / subset
    for path in (table_dir, figure_dir, report_dir):
        path.mkdir(parents=True, exist_ok=True)
    return table_dir, figure_dir, report_dir


def save_row_counts(frames: dict[str, pd.DataFrame], table_dir: Path) -> pd.DataFrame:
    rows = [{"split": split, "rows": len(frame)} for split, frame in frames.items()]
    table = pd.DataFrame(rows)
    table.to_csv(table_dir / "row_counts.csv", index=False)
    return table


def save_descriptive_stats(frames: dict[str, pd.DataFrame], table_dir: Path) -> pd.DataFrame:
    numeric_rows = []
    for split, frame in frames.items():
        numeric_columns = frame.select_dtypes(include="number").columns
        if len(numeric_columns) == 0:
            continue
        stats = frame[numeric_columns].describe().T.reset_index().rename(columns={"index": "feature"})
        stats.insert(0, "split", split)
        numeric_rows.append(stats)
    if not numeric_rows:
        return pd.DataFrame()
    table = pd.concat(numeric_rows, ignore_index=True)
    table.to_csv(table_dir / "numeric_descriptive_stats.csv", index=False)
    return table


def plot_simple_bar(labels: Iterable[str], values: Iterable[float], title: str, path: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.bar(list(labels), list(values), color="#2f6f9f")
    plt.title(title)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_histogram(frame: pd.DataFrame, columns: list[str], title: str, path: Path) -> None:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return
    plt.figure(figsize=(10, 6))
    for column in available:
        plt.hist(frame[column].dropna(), bins=50, alpha=0.45, label=column)
    plt.title(title)
    plt.xlabel("Length")
    plt.ylabel("Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def plot_box_by_category(
    frame: pd.DataFrame,
    value_column: str,
    category_column: str,
    title: str,
    path: Path,
) -> None:
    if value_column not in frame.columns or category_column not in frame.columns:
        return
    groups = []
    labels = []
    for label, group in frame.groupby(category_column):
        values = group[value_column].dropna()
        if not values.empty:
            groups.append(values)
            labels.append(str(label))
    if not groups:
        return

    plt.figure(figsize=(9, 5))
    plt.boxplot(groups, labels=labels, showfliers=False)
    plt.title(title)
    plt.xlabel(category_column)
    plt.ylabel(value_column)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()


def analyze_pair_class(
    frames: dict[str, pd.DataFrame],
    table_dir: Path,
    figure_dir: Path,
    report_dir: Path,
    top_n_words: int,
    max_plot_rows: int,
) -> dict[str, object]:
    label_tables = []
    for split, frame in frames.items():
        counts = frame["label_name"].value_counts().rename_axis("label_name").reset_index(name="count")
        counts.insert(0, "split", split)
        counts["percent"] = counts["count"] / counts["count"].sum()
        label_tables.append(counts)

    label_distribution = pd.concat(label_tables, ignore_index=True)
    label_distribution.to_csv(table_dir / "label_distribution.csv", index=False)

    pivot = label_distribution.pivot(index="split", columns="label_name", values="count").fillna(0)
    pivot.plot(kind="bar", figsize=(10, 5), color=["#2f6f9f", "#d08c2f", "#7a4f9f"])
    plt.title("AllNLI pair-class label distribution")
    plt.xlabel("Split")
    plt.ylabel("Rows")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figure_dir / "label_distribution.png", dpi=180)
    plt.close()

    train = frames.get("train")
    if train is not None:
        plot_frame = train.sample(min(len(train), max_plot_rows), random_state=42)
        plot_histogram(
            plot_frame,
            ["premise_token_len", "hypothesis_token_len"],
            "Sentence token length distribution",
            figure_dir / "token_length_histogram.png",
        )
        plot_box_by_category(
            plot_frame,
            "lexical_overlap",
            "label_name",
            "Lexical overlap by NLI label",
            figure_dir / "lexical_overlap_by_label.png",
        )

        examples = (
            train.groupby("label_name", group_keys=False)
            .head(5)[["premise", "hypothesis", "label_name", "lexical_overlap"]]
            .reset_index(drop=True)
        )
        examples.to_csv(table_dir / "examples_by_label.csv", index=False)

        words = top_words(
            list(train["premise_clean"].head(max_plot_rows))
            + list(train["hypothesis_clean"].head(max_plot_rows)),
            top_n=top_n_words,
        )
        top_word_table = pd.DataFrame(words, columns=["word", "count"])
        top_word_table.to_csv(table_dir / "top_words.csv", index=False)
        plot_simple_bar(
            top_word_table["word"],
            top_word_table["count"],
            f"Top {top_n_words} words in train split",
            figure_dir / "top_words.png",
        )

    report = {
        "label_distribution_table": str(table_dir / "label_distribution.csv"),
        "figures": [
            str(figure_dir / "label_distribution.png"),
            str(figure_dir / "token_length_histogram.png"),
            str(figure_dir / "lexical_overlap_by_label.png"),
            str(figure_dir / "top_words.png"),
        ],
    }
    (report_dir / "eda_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def analyze_pair_score(
    frames: dict[str, pd.DataFrame],
    table_dir: Path,
    figure_dir: Path,
    report_dir: Path,
    top_n_words: int,
    max_plot_rows: int,
) -> dict[str, object]:
    score_tables = []
    for split, frame in frames.items():
        counts = frame["score"].value_counts().sort_index().rename_axis("score").reset_index(name="count")
        counts.insert(0, "split", split)
        counts["percent"] = counts["count"] / counts["count"].sum()
        score_tables.append(counts)

    score_distribution = pd.concat(score_tables, ignore_index=True)
    score_distribution.to_csv(table_dir / "score_distribution.csv", index=False)

    pivot = score_distribution.pivot(index="split", columns="score", values="count").fillna(0)
    pivot.plot(kind="bar", figsize=(10, 5), color=["#7a4f9f", "#d08c2f", "#2f6f9f"])
    plt.title("AllNLI pair-score distribution")
    plt.xlabel("Split")
    plt.ylabel("Rows")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig(figure_dir / "score_distribution.png", dpi=180)
    plt.close()

    train = frames.get("train")
    if train is not None:
        plot_frame = train.sample(min(len(train), max_plot_rows), random_state=42)
        plot_histogram(
            plot_frame,
            ["sentence1_token_len", "sentence2_token_len"],
            "Sentence token length distribution",
            figure_dir / "token_length_histogram.png",
        )
        plot_box_by_category(
            plot_frame,
            "lexical_overlap",
            "score",
            "Lexical overlap by similarity score",
            figure_dir / "lexical_overlap_by_score.png",
        )
        words = top_words(
            list(train["sentence1_clean"].head(max_plot_rows))
            + list(train["sentence2_clean"].head(max_plot_rows)),
            top_n=top_n_words,
        )
        top_word_table = pd.DataFrame(words, columns=["word", "count"])
        top_word_table.to_csv(table_dir / "top_words.csv", index=False)
        plot_simple_bar(
            top_word_table["word"],
            top_word_table["count"],
            f"Top {top_n_words} words in train split",
            figure_dir / "top_words.png",
        )

    report = {
        "score_distribution_table": str(table_dir / "score_distribution.csv"),
        "figures": [
            str(figure_dir / "score_distribution.png"),
            str(figure_dir / "token_length_histogram.png"),
            str(figure_dir / "lexical_overlap_by_score.png"),
            str(figure_dir / "top_words.png"),
        ],
    }
    (report_dir / "eda_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def analyze_pair_or_triplet(
    frames: dict[str, pd.DataFrame],
    subset: str,
    table_dir: Path,
    figure_dir: Path,
    report_dir: Path,
    top_n_words: int,
    max_plot_rows: int,
) -> dict[str, object]:
    train = frames.get("train")
    if train is not None:
        plot_frame = train.sample(min(len(train), max_plot_rows), random_state=42)
        if subset == "pair":
            length_columns = ["anchor_token_len", "positive_token_len"]
            overlap_columns = ["anchor_positive_overlap"]
            text_columns = ["anchor_clean", "positive_clean"]
        else:
            length_columns = ["anchor_token_len", "positive_token_len", "negative_token_len"]
            overlap_columns = ["anchor_positive_overlap", "anchor_negative_overlap"]
            text_columns = ["anchor_clean", "positive_clean", "negative_clean"]

        plot_histogram(
            plot_frame,
            length_columns,
            f"AllNLI {subset} token length distribution",
            figure_dir / "token_length_histogram.png",
        )
        for column in overlap_columns:
            if column in plot_frame.columns:
                plt.figure(figsize=(8, 5))
                plt.hist(plot_frame[column].dropna(), bins=50, color="#2f6f9f")
                plt.title(column)
                plt.xlabel("Jaccard overlap")
                plt.ylabel("Count")
                plt.tight_layout()
                plt.savefig(figure_dir / f"{column}.png", dpi=180)
                plt.close()

        text_values: list[str] = []
        for column in text_columns:
            text_values.extend(list(train[column].head(max_plot_rows)))
        words = top_words(text_values, top_n=top_n_words)
        top_word_table = pd.DataFrame(words, columns=["word", "count"])
        top_word_table.to_csv(table_dir / "top_words.csv", index=False)
        plot_simple_bar(
            top_word_table["word"],
            top_word_table["count"],
            f"Top {top_n_words} words in train split",
            figure_dir / "top_words.png",
        )

    report = {
        "subset": subset,
        "figures_dir": str(figure_dir),
        "tables_dir": str(table_dir),
    }
    (report_dir / "eda_summary.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    args = parse_args()
    input_dir = Path(args.input_dir)
    table_dir, figure_dir, report_dir = ensure_dirs(Path(args.output_dir), args.subset)

    frames = {
        split: frame
        for split in SPLITS
        if (frame := load_split(input_dir, args.subset, split)) is not None
    }
    if not frames:
        raise FileNotFoundError(
            f"No processed files found for subset '{args.subset}' under {input_dir}. "
            "Run prepare_allnli.py first."
        )

    row_counts = save_row_counts(frames, table_dir)
    save_descriptive_stats(frames, table_dir)
    plot_simple_bar(row_counts["split"], row_counts["rows"], "Rows by split", figure_dir / "rows_by_split.png")

    if args.subset == "pair-class":
        report = analyze_pair_class(
            frames,
            table_dir,
            figure_dir,
            report_dir,
            args.top_n_words,
            args.max_plot_rows,
        )
    elif args.subset == "pair-score":
        report = analyze_pair_score(
            frames,
            table_dir,
            figure_dir,
            report_dir,
            args.top_n_words,
            args.max_plot_rows,
        )
    else:
        report = analyze_pair_or_triplet(
            frames,
            args.subset,
            table_dir,
            figure_dir,
            report_dir,
            args.top_n_words,
            args.max_plot_rows,
        )

    print(f"Loaded splits: {', '.join(frames.keys())}")
    print(f"Tables: {table_dir}")
    print(f"Figures: {figure_dir}")
    print(f"Report: {report_dir / 'eda_summary.json'}")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

