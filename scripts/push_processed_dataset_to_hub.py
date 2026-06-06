"""Push a processed AllNLI subset to Hugging Face Dataset Hub.

Example:
    python scripts/push_processed_dataset_to_hub.py \
        --subset pair-class \
        --repo-id your-team/allnli-pair-class-processed \
        --private
"""

from __future__ import annotations

import argparse
from pathlib import Path


SPLITS = ("train", "dev", "test")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-dir",
        default="data/processed/allnli",
        help="Root directory containing processed AllNLI subsets.",
    )
    parser.add_argument(
        "--subset",
        required=True,
        help="Processed subset directory name, for example pair-class or triplet.",
    )
    parser.add_argument(
        "--repo-id",
        required=True,
        help="Hugging Face dataset repo id, for example team-name/allnli-pair-class-processed.",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Create/update the dataset repo as private.",
    )
    parser.add_argument(
        "--rename-dev-to-validation",
        action="store_true",
        help="Publish local split 'dev' as 'validation'.",
    )
    parser.add_argument(
        "--commit-message",
        default="Upload processed AllNLI subset",
        help="Commit message on Hugging Face Hub.",
    )
    return parser.parse_args()


def find_split_file(subset_dir: Path, split: str) -> Path | None:
    for suffix in (".parquet", ".csv"):
        path = subset_dir / f"{split}{suffix}"
        if path.exists():
            return path
    return None


def load_split(path: Path):
    from datasets import Dataset

    if path.suffix == ".parquet":
        return Dataset.from_parquet(str(path))
    if path.suffix == ".csv":
        return Dataset.from_csv(str(path))
    raise ValueError(f"Unsupported file type: {path}")


def main() -> None:
    args = parse_args()
    from datasets import DatasetDict

    subset_dir = Path(args.input_dir) / args.subset
    if not subset_dir.exists():
        raise FileNotFoundError(f"Processed subset directory not found: {subset_dir}")

    datasets = {}
    for split in SPLITS:
        split_file = find_split_file(subset_dir, split)
        if split_file is None:
            continue
        hub_split = "validation" if split == "dev" and args.rename_dev_to_validation else split
        datasets[hub_split] = load_split(split_file)
        print(f"Loaded {split_file} as split '{hub_split}' with {len(datasets[hub_split]):,} rows")

    if not datasets:
        raise FileNotFoundError(f"No parquet/csv split files found under {subset_dir}")

    dataset_dict = DatasetDict(datasets)
    dataset_dict.push_to_hub(
        args.repo_id,
        private=args.private,
        commit_message=args.commit_message,
    )
    print(f"Pushed processed dataset to https://huggingface.co/datasets/{args.repo_id}")


if __name__ == "__main__":
    main()
