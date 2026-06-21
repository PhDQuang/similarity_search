import math
import os
import torch
from typing import Any, cast
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from datasets import Dataset as HFDataset, DatasetDict, load_dataset, load_from_disk


def get_tokenizer(tokenizer_name: str = "bert-base-uncased"):
    return AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)


def _long(value):
    return torch.as_tensor(value, dtype=torch.long).clone().detach()


def _float(value):
    return torch.as_tensor(value, dtype=torch.float32).clone().detach()


def _coprime_multiplier(size: int, seed: int) -> int:
    if size <= 1:
        return 1
    value = max(1, (0x9E3779B97F4A7C15 ^ abs(seed)) % size)
    while math.gcd(value, size) != 1:
        value = (value + 2) % size or 1
    return value


class RandomIndexSplitDataset(Dataset):
    """Split ngẫu nhiên quyết định, không lưu list index lớn trong RAM."""

    def __init__(self, dataset: Any, split: str,
                 validation_ratio: float = 0.02, seed: int = 42):
        if split not in {"train", "validation"}:
            raise ValueError(f"Unsupported split: {split}")
        if not 0.0 < validation_ratio < 1.0:
            raise ValueError("validation_ratio phải nằm trong khoảng (0, 1)")

        self.dataset = dataset
        self.total = len(cast(Any, dataset))
        self.val_size = min(max(1, round(self.total * validation_ratio)), self.total - 1)
        self.start = 0 if split == "validation" else self.val_size
        self.length = self.val_size if split == "validation" else self.total - self.val_size
        self.multiplier = _coprime_multiplier(self.total, seed)
        self.offset = seed % max(1, self.total)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise IndexError(idx)
        mapped = (self.offset + (self.start + idx) * self.multiplier) % self.total
        return self.dataset[mapped]


class WikipediaDistillationDataset(Dataset):
    """Wikipedia sentence cache dùng cho teacher-student distillation."""

    def __init__(self, cache_dir: str):
        cache_path = os.path.join(cache_dir, "wikipedia_tokenized")
        if not os.path.exists(cache_path):
            raise RuntimeError(
                f"Không tìm thấy {cache_path}. Hãy chạy src/prepare_data.py trước."
            )
        self.data = cast(HFDataset, load_from_disk(cache_path))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        return {
            "input_ids": _long(item["input_ids"]),
            "attention_mask": _long(item["attention_mask"]),
        }


class STSBDataset(Dataset):
    """STS-B dùng để đánh giá Spearman correlation."""

    def __init__(self, cache_dir: str, tokenizer, split: str = "validation",
                 max_length: int = 128):
        cache_path = os.path.join(cache_dir, "stsb_tokenized")
        self.use_cache = os.path.exists(cache_path)
        if self.use_cache:
            stsb_cache = cast(DatasetDict, load_from_disk(cache_path))
            self.data = cast(HFDataset, stsb_cache[split])
        else:
            self.tokenizer = tokenizer
            self.data = cast(HFDataset, load_dataset("mteb/stsbenchmark-sts", split=split))
        self.max_length = max_length

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        if self.use_cache:
            return {
                "input_ids_a": _long(item["input_ids_a"]),
                "attention_mask_a": _long(item["attention_mask_a"]),
                "input_ids_b": _long(item["input_ids_b"]),
                "attention_mask_b": _long(item["attention_mask_b"]),
                "score": _float(item["score"]),
            }

        enc_a = self.tokenizer(
            item["sentence1"], max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        enc_b = self.tokenizer(
            item["sentence2"], max_length=self.max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        )
        return {
            "input_ids_a": enc_a["input_ids"].squeeze(0),
            "attention_mask_a": enc_a["attention_mask"].squeeze(0),
            "input_ids_b": enc_b["input_ids"].squeeze(0),
            "attention_mask_b": enc_b["attention_mask"].squeeze(0),
            "score": torch.tensor(float(item["score"]) / 5.0, dtype=torch.float32),
        }


def create_dataloader(dataset: Dataset, batch_size: int, shuffle: bool = True,
                      num_workers: int = 0, drop_last: bool = True) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": torch.cuda.is_available(),
        "drop_last": drop_last,
    }
    if num_workers > 0:
        kwargs["prefetch_factor"] = 2
    return DataLoader(dataset, **kwargs)
