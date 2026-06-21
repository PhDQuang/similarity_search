import argparse
import os
import time
from typing import cast

from datasets import Dataset, DatasetDict, load_dataset
from transformers import AutoTokenizer

from similarity_search.sftbe.config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG

MAX_SEQ_LENGTH = MODEL_CONFIG["max_seq_length"]
MAP_BATCH_SIZE = int(os.environ.get("SFTBE_PREPARE_MAP_BATCH_SIZE", "5000"))
SENTENCE_BATCH_SIZE = int(os.environ.get("SFTBE_PREPARE_SENTENCE_BATCH_SIZE", "1000"))


def cpu_workers() -> int:
    try:
        sched_getaffinity = getattr(os, "sched_getaffinity")
        return max(1, len(sched_getaffinity(0)))
    except Exception:
        return max(1, os.cpu_count() or 1)


def prepare_wikipedia(tokenizer, cache_dir: str, num_proc: int):
    save_path = os.path.join(cache_dir, "wikipedia_tokenized")
    if os.path.exists(save_path):
        print(f"Skip existing cache: {save_path}")
        return

    start = time.time()
    wiki = cast(
        Dataset,
        load_dataset("wikimedia/wikipedia", "20231101.en", split="train"),
    )

    def extract_sentences(batch):
        sentences = []
        for text in batch["text"]:
            for sent in text.split(". "):
                sent = sent.strip()
                if 20 <= len(sent) <= 500:
                    sentences.append(sent)
        return {"sentence": sentences}

    sentences = wiki.map(
        extract_sentences,
        batched=True,
        batch_size=SENTENCE_BATCH_SIZE,
        remove_columns=wiki.column_names,
        num_proc=num_proc,
        desc="Split Wikipedia into sentences",
    )

    def tokenize(batch):
        encoded = tokenizer(
            batch["sentence"],
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids": encoded["input_ids"],
            "attention_mask": encoded["attention_mask"],
        }

    tokenized = sentences.map(
        tokenize,
        batched=True,
        batch_size=MAP_BATCH_SIZE,
        remove_columns=["sentence"],
        num_proc=num_proc,
        desc="Tokenize Wikipedia",
    )
    tokenized.save_to_disk(save_path)
    print(f"Saved {len(tokenized):,} Wikipedia sentences to {save_path} in {time.time() - start:.0f}s")


def prepare_stsb(tokenizer, cache_dir: str, num_proc: int):
    save_path = os.path.join(cache_dir, "stsb_tokenized")
    if os.path.exists(save_path):
        print(f"Skip existing cache: {save_path}")
        return

    stsb = load_dataset("mteb/stsbenchmark-sts")
    assert isinstance(stsb, DatasetDict)

    def tokenize(batch):
        enc_a = tokenizer(
            batch["sentence1"],
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
        )
        enc_b = tokenizer(
            batch["sentence2"],
            max_length=MAX_SEQ_LENGTH,
            padding="max_length",
            truncation=True,
        )
        return {
            "input_ids_a": enc_a["input_ids"],
            "attention_mask_a": enc_a["attention_mask"],
            "input_ids_b": enc_b["input_ids"],
            "attention_mask_b": enc_b["attention_mask"],
            "score": [score / 5.0 for score in batch["score"]],
        }

    tokenized = stsb.map(
        tokenize,
        batched=True,
        batch_size=MAP_BATCH_SIZE,
        remove_columns=["sentence1", "sentence2"],
        num_proc=num_proc,
        desc="Tokenize STS-B",
    )
    tokenized.save_to_disk(save_path)
    print(f"Saved STS-B cache to {save_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", default=TRAIN_CONFIG["data_cache_dir"])
    parser.add_argument("--num-proc", type=int, default=cpu_workers())
    args = parser.parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    os.makedirs(args.cache_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(DATA_CONFIG["tokenizer_name"], use_fast=True)
    prepare_wikipedia(tokenizer, args.cache_dir, args.num_proc)
    prepare_stsb(tokenizer, args.cache_dir, args.num_proc)


if __name__ == "__main__":
    main()
