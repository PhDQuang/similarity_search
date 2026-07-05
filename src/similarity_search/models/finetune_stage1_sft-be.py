import argparse
import csv
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, cast

import torch
import torch.nn.functional as F
from datasets import load_dataset, load_from_disk
from torch.amp import GradScaler, autocast  # pyright: ignore[reportPrivateImportUsage]
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import DATA_CONFIG, MODEL_CONFIG, get_device
from dataset import get_tokenizer
from model.encoder import create_sftbe_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Fine-tune Stage 0 on text triplets."
    )
    parser.add_argument("--stage-name", default="Stage 1")
    parser.add_argument(
        "--stage0", default=str(PROJECT_ROOT / "checkpoints" / "stage0_final.pt")
    )
    parser.add_argument(
        "--output", default=str(PROJECT_ROOT / "checkpoints" / "stage1_final.pt")
    )
    parser.add_argument(
        "--metrics",
        default=str(PROJECT_ROOT / "checkpoints" / "stage1_allnli_metrics.json"),
    )
    parser.add_argument(
        "--history",
        default=str(PROJECT_ROOT / "checkpoints" / "stage1_training_history.csv"),
    )
    parser.add_argument("--dataset", default="sentence-transformers/all-nli")
    parser.add_argument("--dataset-subset", default="triplet")
    parser.add_argument("--test-ratio", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--max-train-samples", type=int)
    parser.add_argument("--max-test-samples", type=int)
    parser.add_argument("--log-every", type=int, default=100)
    return parser.parse_args(argv)


def seed_everything(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_state_dict(path: str, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        return state["model_state_dict"]
    return state


class TripletCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def _encode(self, texts):
        return self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

    def __call__(self, examples):
        return {
            "anchor": self._encode([item["anchor"] for item in examples]),
            "positive": self._encode([item["positive"] for item in examples]),
            "negative": self._encode([item["negative"] for item in examples]),
        }


def encode(model, encoded, device):
    return model(
        encoded["input_ids"].to(device),
        encoded["attention_mask"].to(device),
    )


def triplet_cosine_loss(anchor, positive, negative, margin: float):
    positive_similarity = F.cosine_similarity(anchor, positive, dim=-1)
    negative_similarity = F.cosine_similarity(anchor, negative, dim=-1)
    loss = F.relu(margin + negative_similarity - positive_similarity).mean()
    return loss, positive_similarity, negative_similarity


@torch.no_grad()
def evaluate(model, loader, device, margin: float, use_amp: bool, description: str):
    model.eval()
    count = 0
    loss_sum = positive_sum = negative_sum = correct_sum = 0.0
    progress = tqdm(loader, desc=description, unit="batch", dynamic_ncols=True)
    for batch in progress:
        with autocast(device_type=device.type, enabled=use_amp):
            anchor = encode(model, batch["anchor"], device)
            positive = encode(model, batch["positive"], device)
            negative = encode(model, batch["negative"], device)
            loss, positive_similarity, negative_similarity = triplet_cosine_loss(
                anchor, positive, negative, margin
            )
        batch_size = anchor.size(0)
        count += batch_size
        loss_sum += loss.item() * batch_size
        positive_sum += positive_similarity.float().sum().item()
        negative_sum += negative_similarity.float().sum().item()
        correct_sum += (positive_similarity > negative_similarity).sum().item()
        progress.set_postfix(
            loss=f"{loss_sum / max(1, count):.4f}",
            accuracy=f"{correct_sum / max(1, count):.3f}",
        )

    positive_cosine = positive_sum / max(1, count)
    negative_cosine = negative_sum / max(1, count)
    return {
        "samples": count,
        "loss": loss_sum / max(1, count),
        "ranking_accuracy": correct_sum / max(1, count),
        "positive_cosine": positive_cosine,
        "negative_cosine": negative_cosine,
        "cosine_gap": positive_cosine - negative_cosine,
    }


def learning_rate_at(step: int, warmup_steps: int, total_steps: int) -> float:
    if step < warmup_steps:
        return (step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))


def main(argv=None):
    args = parse_args(argv)
    if not 0.0 < args.test_ratio < 1.0:
        raise ValueError("--test-ratio must be between 0 and 1")
    if not Path(args.stage0).exists():
        raise FileNotFoundError(f"Stage 0 checkpoint not found: {args.stage0}")

    seed_everything(args.seed)
    device = get_device()
    use_amp = device.type == "cuda"
    tokenizer = get_tokenizer(DATA_CONFIG["tokenizer_name"])
    collator = TripletCollator(tokenizer, MODEL_CONFIG["max_seq_length"])

    logger.info(
        "Loading %s triplets and creating deterministic %.1f%% test split",
        args.dataset,
        args.test_ratio * 100,
    )
    if Path(args.dataset).is_dir():
        dataset = load_from_disk(args.dataset)
    else:
        dataset = load_dataset(args.dataset, args.dataset_subset, split="train")
    split = dataset.train_test_split(test_size=args.test_ratio, seed=args.seed)
    train_data, test_data = split["train"], split["test"]
    if args.max_train_samples is not None:
        train_data = train_data.select(range(min(args.max_train_samples, len(train_data))))
    if args.max_test_samples is not None:
        test_data = test_data.select(range(min(args.max_test_samples, len(test_data))))

    train_loader = DataLoader(
        cast(Any, train_data),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        cast(Any, test_data),
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collator,
        pin_memory=device.type == "cuda",
    )

    model = create_sftbe_model(MODEL_CONFIG).to(device)
    model.load_state_dict(load_state_dict(args.stage0, device))
    logger.info(
        "Device: %s | train: %d samples (%d batches) | test: %d samples (%d batches)",
        device,
        len(train_data),
        len(train_loader),
        len(test_data),
        len(test_loader),
    )
    before = evaluate(
        model, test_loader, device, args.margin, use_amp, "Evaluate before"
    )
    logger.info("Before %s: %s", args.stage_name, before)

    optimizer = AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    accumulation = max(1, args.gradient_accumulation)
    updates_per_epoch = math.ceil(len(train_loader) / accumulation)
    total_updates = max(1, updates_per_epoch * args.epochs)
    warmup_steps = int(total_updates * args.warmup_ratio)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: learning_rate_at(step, warmup_steps, total_updates),
    )
    scaler = GradScaler(enabled=use_amp)
    global_step = 0
    training_start = time.time()
    history_path = Path(args.history)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    history_file = history_path.open("w", newline="", encoding="utf-8")
    history_writer = csv.DictWriter(
        history_file,
        fieldnames=[
            "epoch",
            "interval_start_update",
            "update",
            "interval_loss",
            "learning_rate",
            "elapsed_seconds",
        ],
    )
    history_writer.writeheader()

    try:
        for epoch in range(args.epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            interval_loss_sum = 0.0
            interval_batches = 0
            interval_start_update = global_step + 1
            displayed_interval_loss = 0.0
            progress = tqdm(
                train_loader,
                desc=f"Train {epoch + 1}/{args.epochs}",
                unit="batch",
                dynamic_ncols=True,
            )
            for batch_index, batch in enumerate(progress):
                with autocast(device_type=device.type, enabled=use_amp):
                    anchor = encode(model, batch["anchor"], device)
                    positive = encode(model, batch["positive"], device)
                    negative = encode(model, batch["negative"], device)
                    loss, _, _ = triplet_cosine_loss(
                        anchor, positive, negative, args.margin
                    )
                    divisor = min(
                        accumulation,
                        len(train_loader)
                        - (batch_index // accumulation) * accumulation,
                    )
                    scaled_loss = loss / divisor
                scaler.scale(scaled_loss).backward()
                interval_loss_sum += loss.item()
                interval_batches += 1

                if (
                    (batch_index + 1) % accumulation == 0
                    or batch_index + 1 == len(train_loader)
                ):
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    interval_finished = (
                        global_step % args.log_every == 0
                        or global_step == total_updates
                        or batch_index + 1 == len(train_loader)
                    )
                    if interval_finished:
                        displayed_interval_loss = interval_loss_sum / max(
                            1, interval_batches
                        )
                        record = {
                            "epoch": epoch + 1,
                            "interval_start_update": interval_start_update,
                            "update": global_step,
                            "interval_loss": displayed_interval_loss,
                            "learning_rate": scheduler.get_last_lr()[0],
                            "elapsed_seconds": time.time() - training_start,
                        }
                        history_writer.writerow(record)
                        history_file.flush()
                        progress.write(
                            f"Epoch {epoch + 1}/{args.epochs} | "
                            f"updates {interval_start_update}-{global_step}/"
                            f"{total_updates} | "
                            f"interval loss {record['interval_loss']:.4f} | "
                            f"lr {record['learning_rate']:.2e}",
                            file=sys.stderr,
                        )
                        interval_loss_sum = 0.0
                        interval_batches = 0
                        interval_start_update = global_step + 1
                    else:
                        displayed_interval_loss = interval_loss_sum / max(
                            1, interval_batches
                        )
                progress.set_postfix(
                    interval_loss=f"{displayed_interval_loss:.4f}",
                    lr=f"{scheduler.get_last_lr()[0]:.2e}",
                    updates=f"{global_step}/{total_updates}",
                )
    finally:
        history_file.close()

    logger.info("Saved training history to %s", args.history)

    after = evaluate(
        model, test_loader, device, args.margin, use_amp, "Evaluate after"
    )
    logger.info("After %s: %s", args.stage_name, after)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), args.output)
    metrics = {
        "stage0_checkpoint": args.stage0,
        "stage_name": args.stage_name,
        "output_checkpoint": args.output,
        "dataset": args.dataset,
        "dataset_subset": args.dataset_subset,
        "seed": args.seed,
        "test_ratio": args.test_ratio,
        "train_samples": len(train_data),
        "test_samples": len(test_data),
        "epochs": args.epochs,
        "margin": args.margin,
        "training_loss": f"interval mean every {args.log_every} optimizer updates",
        "training_history": args.history,
        "before": before,
        "after": after,
    }
    Path(args.metrics).parent.mkdir(parents=True, exist_ok=True)
    Path(args.metrics).write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    logger.info("Saved %s checkpoint to %s", args.stage_name, args.output)
    logger.info("Saved metrics to %s", args.metrics)


if __name__ == "__main__":
    main()
