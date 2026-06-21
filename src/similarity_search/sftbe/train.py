

import logging
import math
import os
import time
from typing import Any, cast

import torch
import torch.nn.functional as F
from torch.amp import GradScaler, autocast  # pyright: ignore[reportPrivateImportUsage]
from torch.optim import AdamW

from similarity_search.sftbe.config import DATA_CONFIG, MODEL_CONFIG, TRAIN_CONFIG, get_device
from similarity_search.sftbe.dataset import (
    RandomIndexSplitDataset,
    STSBDataset,
    WikipediaDistillationDataset,
    create_dataloader,
    get_tokenizer,
)
from similarity_search.sftbe.losses import Stage0TeacherDistillationLoss
from similarity_search.sftbe.model import create_sftbe_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)


class CosineAnnealingWithWarmup:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int):
        self.optimizer = optimizer
        self.warmup_steps = max(1, warmup_steps)
        self.total_steps = max(1, total_steps)
        self.base_lr = optimizer.param_groups[0]["lr"]
        self.current_step = 0

    def step(self):
        self.current_step += 1
        if self.current_step <= self.warmup_steps:
            lr = self.base_lr * self.current_step / self.warmup_steps
        else:
            progress = (self.current_step - self.warmup_steps) / max(
                1, self.total_steps - self.warmup_steps
            )
            progress = min(1.0, max(0.0, progress))
            lr = self.base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


def save_checkpoint(model, optimizer, scheduler, epoch, step, loss, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_step": scheduler.current_step,
            "epoch": epoch,
            "step": step,
            "loss": loss,
        },
        path,
    )
    logger.info("Saved checkpoint: %s", path)


def load_checkpoint(model, optimizer, scheduler, path, device):
    if not os.path.exists(path):
        return 0, 0
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    scheduler.current_step = checkpoint["scheduler_step"]
    logger.info("Loaded checkpoint: %s", path)
    return int(checkpoint["epoch"]), int(checkpoint["step"])


def accumulation_divisor(batch_idx: int, num_batches: int, accumulation_steps: int) -> int:
    block_start = (batch_idx // accumulation_steps) * accumulation_steps
    return min(block_start + accumulation_steps, num_batches) - block_start


def should_step(batch_idx: int, num_batches: int, accumulation_steps: int) -> bool:
    return (batch_idx + 1) % accumulation_steps == 0 or batch_idx + 1 == num_batches


def load_teacher(device):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise RuntimeError("Cần cài sentence-transformers để train Stage 0.") from exc

    teacher = SentenceTransformer(TRAIN_CONFIG["teacher_model"], device=str(device))
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    teacher_dim = teacher.get_sentence_embedding_dimension()
    if teacher_dim != MODEL_CONFIG["hidden_size"]:
        raise ValueError(
            f"Teacher dim phải bằng student dim: {teacher_dim} != {MODEL_CONFIG['hidden_size']}"
        )
    return teacher


def teacher_encode(teacher, tokenizer, input_ids, device):
    texts = tokenizer.batch_decode(
        input_ids.detach().cpu().tolist(),
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    with torch.no_grad():
        embeddings = teacher.encode(
            texts,
            batch_size=TRAIN_CONFIG["teacher_batch_size"],
            convert_to_tensor=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
    return embeddings.to(device=device, dtype=torch.float32)


def evaluate_stsb(model, loader, device):
    from scipy.stats import spearmanr

    model.eval()
    predictions, labels = [], []
    with torch.no_grad():
        for batch in loader:
            ids_a = batch["input_ids_a"].to(device)
            mask_a = batch["attention_mask_a"].to(device)
            ids_b = batch["input_ids_b"].to(device)
            mask_b = batch["attention_mask_b"].to(device)
            emb_a = model(ids_a, mask_a)
            emb_b = model(ids_b, mask_b)
            cosine = F.cosine_similarity(emb_a, emb_b, dim=-1)
            predictions.extend(cosine.cpu().tolist())
            labels.extend(batch["score"].tolist())
    model.train()
    correlation = spearmanr(predictions, labels)[0]
    return float(cast(Any, correlation))


def evaluate_distillation(model, teacher, tokenizer, loader, criterion, device, use_amp):
    model.eval()
    total_loss, total_samples = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            targets = teacher_encode(teacher, tokenizer, ids, device)
            with autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(ids, mask), targets)
            total_loss += loss.item() * ids.size(0)
            total_samples += ids.size(0)
    model.train()
    avg_loss = total_loss / max(1, total_samples)
    return avg_loss, 1.0 - avg_loss


def main():
    device = get_device()
    tokenizer = get_tokenizer(DATA_CONFIG["tokenizer_name"])
    os.makedirs(TRAIN_CONFIG["checkpoint_dir"], exist_ok=True)

    model = create_sftbe_model(MODEL_CONFIG).to(device)
    teacher = load_teacher(device)
    criterion = Stage0TeacherDistillationLoss()

    full_dataset = WikipediaDistillationDataset(TRAIN_CONFIG["data_cache_dir"])
    train_dataset = RandomIndexSplitDataset(
        full_dataset,
        "train",
        validation_ratio=TRAIN_CONFIG["validation_ratio"],
        seed=TRAIN_CONFIG["split_seed"],
    )
    val_dataset = RandomIndexSplitDataset(
        full_dataset,
        "validation",
        validation_ratio=TRAIN_CONFIG["validation_ratio"],
        seed=TRAIN_CONFIG["split_seed"],
    )

    num_workers = 8 if device.type == "cuda" else 0
    batch_size = TRAIN_CONFIG["batch_size"]
    train_loader = create_dataloader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_loader = create_dataloader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, drop_last=False
    )
    stsb_loader = create_dataloader(
        STSBDataset(
            TRAIN_CONFIG["data_cache_dir"],
            tokenizer,
            split="validation",
            max_length=MODEL_CONFIG["max_seq_length"],
        ),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        drop_last=False,
    )

    optimizer = AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG["learning_rate"],
        betas=(TRAIN_CONFIG["adam_beta1"], TRAIN_CONFIG["adam_beta2"]),
        eps=TRAIN_CONFIG["adam_epsilon"],
        weight_decay=TRAIN_CONFIG["weight_decay"],
    )
    accumulation_steps = max(1, TRAIN_CONFIG["gradient_accumulation_steps"])
    updates_per_epoch = math.ceil(len(train_loader) / accumulation_steps)
    total_updates = updates_per_epoch * TRAIN_CONFIG["epochs"]
    scheduler = CosineAnnealingWithWarmup(
        optimizer,
        warmup_steps=int(TRAIN_CONFIG["warmup_ratio"] * total_updates),
        total_steps=total_updates,
    )
    use_amp = TRAIN_CONFIG["use_amp_on_cuda"] and device.type == "cuda"
    scaler = GradScaler(enabled=use_amp)

    latest_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], "stage0_latest.pt")
    final_path = os.path.join(TRAIN_CONFIG["checkpoint_dir"], "stage0_final.pt")
    start_epoch, global_step = load_checkpoint(model, optimizer, scheduler, latest_path, device)

    logger.info("SFT-BE params: %.1fM", model.count_parameters() / 1e6)
    logger.info("Train samples: %s | validation samples: %s", len(train_dataset), len(val_dataset))
    logger.info("Batch: %s | grad accumulation: %s | AMP: %s", batch_size, accumulation_steps, use_amp)

    start_time = time.time()
    for epoch in range(start_epoch, TRAIN_CONFIG["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        epoch_loss, seen_batches = 0.0, 0

        for batch_idx, batch in enumerate(train_loader):
            if epoch == start_epoch and batch_idx < global_step % len(train_loader):
                continue

            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            targets = teacher_encode(teacher, tokenizer, ids, device)

            with autocast(device_type=device.type, enabled=use_amp):
                loss = criterion(model(ids, mask), targets)

            raw_loss = loss.detach()
            loss = loss / accumulation_divisor(batch_idx, len(train_loader), accumulation_steps)
            scaler.scale(loss).backward()

            if should_step(batch_idx, len(train_loader), accumulation_steps):
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)

            epoch_loss += raw_loss.item()
            seen_batches += 1
            global_step += 1

            if global_step % TRAIN_CONFIG["log_every_steps"] == 0:
                avg_loss = epoch_loss / max(1, seen_batches)
                elapsed = max(1e-9, time.time() - start_time)
                logger.info(
                    "Epoch %s/%s | step %s/%s | loss %.4f | lr %.2e | %.2f batch/s",
                    epoch + 1,
                    TRAIN_CONFIG["epochs"],
                    global_step,
                    len(train_loader) * TRAIN_CONFIG["epochs"],
                    avg_loss,
                    scheduler.get_lr(),
                    seen_batches / elapsed,
                )

            if global_step % TRAIN_CONFIG["checkpoint_every_steps"] == 0:
                save_checkpoint(
                    model, optimizer, scheduler, epoch, global_step,
                    epoch_loss / max(1, seen_batches), latest_path
                )

        val_loss, val_cosine = evaluate_distillation(
            model, teacher, tokenizer, val_loader, criterion, device, use_amp
        )
        stsb = evaluate_stsb(model, stsb_loader, device)
        avg_loss = epoch_loss / max(1, seen_batches)
        logger.info(
            "Epoch %s done | train loss %.4f | val loss %.4f | val cosine %.4f | STS-B %.4f",
            epoch + 1,
            avg_loss,
            val_loss,
            val_cosine,
            stsb,
        )
        save_checkpoint(model, optimizer, scheduler, epoch + 1, global_step, avg_loss, latest_path)

    torch.save(model.state_dict(), final_path)
    logger.info("Saved final model: %s", final_path)


if __name__ == "__main__":
    main()
