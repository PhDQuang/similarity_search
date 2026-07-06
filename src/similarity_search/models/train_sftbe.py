"""Continue training the current SFT-BE checkpoint on the fixed AllNLI split."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import get_cosine_schedule_with_warmup

from similarity_search.models.evaluation import (
    POSITIVE_LABEL,
    SCORE_BY_LABEL,
    binary_confusion,
    entailment_targets,
    evaluate_pair_splits,
    evaluate_retrieval_splits,
    evaluate_test_sample_performance,
    json_safe,
    load_splits,
    save_json,
    save_pair_predictions,
    score_targets,
    validate_pair_class_frame,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", default="data/processed/allnli_70_15_15_clean/pair-class")
    parser.add_argument("--output-dir", default="outputs/sftbe_checkpoint")
    parser.add_argument("--model-dir", default="models/sftbe_checkpoint")
    parser.add_argument("--checkpoint-path", default="models/sftbe_checkpoint/stage0_final.pt")
    parser.add_argument("--num-train-epochs", type=float, default=5.0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--log-every-steps", type=int, default=100)
    parser.add_argument("--early-stopping-patience", type=int, default=2)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-4)
    parser.add_argument("--eval-during-training-samples", type=int, default=20_000)
    parser.add_argument("--retrieval-pool-size", type=int, default=20)
    parser.add_argument("--max-retrieval-queries", type=int, default=0)
    parser.add_argument("--test-sample-size", type=int, default=5_000)
    parser.add_argument("--max-train-rows", type=int, default=0, help="0 means full train split.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--allow-cpu", action="store_true")
    return parser.parse_args()


class PairScoreDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, tokenizer: Any, max_length: int) -> None:
        self.left = frame["premise_clean"].astype(str).tolist()
        self.right = frame["hypothesis_clean"].astype(str).tolist()
        self.scores = frame["label_name"].map(SCORE_BY_LABEL).astype(float).tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.left)

    def _encode(self, text: str) -> dict[str, torch.Tensor]:
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {key: value.squeeze(0) for key, value in encoded.items()}

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        left = self._encode(self.left[index])
        right = self._encode(self.right[index])
        return {
            "input_ids_a": left["input_ids"],
            "attention_mask_a": left["attention_mask"],
            "input_ids_b": right["input_ids"],
            "attention_mask_b": right["attention_mask"],
            "score": torch.tensor(self.scores[index], dtype=torch.float32),
        }


def resolve_checkpoint_path(checkpoint_path: Path) -> Path:
    if checkpoint_path.exists():
        return checkpoint_path

    kaggle_input = Path("/kaggle/input")
    candidates: list[Path] = []
    if kaggle_input.exists():
        candidates.extend(sorted(kaggle_input.glob("**/stage0_final.pt")))
        candidates.extend(sorted(kaggle_input.glob("**/stage0*.pt")))
        candidates.extend(sorted(kaggle_input.glob("**/sftbe*.pt")))

    for candidate in candidates:
        if candidate.exists():
            print(f"Resolved SFT-BE checkpoint from Kaggle input: {candidate}")
            return candidate

    found_pt_files = [str(path) for path in sorted(kaggle_input.glob("**/*.pt"))[:30]] if kaggle_input.exists() else []
    raise FileNotFoundError(
        f"SFT-BE checkpoint not found: {checkpoint_path}. "
        "Upload stage0_final.pt to a Kaggle Dataset and attach it to this notebook. "
        "The script searches /kaggle/input/**/stage0_final.pt, /kaggle/input/**/stage0*.pt, "
        f"and /kaggle/input/**/sftbe*.pt. Found .pt files: {found_pt_files}"
    )


def load_current_sftbe(checkpoint_path: Path, device: torch.device) -> tuple[Any, Any, dict[str, Any]]:
    from similarity_search.sftbe.config import DATA_CONFIG, MODEL_CONFIG
    from similarity_search.sftbe.dataset import get_tokenizer
    from similarity_search.sftbe.model import create_sftbe_model

    checkpoint_path = resolve_checkpoint_path(checkpoint_path)
    tokenizer = get_tokenizer(DATA_CONFIG["tokenizer_name"])
    model = create_sftbe_model(MODEL_CONFIG).to(device)
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state.get("model_state_dict", state), strict=False)
    return model, tokenizer, MODEL_CONFIG


def pair_scores_sftbe(
    model: Any,
    tokenizer: Any,
    frame: pd.DataFrame,
    batch_size: int,
    max_length: int,
    device: torch.device,
    left_col: str,
    right_col: str,
) -> np.ndarray:
    model.eval()
    left = frame[left_col].astype(str).tolist()
    right = frame[right_col].astype(str).tolist()
    scores: list[torch.Tensor] = []
    with torch.no_grad():
        for start in range(0, len(frame), batch_size):
            end = start + batch_size
            encoded_left = tokenizer(
                left[start:end],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            encoded_right = tokenizer(
                right[start:end],
                truncation=True,
                padding=True,
                max_length=max_length,
                return_tensors="pt",
            )
            ids_a = encoded_left["input_ids"].to(device)
            mask_a = encoded_left["attention_mask"].to(device)
            ids_b = encoded_right["input_ids"].to(device)
            mask_b = encoded_right["attention_mask"].to(device)
            emb_a = F.normalize(model(ids_a, mask_a), p=2, dim=-1)
            emb_b = F.normalize(model(ids_b, mask_b), p=2, dim=-1)
            scores.append(F.cosine_similarity(emb_a, emb_b, dim=-1).detach().cpu())
    return torch.cat(scores).numpy() if scores else np.asarray([], dtype=float)


def evaluate_loss(
    model: Any,
    tokenizer: Any,
    frame: pd.DataFrame,
    batch_size: int,
    max_length: int,
    device: torch.device,
) -> float:
    scores = pair_scores_sftbe(model, tokenizer, frame, batch_size, max_length, device, "premise_clean", "hypothesis_clean")
    targets = score_targets(frame)
    return float(np.mean((scores - targets) ** 2))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif args.allow_cpu:
        device = torch.device("cpu")
    else:
        raise RuntimeError("CUDA GPU not found. In Kaggle, enable GPU or pass --allow-cpu for a tiny smoke test.")

    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_dir.mkdir(parents=True, exist_ok=True)

    frames = load_splits(args.input_dir)
    for split, frame in frames.items():
        validate_pair_class_frame(frame, split)
    train_frame = frames["train"]
    if args.max_train_rows > 0 and len(train_frame) > args.max_train_rows:
        train_frame = train_frame.sample(n=args.max_train_rows, random_state=args.seed).reset_index(drop=True)

    model, tokenizer, model_config = load_current_sftbe(Path(args.checkpoint_path), device)
    max_length = int(model_config["max_seq_length"])
    train_dataset = PairScoreDataset(train_frame, tokenizer, max_length)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    updates_per_epoch = max(1, math.ceil(len(train_loader) / max(1, args.gradient_accumulation_steps)))
    total_steps = max(1, int(updates_per_epoch * args.num_train_epochs))
    warmup_steps = int(args.warmup_ratio * total_steps)
    scheduler = get_cosine_schedule_with_warmup(optimizer, warmup_steps, total_steps)
    use_amp = device.type == "cuda"
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    train_log_path = output_dir / "training_log.csv"
    eval_frame = frames["val"]
    if args.eval_during_training_samples > 0 and len(eval_frame) > args.eval_during_training_samples:
        eval_frame = eval_frame.sample(n=args.eval_during_training_samples, random_state=args.seed).reset_index(drop=True)

    global_step = 0
    running_loss = 0.0
    best_eval_mse = float("inf")
    bad_eval_count = 0
    stop_training = False
    best_checkpoint = model_dir / "stage1_allnli_best.pt"
    model.train()
    with train_log_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["step", "epoch", "train_loss", "eval_mse", "learning_rate"])
        writer.writeheader()
        optimizer.zero_grad(set_to_none=True)
        for epoch in range(max(1, math.ceil(args.num_train_epochs))):
            for batch_index, batch in enumerate(train_loader):
                ids_a = batch["input_ids_a"].to(device)
                mask_a = batch["attention_mask_a"].to(device)
                ids_b = batch["input_ids_b"].to(device)
                mask_b = batch["attention_mask_b"].to(device)
                targets = batch["score"].to(device)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    emb_a = F.normalize(model(ids_a, mask_a), p=2, dim=-1)
                    emb_b = F.normalize(model(ids_b, mask_b), p=2, dim=-1)
                    cosine = F.cosine_similarity(emb_a, emb_b, dim=-1)
                    loss = F.mse_loss(cosine, targets)
                    scaled_loss = loss / max(1, args.gradient_accumulation_steps)
                scaler.scale(scaled_loss).backward()
                running_loss += float(loss.detach().cpu())

                should_step = (batch_index + 1) % max(1, args.gradient_accumulation_steps) == 0 or (
                    batch_index + 1 == len(train_loader)
                )
                if should_step:
                    scaler.step(optimizer)
                    scaler.update()
                    scheduler.step()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                    if global_step % args.log_every_steps == 0:
                        avg_loss = running_loss / args.log_every_steps
                        running_loss = 0.0
                        eval_mse = evaluate_loss(
                            model,
                            tokenizer,
                            eval_frame,
                            args.eval_batch_size,
                            max_length,
                            device,
                        )
                        model.train()
                        row = {
                            "step": global_step,
                            "epoch": epoch + 1,
                            "train_loss": avg_loss,
                            "eval_mse": eval_mse,
                            "learning_rate": scheduler.get_last_lr()[0],
                        }
                        writer.writerow(row)
                        file.flush()
                        print(row)
                        if eval_mse < best_eval_mse - args.early_stopping_min_delta:
                            best_eval_mse = eval_mse
                            bad_eval_count = 0
                            torch.save(
                                {
                                    "model_state_dict": model.state_dict(),
                                    "source_checkpoint": str(args.checkpoint_path),
                                    "fixed_dataset": "AllNLI pair-class full 70/15/15",
                                    "model_config": model_config,
                                    "training_args": vars(args),
                                    "best_eval_mse": best_eval_mse,
                                    "global_step": global_step,
                                },
                                best_checkpoint,
                            )
                        else:
                            bad_eval_count += 1
                            if bad_eval_count >= args.early_stopping_patience:
                                print(
                                    "Early stopping: "
                                    f"best_eval_mse={best_eval_mse:.6f}, "
                                    f"bad_eval_count={bad_eval_count}"
                                )
                                stop_training = True
                                break
                if global_step >= total_steps:
                    break
                if stop_training:
                    break
            if global_step >= total_steps:
                break
            if stop_training:
                break

    final_checkpoint = model_dir / "stage1_allnli_final.pt"
    if best_checkpoint.exists():
        best_state = torch.load(best_checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(best_state["model_state_dict"], strict=False)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "source_checkpoint": str(args.checkpoint_path),
            "fixed_dataset": "AllNLI pair-class full 70/15/15",
            "model_config": model_config,
            "training_args": vars(args),
        },
        final_checkpoint,
    )

    val_scores = pair_scores_sftbe(model, tokenizer, frames["val"], args.eval_batch_size, max_length, device, "premise_clean", "hypothesis_clean")
    test_scores = pair_scores_sftbe(model, tokenizer, frames["test"], args.eval_batch_size, max_length, device, "premise_clean", "hypothesis_clean")
    threshold, pair_report = evaluate_pair_splits(frames["val"], frames["test"], val_scores, test_scores)
    retrieval = evaluate_retrieval_splits(
        frames,
        score_pairs=lambda retrieval_frame: pair_scores_sftbe(
            model,
            tokenizer,
            retrieval_frame,
            args.eval_batch_size,
            max_length,
            device,
            "text_a",
            "text_b",
        ),
        pool_size=args.retrieval_pool_size,
        max_queries=args.max_retrieval_queries,
        seed=args.seed,
    )
    test5k_frame, test5k_scores, test5k_report = evaluate_test_sample_performance(
        frames["test"],
        score_pairs=lambda sample_frame: pair_scores_sftbe(
            model,
            tokenizer,
            sample_frame,
            args.eval_batch_size,
            max_length,
            device,
            "premise_clean",
            "hypothesis_clean",
        ),
        threshold=threshold,
        sample_size=args.test_sample_size,
        seed=args.seed,
    )

    metrics: dict[str, Any] = {
        "task": "entailment-as-semantic-similarity",
        "fixed_dataset": "AllNLI pair-class full 70/15/15",
        "positive_label": POSITIVE_LABEL,
        "model": {
            "name": "SFT-BE AllNLI fine-tuned",
            "source_checkpoint": str(args.checkpoint_path),
            "final_checkpoint": str(final_checkpoint),
            "trained_in_project": True,
            "loss": "CosineMSELoss",
            "score_mapping": SCORE_BY_LABEL,
            "hidden_size": int(model_config["hidden_size"]),
        },
        **pair_report,
        "retrieval": retrieval,
        "test_sample_performance": test5k_report,
    }
    metadata = {
        "source_checkpoint": str(args.checkpoint_path),
        "fixed_dataset": "AllNLI pair-class full 70/15/15",
        "train_rows": len(train_frame),
        "val_rows": len(frames["val"]),
        "test_rows": len(frames["test"]),
        "loss": "CosineMSELoss",
        "epochs": args.num_train_epochs,
        "batch_size": args.batch_size,
        "eval_batch_size": args.eval_batch_size,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "warmup_ratio": args.warmup_ratio,
        "weight_decay": args.weight_decay,
        "early_stopping_patience": args.early_stopping_patience,
        "early_stopping_min_delta": args.early_stopping_min_delta,
        "best_eval_mse": best_eval_mse,
        "best_checkpoint": str(best_checkpoint) if best_checkpoint.exists() else None,
        "device": str(device),
        "final_checkpoint": str(final_checkpoint),
        "training_log": str(train_log_path),
    }

    save_json(metrics, output_dir / "metrics.json")
    save_json(json_safe(metadata), output_dir / "training_metadata.json")
    save_pair_predictions(frames["val"], val_scores, threshold, "sftbe_cosine", output_dir / "val_predictions.csv")
    save_pair_predictions(frames["test"], test_scores, threshold, "sftbe_cosine", output_dir / "test_predictions.csv")
    save_json(test5k_report, output_dir / "test5k_performance.json")
    save_pair_predictions(
        test5k_frame,
        test5k_scores,
        threshold,
        "sftbe_cosine",
        output_dir / "test5k_predictions.csv",
    )
    binary_confusion(entailment_targets(frames["test"]), test_scores, threshold).to_csv(
        output_dir / "binary_confusion_matrix.csv"
    )
    print(json.dumps(json_safe(metadata), indent=2))
    print(f"Saved SFT-BE checkpoint -> {final_checkpoint}")
    print(f"Saved SFT-BE outputs -> {output_dir}")


if __name__ == "__main__":
    main()

