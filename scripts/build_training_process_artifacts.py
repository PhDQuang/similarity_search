"""Build training-process figures and summary tables from existing logs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sftbe-loss-csv",
        default="models/sftbe_checkpoint/stage0_loss_reconstructed.csv",
    )
    parser.add_argument(
        "--sftbe-summary-json",
        default="models/sftbe_checkpoint/stage0_interval_loss_summary.json",
    )
    parser.add_argument(
        "--cross-trainer-state",
        default="models/allnli-cross-encoder-nli/checkpoints/checkpoint-4688/trainer_state.json",
    )
    parser.add_argument(
        "--minilm-metadata",
        default="outputs/finetuned_minilm/training_metadata.json",
    )
    parser.add_argument(
        "--cross-metadata",
        default="outputs/cross_encoder_outputs/cross_encoder_training_metadata.json",
    )
    parser.add_argument("--figure-dir", default="outputs/figures/training")
    parser.add_argument("--table-dir", default="outputs/tables")
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def save_sftbe_figures(loss_csv: Path, figure_dir: Path) -> dict[str, Any] | None:
    if not loss_csv.exists():
        return None

    frame = pd.read_csv(loss_csv)
    loss_path = figure_dir / "sftbe_stage0_loss.png"
    lr_path = figure_dir / "sftbe_stage0_learning_rate.png"

    plt.figure(figsize=(10, 5))
    plt.plot(frame["global_step"], frame["interval_loss_reconstructed"], label="Interval loss", linewidth=1.2)
    if "interval_loss_rolling_mean_50" in frame.columns:
        plt.plot(frame["global_step"], frame["interval_loss_rolling_mean_50"], label="Rolling mean (50)", linewidth=1.8)
    plt.title("SFT-BE Stage 0 Distillation Loss")
    plt.xlabel("Global step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=180)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(frame["global_step"], frame["lr"], color="#7a4f9f", linewidth=1.4)
    plt.title("SFT-BE Learning Rate Schedule")
    plt.xlabel("Global step")
    plt.ylabel("Learning rate")
    plt.tight_layout()
    plt.savefig(lr_path, dpi=180)
    plt.close()

    return {
        "train_loss_first": float(frame["interval_loss_reconstructed"].iloc[0]),
        "train_loss_last": float(frame["interval_loss_reconstructed"].iloc[-1]),
        "train_loss_min": float(frame["interval_loss_reconstructed"].min()),
        "steps": int(frame["global_step"].iloc[-1]),
        "elapsed_hours": float(frame["elapsed_hours"].iloc[-1]),
        "loss_figure": str(loss_path),
        "lr_figure": str(lr_path),
    }


def trainer_history_frame(trainer_state: Path) -> pd.DataFrame | None:
    state = read_json(trainer_state)
    if not state:
        return None
    history = state.get("log_history", [])
    if not history:
        return None
    return pd.DataFrame(history)


def save_cross_encoder_figures(trainer_state: Path, figure_dir: Path) -> dict[str, Any] | None:
    history = trainer_history_frame(trainer_state)
    if history is None:
        return None

    train_rows = history[history["loss"].notna()].copy() if "loss" in history else pd.DataFrame()
    eval_rows = history[history["eval_loss"].notna()].copy() if "eval_loss" in history else pd.DataFrame()
    if train_rows.empty and eval_rows.empty:
        return None

    loss_path = figure_dir / "cross_encoder_train_eval_loss.png"
    metrics_path = figure_dir / "cross_encoder_eval_metrics.png"

    plt.figure(figsize=(10, 5))
    if not train_rows.empty:
        plt.plot(train_rows["step"], train_rows["loss"], label="Train loss", marker="o", markersize=2)
    if not eval_rows.empty:
        plt.plot(eval_rows["step"], eval_rows["eval_loss"], label="Eval loss", marker="s", markersize=4)
    plt.title("Cross-Encoder Training and Evaluation Loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(loss_path, dpi=180)
    plt.close()

    if not eval_rows.empty:
        plt.figure(figsize=(10, 5))
        for column, label in [
            ("eval_accuracy", "Accuracy"),
            ("eval_macro_f1", "Macro F1"),
            ("eval_entailment_f1", "Entailment F1"),
        ]:
            if column in eval_rows:
                plt.plot(eval_rows["step"], eval_rows[column], marker="o", label=label)
        plt.title("Cross-Encoder Validation Metrics")
        plt.xlabel("Step")
        plt.ylabel("Score")
        plt.ylim(0.0, 1.0)
        plt.legend()
        plt.tight_layout()
        plt.savefig(metrics_path, dpi=180)
        plt.close()

    result: dict[str, Any] = {
        "loss_figure": str(loss_path),
        "metrics_figure": str(metrics_path) if not eval_rows.empty else "",
    }
    if not train_rows.empty:
        result.update(
            {
                "train_loss_first": float(train_rows["loss"].iloc[0]),
                "train_loss_last": float(train_rows["loss"].iloc[-1]),
                "steps": int(train_rows["step"].iloc[-1]),
            }
        )
    if not eval_rows.empty:
        result.update(
            {
                "eval_loss_first": float(eval_rows["eval_loss"].iloc[0]),
                "eval_loss_last": float(eval_rows["eval_loss"].iloc[-1]),
                "eval_accuracy_last": float(eval_rows["eval_accuracy"].iloc[-1]),
                "eval_macro_f1_last": float(eval_rows["eval_macro_f1"].iloc[-1]),
            }
        )
    return result


def build_summary_rows(args: argparse.Namespace, figure_dir: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []

    sftbe = save_sftbe_figures(Path(args.sftbe_loss_csv), figure_dir)
    sftbe_summary = read_json(Path(args.sftbe_summary_json)) or {}
    if sftbe:
        rows.append(
            {
                "model": "SFT-BE",
                "training_data": "Wikimedia Wikipedia 20231101.en distillation",
                "loss": "Stage0TeacherDistillationLoss",
                "epochs": 1,
                "batch_size": 16,
                "gradient_accumulation_steps": 8,
                "learning_rate": "cosine schedule",
                "gpu": "RTX 3090",
                "train_loss_start": sftbe["train_loss_first"],
                "train_loss_end": sftbe["train_loss_last"],
                "eval_loss_end": "",
                "main_eval": sftbe_summary.get("points", [{}])[-1].get("interval_cosine_approx", ""),
                "steps": sftbe["steps"],
                "elapsed_hours": sftbe["elapsed_hours"],
                "artifact": sftbe["loss_figure"],
            }
        )

    cross = save_cross_encoder_figures(Path(args.cross_trainer_state), figure_dir)
    cross_metadata = read_json(Path(args.cross_metadata)) or {}
    if cross:
        rows.append(
            {
                "model": "Cross-Encoder NLI",
                "training_data": "AllNLI pair-class",
                "loss": "CrossEntropyLoss",
                "epochs": cross_metadata.get("epochs", 1),
                "batch_size": cross_metadata.get("batch_size", 32),
                "gradient_accumulation_steps": cross_metadata.get("gradient_accumulation_steps", 1),
                "learning_rate": cross_metadata.get("learning_rate", 2e-5),
                "gpu": cross_metadata.get("gpu", ""),
                "train_loss_start": cross.get("train_loss_first", ""),
                "train_loss_end": cross.get("train_loss_last", ""),
                "eval_loss_end": cross.get("eval_loss_last", ""),
                "main_eval": cross_metadata.get("test_nli_metrics", {}).get("test_macro_f1", ""),
                "steps": cross.get("steps", ""),
                "elapsed_hours": cross_metadata.get("train_result", {}).get("train_runtime", 0) / 3600,
                "artifact": cross["loss_figure"],
            }
        )

    minilm = read_json(Path(args.minilm_metadata))
    if minilm:
        rows.append(
            {
                "model": "Fine-tuned MiniLM",
                "training_data": minilm.get("train_dataset", ""),
                "loss": minilm.get("loss", ""),
                "epochs": minilm.get("epochs", ""),
                "batch_size": minilm.get("batch_size", ""),
                "gradient_accumulation_steps": minilm.get("gradient_accumulation_steps", ""),
                "learning_rate": minilm.get("learning_rate", ""),
                "gpu": minilm.get("gpu", ""),
                "train_loss_start": "",
                "train_loss_end": "",
                "eval_loss_end": "",
                "main_eval": minilm.get("final_eval", {}).get("allnli-pair-score-dev_spearman_cosine", ""),
                "steps": "",
                "elapsed_hours": "",
                "artifact": "metadata only; no step-level loss log available",
            }
        )

    return rows


def main() -> None:
    args = parse_args()
    figure_dir = Path(args.figure_dir)
    table_dir = Path(args.table_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    rows = build_summary_rows(args, figure_dir)
    table = pd.DataFrame(rows)
    summary_path = table_dir / "training_process_summary.csv"
    table.to_csv(summary_path, index=False)
    print(table.to_string(index=False))
    print(f"Saved training summary -> {summary_path}")
    print(f"Saved training figures -> {figure_dir}")


if __name__ == "__main__":
    main()

