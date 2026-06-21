import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]


def get_device():
    import torch

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


MODEL_CONFIG = {
    "vocab_size": 30_522,
    "embedding_dim": 128,
    "hidden_size": 768,
    "max_seq_length": 128,
    "num_layers": 6,
    "num_heads": 12,
    "ffn_hidden_size": 3072,
    "dropout": 0.1,
    "layer_norm_eps": 1e-12,
    "position_encoding": "sinusoidal",
    "norm_type": "pre_ln",
}


TRAIN_CONFIG = {
    "learning_rate": 5e-4,
    "weight_decay": 0.01,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "adam_epsilon": 1e-8,
    "warmup_ratio": 0.1,
    "epochs": int(os.environ.get("SFTBE_EPOCHS", "1")),
    "batch_size": int(os.environ.get("SFTBE_BATCH_SIZE", "64")),
    "gradient_accumulation_steps": int(os.environ.get("SFTBE_GRAD_ACCUM", "4")),
    "checkpoint_every_steps": int(os.environ.get("SFTBE_CHECKPOINT_EVERY_STEPS", "10000")),
    "log_every_steps": int(os.environ.get("SFTBE_LOG_EVERY_STEPS", "500")),
    "use_amp_on_cuda": True,
    "checkpoint_dir": os.environ.get(
        "SFTBE_CHECKPOINT_DIR",
        str(PROJECT_ROOT / "models" / "sftbe_checkpoint"),
    ),
    "data_cache_dir": os.environ.get(
        "SFTBE_CACHE_DIR",
        str(PROJECT_ROOT / "data_cache"),
    ),
    "teacher_model": os.environ.get(
        "SFTBE_TEACHER_MODEL",
        "sentence-transformers/all-mpnet-base-v2",
    ),
    "teacher_batch_size": int(os.environ.get("SFTBE_TEACHER_BATCH_SIZE", "64")),
    "validation_ratio": float(os.environ.get("SFTBE_VALIDATION_RATIO", "0.02")),
    "split_seed": int(os.environ.get("SFTBE_SPLIT_SEED", "42")),
}


DATA_CONFIG = {
    "tokenizer_name": "bert-base-uncased",
    "stsb_dataset": "mteb/stsbenchmark-sts",
    "wikipedia_dataset": ("wikimedia/wikipedia", "20231101.en"),
}
