import math
from typing import cast

import torch
import torch.nn as nn


class FactorizedEmbedding(nn.Module):
    """Token embedding followed by a projection to the model hidden size."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        layer_norm_eps: float = 1e-12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.hidden_size = hidden_size

        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        nn.init.xavier_uniform_(self.token_embedding.weight.data[1:])

        self.projection = nn.Linear(embedding_dim, hidden_size, bias=False)
        nn.init.xavier_uniform_(self.projection.weight)

        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        token_embeds = self.token_embedding(input_ids)
        return self.projection(token_embeds)


class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding."""

    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pe = cast(torch.Tensor, self.get_buffer("pe"))
        return x + pe[:, :x.size(1), :]


class TransformerEmbedding(nn.Module):
    """Input embedding stack for the encoder."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        max_seq_length: int,
        layer_norm_eps: float = 1e-12,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.factorized_embedding = FactorizedEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
        )
        self.positional_encoding = SinusoidalPositionalEncoding(
            d_model=hidden_size,
            max_len=max_seq_length,
        )
        self.layer_norm = nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        embeddings = self.factorized_embedding(input_ids)
        embeddings = self.positional_encoding(embeddings)
        embeddings = self.layer_norm(embeddings)
        return self.dropout(embeddings)
