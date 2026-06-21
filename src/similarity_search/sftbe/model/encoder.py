import torch
import torch.nn as nn

from similarity_search.sftbe.model.embedding import TransformerEmbedding
from similarity_search.sftbe.model.encoder_blocks import TransformerEncoder


class MeanPooling(nn.Module):
    """Attention-mask-aware mean pooling."""

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1).float()
        sum_embeddings = (hidden_states * mask_expanded).sum(dim=1)
        sum_mask = mask_expanded.sum(dim=1).clamp(min=1e-9)
        return sum_embeddings / sum_mask


class SFTBEModel(nn.Module):
    """Shallow Factorized Transformer Bi-Encoder."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        hidden_size: int,
        max_seq_length: int,
        num_layers: int,
        num_heads: int,
        ffn_hidden_size: int,
        dropout: float = 0.1,
        layer_norm_eps: float = 1e-12,
    ):
        super().__init__()
        self.embedding = TransformerEmbedding(
            vocab_size=vocab_size,
            embedding_dim=embedding_dim,
            hidden_size=hidden_size,
            max_seq_length=max_seq_length,
            layer_norm_eps=layer_norm_eps,
            dropout=dropout,
        )
        self.encoder = TransformerEncoder(
            num_layers=num_layers,
            d_model=hidden_size,
            num_heads=num_heads,
            d_ff=ffn_hidden_size,
            dropout=dropout,
            layer_norm_eps=layer_norm_eps,
        )
        self.pooling = MeanPooling()
        self.hidden_size = hidden_size

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        embeddings = self.embedding(input_ids)
        hidden_states = self.encoder(embeddings, attention_mask)
        return self.pooling(hidden_states, attention_mask)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def create_sftbe_model(config: dict) -> SFTBEModel:
    return SFTBEModel(
        vocab_size=config["vocab_size"],
        embedding_dim=config["embedding_dim"],
        hidden_size=config["hidden_size"],
        max_seq_length=config["max_seq_length"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        ffn_hidden_size=config["ffn_hidden_size"],
        dropout=config["dropout"],
        layer_norm_eps=config["layer_norm_eps"],
    )
