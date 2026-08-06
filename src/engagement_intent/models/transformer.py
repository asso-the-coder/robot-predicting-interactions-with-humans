"""Transformer encoder for fixed-length pedestrian observation sequences."""

from __future__ import annotations

import math

import torch
from torch import nn


class SinusoidalPositionEncoding(nn.Module):
    """Add deterministic temporal position information to batch-first embeddings."""

    def __init__(self, embedding_size: int, max_sequence_length: int) -> None:
        super().__init__()
        if embedding_size <= 0 or max_sequence_length <= 0:
            raise ValueError("Embedding size and maximum sequence length must be positive")

        positions = torch.arange(max_sequence_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(
            torch.arange(0, embedding_size, 2, dtype=torch.float32)
            * (-math.log(10_000.0) / embedding_size)
        )
        encoding = torch.zeros(max_sequence_length, embedding_size)
        encoding[:, 0::2] = torch.sin(positions * frequencies)
        if embedding_size > 1:
            encoding[:, 1::2] = torch.cos(
                positions * frequencies[: encoding[:, 1::2].shape[1]]
            )
        self.register_buffer("encoding", encoding.unsqueeze(0), persistent=False)

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if embeddings.ndim != 3:
            raise ValueError(
                f"Expected [batch, time, embedding], got {tuple(embeddings.shape)}"
            )
        if embeddings.shape[1] > self.encoding.shape[1]:
            raise ValueError(
                f"Sequence length {embeddings.shape[1]} exceeds configured maximum "
                f"{self.encoding.shape[1]}"
            )
        return embeddings + self.encoding[:, : embeddings.shape[1]]


class TransformerClassifier(nn.Module):
    """Map a `[batch, time, features]` sequence to one raw binary logit."""

    def __init__(
        self,
        input_size: int,
        model_size: int = 64,
        num_heads: int = 4,
        num_layers: int = 2,
        feedforward_size: int = 128,
        dropout: float = 0.2,
        max_sequence_length: int = 32,
    ) -> None:
        super().__init__()
        if min(input_size, model_size, num_heads, num_layers, feedforward_size) <= 0:
            raise ValueError("Transformer dimensions and layer count must be positive")
        if model_size % num_heads != 0:
            raise ValueError("Model size must be divisible by the number of attention heads")
        if not 0.0 <= dropout < 1.0:
            raise ValueError("Dropout must be in [0, 1)")

        self.input_projection = nn.Linear(input_size, model_size)
        self.input_normalization = nn.LayerNorm(model_size)
        self.position_encoding = SinusoidalPositionEncoding(
            model_size, max_sequence_length=max_sequence_length + 1
        )
        self.class_token = nn.Parameter(torch.zeros(1, 1, model_size))
        nn.init.normal_(self.class_token, std=0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_size,
            nhead=num_heads,
            dim_feedforward=feedforward_size,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers, enable_nested_tensor=False
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(model_size),
            nn.Dropout(dropout),
            nn.Linear(model_size, 1),
        )

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        if sequences.ndim != 3:
            raise ValueError(f"Expected [batch, time, features], got {tuple(sequences.shape)}")
        embeddings = self.input_normalization(self.input_projection(sequences))
        class_tokens = self.class_token.expand(sequences.shape[0], -1, -1)
        embeddings = torch.cat((class_tokens, embeddings), dim=1)
        encoded = self.encoder(self.position_encoding(embeddings))
        return self.classifier(encoded[:, 0]).squeeze(-1)
