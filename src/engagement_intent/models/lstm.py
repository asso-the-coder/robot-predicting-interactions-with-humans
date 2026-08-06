"""LSTM classifier for fixed-length pedestrian observation sequences."""

from __future__ import annotations

import torch
from torch import nn


class LSTMClassifier(nn.Module):
    """Map a `[batch, time, features]` sequence to one raw binary logit."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 64,
        num_layers: int = 2,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if input_size <= 0 or hidden_size <= 0 or num_layers <= 0:
            raise ValueError("Input size, hidden size, and layer count must be positive")
        recurrent_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=recurrent_dropout,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, 1)

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        if sequences.ndim != 3:
            raise ValueError(f"Expected [batch, time, features], got {tuple(sequences.shape)}")
        _, (hidden, _) = self.lstm(sequences)
        return self.classifier(self.dropout(hidden[-1])).squeeze(-1)

