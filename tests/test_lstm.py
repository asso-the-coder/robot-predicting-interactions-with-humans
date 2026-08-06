from __future__ import annotations

from pathlib import Path

import torch
from torch import nn

from engagement_intent.models import LSTMClassifier


def test_lstm_emits_one_finite_raw_logit_per_sequence() -> None:
    model = LSTMClassifier(input_size=6, hidden_size=8, num_layers=1, dropout=0.0)
    logits = model(torch.randn(4, 10, 6))

    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()


def test_lstm_checkpoint_round_trip_reproduces_logits(tmp_path: Path) -> None:
    torch.manual_seed(4)
    model = LSTMClassifier(input_size=3, hidden_size=5, num_layers=1, dropout=0.0)
    model.eval()
    batch = torch.randn(2, 4, 3)
    expected = model(batch)
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)

    restored = LSTMClassifier(input_size=3, hidden_size=5, num_layers=1, dropout=0.0)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    restored.eval()

    torch.testing.assert_close(restored(batch), expected)


def test_weighted_binary_loss_is_finite_for_both_classes() -> None:
    logits = torch.tensor([-1.0, 1.0])
    targets = torch.tensor([0.0, 1.0])
    loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(3.0))(logits, targets)
    assert torch.isfinite(loss)

