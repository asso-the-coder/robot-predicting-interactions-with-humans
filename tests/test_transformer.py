from __future__ import annotations

from pathlib import Path

import pytest
import torch

from engagement_intent.models import TransformerClassifier


def test_transformer_emits_one_finite_raw_logit_per_sequence() -> None:
    model = TransformerClassifier(
        input_size=12,
        model_size=16,
        num_heads=4,
        num_layers=2,
        feedforward_size=32,
        dropout=0.0,
        max_sequence_length=10,
    )
    logits = model(torch.randn(4, 10, 12))

    assert logits.shape == (4,)
    assert torch.isfinite(logits).all()


def test_transformer_uses_temporal_position_information() -> None:
    torch.manual_seed(7)
    model = TransformerClassifier(
        input_size=3,
        model_size=12,
        num_heads=3,
        num_layers=1,
        feedforward_size=24,
        dropout=0.0,
        max_sequence_length=4,
    )
    model.eval()
    sequence = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])

    original = model(sequence)
    reversed_sequence = model(sequence.flip(1))

    assert not torch.allclose(original, reversed_sequence)


def test_transformer_rejects_invalid_attention_dimensions() -> None:
    with pytest.raises(ValueError, match="divisible"):
        TransformerClassifier(input_size=6, model_size=10, num_heads=4)


def test_transformer_checkpoint_round_trip_reproduces_logits(tmp_path: Path) -> None:
    torch.manual_seed(9)
    settings = {
        "input_size": 3,
        "model_size": 8,
        "num_heads": 2,
        "num_layers": 1,
        "feedforward_size": 16,
        "dropout": 0.0,
        "max_sequence_length": 4,
    }
    model = TransformerClassifier(**settings)
    model.eval()
    batch = torch.randn(2, 4, 3)
    expected = model(batch)
    checkpoint = tmp_path / "transformer.pt"
    torch.save(model.state_dict(), checkpoint)

    restored = TransformerClassifier(**settings)
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    restored.eval()

    torch.testing.assert_close(restored(batch), expected)


def test_transformer_rejects_sequences_longer_than_configured() -> None:
    model = TransformerClassifier(
        input_size=3,
        model_size=8,
        num_heads=2,
        num_layers=1,
        feedforward_size=16,
        max_sequence_length=3,
    )

    with pytest.raises(ValueError, match="exceeds configured maximum"):
        model(torch.randn(2, 4, 3))
