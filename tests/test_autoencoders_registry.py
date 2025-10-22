"""Smoke tests covering every registered autoencoder."""
from __future__ import annotations

import pytest
import torch

from src.autoencoders import get_default_config, get_model, list_autoencoders


def test_registry_contains_models() -> None:
    assert list_autoencoders(), "Autoencoder registry must not be empty"


@pytest.mark.parametrize("name", list_autoencoders())
def test_autoencoder_forward_and_backward(name: str) -> None:
    model = get_model(name, get_default_config(name))

    batch = torch.rand(4, 1, 28, 28)
    device = next(model.parameters()).device
    batch = batch.to(device)

    loss = model.training_step((batch, batch), 0)
    assert torch.isfinite(loss)

    loss.backward()
    for param in model.parameters():
        if param.grad is not None:
            assert torch.isfinite(param.grad).all()
