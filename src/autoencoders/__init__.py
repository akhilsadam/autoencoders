"""Autoencoder registry for training and evaluation."""
from __future__ import annotations
from typing import Any, Dict, Tuple
from pytorch_lightning import LightningModule

from .models.conv_autoencoder import AutoEncoderConfig, LitAutoEncoder, build_model

AUTOENCODER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "conv_autoencoder": {
        "model_cls": LitAutoEncoder,
        "builder": build_model,
        "default_config": AutoEncoderConfig(),
    }
}


def list_autoencoders() -> Tuple[str, ...]:
    """Return all registered autoencoder keys."""
    return tuple(AUTOENCODER_REGISTRY.keys())


def get_model(name: str, config: Any | None = None) -> LightningModule:
    """Instantiate a model from the registry."""
    if name not in AUTOENCODER_REGISTRY:
        raise KeyError(f"Unknown autoencoder '{name}'")
    builder = AUTOENCODER_REGISTRY[name]["builder"]
    return builder(config)


def get_default_config(name: str) -> Any:
    """Return the default configuration for the given model key."""
    if name not in AUTOENCODER_REGISTRY:
        raise KeyError(f"Unknown autoencoder '{name}'")
    return AUTOENCODER_REGISTRY[name]["default_config"]