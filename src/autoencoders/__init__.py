"""Autoencoder registry for training and evaluation."""
from __future__ import annotations
from typing import Any, Dict, Tuple
from pytorch_lightning import LightningModule

from .models.conv_autoencoder import ConvAutoEncoder, Config as CVAE_Config

AUTOENCODER_REGISTRY: Dict[str, Dict[str, Any]] = {
    "conv_autoencoder": {
        "model_class": ConvAutoEncoder,
        "default_config": CVAE_Config(),
    }
}


def list_autoencoders() -> Tuple[str, ...]:
    """Return all registered autoencoder keys."""
    return tuple(AUTOENCODER_REGISTRY.keys())


def get_model(name: str, config: Any | None = None) -> LightningModule:
    """Instantiate a model from the registry."""
    if name not in AUTOENCODER_REGISTRY:
        raise KeyError(f"Unknown autoencoder '{name}'")
    return AUTOENCODER_REGISTRY[name]["model_class"](config)


def get_default_config(name: str) -> Any:
    """Return the default configuration for the given model key."""
    if name not in AUTOENCODER_REGISTRY:
        raise KeyError(f"Unknown autoencoder '{name}'")
    return AUTOENCODER_REGISTRY[name]["default_config"]