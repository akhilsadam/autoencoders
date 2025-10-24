"""Autoencoder registry for training and evaluation."""
from __future__ import annotations
from typing import Any, Dict, Tuple
from pytorch_lightning import LightningModule

# from .models.mnist import MNISTAutoencoder, Config as MNISTConfig

import importlib
import pkgutil

# Dynamically discover all model classes and configs in models submodule
import models

AUTOENCODER_REGISTRY: Dict[str, Dict[str, Any]] = {}
for loader, module_name, is_pkg in pkgutil.iter_modules(models.__path__):
    mod = importlib.import_module(f"models.{module_name}")
    # Convention: model class is named 'Autoencoder' or endswith 'Autoencoder', config is 'Config' or endswith 'Config'
    model_cls = None
    config_cls = None
    for attr in dir(mod):
        obj = getattr(mod, attr)
        if isinstance(obj, type):
            if attr.lower().endswith("autoencoder"):
                model_cls = obj
            elif attr.lower().endswith("config"):
                config_cls = obj
    if model_cls and config_cls:
        AUTOENCODER_REGISTRY[module_name] = {
            "model_class": model_cls,
            "default_config": config_cls(),
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