from __future__ import annotations
"""Autoencoder registry for training and evaluation."""
import os
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'
os.environ['TRITON_BACKEND'] = 'cuda'


from typing import Any, Dict, Tuple
from pytorch_lightning import LightningModule

# from .models.mnist import MNISTAutoencoder, Config as MNISTConfig

import importlib
import pkgutil, os

# Dynamically discover all model classes and configs in models submodule
from . import models

AUTOENCODER_REGISTRY: Dict[str, Dict[str, Any]] = {}

print("Discovering autoencoders in the 'models' submodule...")

for module_name in os.listdir(os.path.join(os.path.dirname(__file__), "models")):

    # Only look into subpackages starting with "project"
    if not module_name.startswith("project"):
        continue

    package = importlib.import_module(f".models.{module_name}", package=__name__)

    # Second level: models/project*/<modules>
    for _, submodule_name, _ in pkgutil.iter_modules(package.__path__):
        full_name = f".models.{module_name}.{submodule_name}"
        mod = importlib.import_module(full_name, package=__name__)

        model_cls = None
        config_cls = None

        for attr in dir(mod):
            obj = getattr(mod, attr)
            if isinstance(obj, type):
                name = attr.lower()
                if name.endswith("autoencoder") or name.endswith("diffusion"):
                    model_cls = obj
                elif name.endswith("config"):
                    config_cls = obj

        if model_cls and config_cls:
            key = f"{module_name}.{submodule_name}"
            # print(f"Registering network '{key}' with model class '{model_cls.__name__}' and config '{config_cls.__name__}'")
            AUTOENCODER_REGISTRY[key] = {
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