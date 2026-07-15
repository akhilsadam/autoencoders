from __future__ import annotations
"""Autoencoder registry for training and evaluation."""
import os
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'
os.environ['TRITON_BACKEND'] = 'cuda'

from typing import Any, Dict, Tuple
from pytorch_lightning import LightningModule

from importlib.metadata import entry_points
import importlib
import pkgutil, os

AUTOENCODER_REGISTRY: Dict[str, Dict[str, Any]] = {}

print("Discovering autoencoders...")

# First: discover models from installed packages
_package_names = os.listdir('packages')
print(f'Packages: {_package_names}')

for pkg_name in _package_names:
    try:
        file = f'packages/{pkg_name}/src'
        pkg_name = os.listdir(file)[0].split('.')[0]
        pkg = importlib.import_module(pkg_name)
        if hasattr(pkg, 'MODEL_REGISTRY'):
            registry = getattr(pkg, 'MODEL_REGISTRY')
            for model_name in registry.list_all():
                model_cls = registry.get(model_name)
                config_cls = registry.get_config_cls(model_name)
                key = f"{model_name}"
                AUTOENCODER_REGISTRY[key] = {
                    "model_class": model_cls,
                    "default_config": config_cls() if config_cls else None,
                }
                print(f"  Loaded: {key}")
    except (ModuleNotFoundError, ImportError, AttributeError, FileNotFoundError):
        pass

# Second: discover models from orchestrator
from . import models

for module_name in os.listdir(os.path.join(os.path.dirname(__file__), "models")):

    # Only look into subpackages starting with "project"
    if not module_name.startswith("project"):
        continue

    package = importlib.import_module(f".models.{module_name}", package=__name__)

    # Second level: models/project*/<modules>
    for _, submodule_name, _ in pkgutil.iter_modules(package.__path__):
        full_name = f".models.{module_name}.{submodule_name}"
        try:
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
                AUTOENCODER_REGISTRY[key] = {
                    "model_class": model_cls,
                    "default_config": config_cls(),
                }
                print(f"  Loaded: {key}")
        except (ModuleNotFoundError, ImportError) as e:
            print(f"  Warning: Could not load {full_name}: {e}")

def list_autoencoders() -> Tuple[str, ...]:
    """Return all registered autoencoder keys."""
    return tuple(AUTOENCODER_REGISTRY.keys())


def get_model(name: str, config: Any | None = None) -> LightningModule:
    """Instantiate a model from the registry."""
    if name not in AUTOENCODER_REGISTRY:
        raise KeyError(f"Unknown autoencoder '{name}'")

    # If no config provided, use default from registry
    if config is None:
        config = AUTOENCODER_REGISTRY[name]["default_config"]

    return AUTOENCODER_REGISTRY[name]["model_class"](config)


def get_default_config(name: str) -> Any:
    """Return the default configuration for the given model key."""
    if name not in AUTOENCODER_REGISTRY:
        raise KeyError(f"Unknown autoencoder '{name}'")
    return AUTOENCODER_REGISTRY[name]["default_config"]