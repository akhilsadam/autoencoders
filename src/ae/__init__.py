"""Autoencoder core models and training infrastructure."""
from mura.registry import Registry
from .train import train_ae

__version__ = '0.1.0'

# Model registry
MODEL_REGISTRY = Registry[object](name="AE_MODEL_REGISTRY")

# Register available models
def _register_ae_models():
    """Register all AE models."""
    from .models.mnist import MNISTAutoencoder, Config as MNISTConfig
    from .models.spatial import SpatialAutoencoder, Config as SpatialConfig

    MODEL_REGISTRY.register("develop.mnist", MNISTConfig, MNISTAutoencoder)
    MODEL_REGISTRY.register("develop.spatial", SpatialConfig, SpatialAutoencoder)

_register_ae_models()

__all__ = ['MODEL_REGISTRY', 'train_ae']
