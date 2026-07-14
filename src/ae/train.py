"""AE training entrypoint."""
from omegaconf import DictConfig


def train_ae(cfg: DictConfig) -> None:
    """Train autoencoder.

    Args:
        cfg: Hydra config with model, data, trainer settings
    """
    raise NotImplementedError("AE training not yet implemented in Phase 7")
