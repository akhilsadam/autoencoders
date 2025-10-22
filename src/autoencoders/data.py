"""Dataset registry entry point."""
from __future__ import annotations

from typing import Any, Dict

from omegaconf import DictConfig

from .datamodules import DATASET_REGISTRY, DataLoaderPair, list_datasets
from .utils.paths import resolve_path


def _to_plain_dict(cfg: DictConfig | Dict[str, Any]) -> Dict[str, Any]:
    if isinstance(cfg, DictConfig):
        return DictConfig.to_container(cfg, resolve=True)  # type: ignore[return-value]
    return dict(cfg)


def build_dataloaders(cfg: DictConfig | Dict[str, Any]) -> DataLoaderPair:
    """Instantiate dataloaders for the dataset described by the config."""

    cfg_dict = _to_plain_dict(cfg)
    dataset_name = cfg_dict.get("name")
    if not dataset_name:
        raise KeyError("Dataset config must include a 'name' field")

    entry = DATASET_REGISTRY.get(dataset_name)
    if entry is None:
        available = ", ".join(list_datasets())
        raise KeyError(f"Unknown dataset '{dataset_name}'. Available: {available}")

    params = cfg_dict.get("params", {}) or {}
    config_obj = entry.config_cls(**params)  # type: ignore[arg-type]
    return entry.builder(config_obj)


__all__ = ["build_dataloaders", "list_datasets", "resolve_path"]
