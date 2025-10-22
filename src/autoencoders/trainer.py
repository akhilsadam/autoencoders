"""Trainer factory to keep Lightning boilerplate out of model files."""
from __future__ import annotations

from typing import Iterable, Optional

import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf


def create_trainer(cfg: DictConfig, logger: Optional[pl.loggers.Logger] = None, callbacks: Optional[Iterable[pl.Callback]] = None) -> pl.Trainer:
    """Instantiate a Lightning Trainer using the provided Hydra config."""

    trainer_cfg = OmegaConf.to_container(cfg, resolve=True) if isinstance(cfg, DictConfig) else cfg
    trainer_kw = dict(trainer_cfg)

    if logger is not None:
        trainer_kw["logger"] = logger
    if callbacks is not None:
        trainer_kw["callbacks"] = list(callbacks)

    print('Trainer config:', trainer_kw)
    return pl.Trainer(**trainer_kw)