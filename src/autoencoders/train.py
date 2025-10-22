"""Hydra-driven training entrypoint for autoencoders."""
from __future__ import annotations

import os
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torchvision.utils import save_image

from . import get_default_config, get_model
from .data import build_dataloaders
from .utils.paths import resolve_path
os.environ.setdefault("WANDB_MODE", "online")

from .trainer import create_trainer


def _prepare_model(cfg: DictConfig) -> pl.LightningModule:
    default_cfg = get_default_config(cfg.model.name)
    params: Dict[str, Any] = asdict(default_cfg)

    if cfg.model.get("params"):
        overrides = OmegaConf.to_container(cfg.model.params, resolve=True)
        params.update(overrides)  # type: ignore[arg-type]

    return get_model(cfg.model.name, params)


def _create_logger(cfg: DictConfig) -> WandbLogger:
    wandb_cfg = OmegaConf.to_container(cfg.wandb, resolve=True)
    kwargs = {k: v for k, v in wandb_cfg.items() if v not in (None, "")}
    logger = WandbLogger(**kwargs, log_model=False)
    logger.experiment.config.update(OmegaConf.to_container(cfg, resolve=True, enum_to_str=True))
    return logger


def _artifact_dirs(cfg: DictConfig) -> Dict[str, Path]:
    artifacts_root = resolve_path(cfg.paths.artifacts_root)
    checkpoints = artifacts_root / "checkpoints"
    reconstructions = artifacts_root / "reconstructions"
    for path in (artifacts_root, checkpoints, reconstructions):
        path.mkdir(parents=True, exist_ok=True)
    return {"root": artifacts_root, "checkpoints": checkpoints, "reconstructions": reconstructions}


def _save_reconstructions(model: pl.LightningModule, dataloader: torch.utils.data.DataLoader, output_dir: Path) -> None:
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        for batch in dataloader:
            inputs = batch[0].to(device)
            recon = model(inputs).cpu()
            save_image(inputs[:8], output_dir / "inputs.png", nrow=4)
            save_image(recon[:8], output_dir / "reconstructions.png", nrow=4)
            break


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    pl.seed_everything(cfg.seed)

    train_loader, val_loader = build_dataloaders(cfg.data)

    model = _prepare_model(cfg)
    logger = _create_logger(cfg)

    dirs = _artifact_dirs(cfg)
    checkpoint_cb = ModelCheckpoint(
        dirpath=str(dirs["checkpoints"]),
        filename="{epoch:02d}-{val_loss:.4f}",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        save_last=True,
        auto_insert_metric_name=False,
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    trainer = create_trainer(cfg.trainer, logger=logger, callbacks=[checkpoint_cb, lr_cb])

    trainer.fit(model, train_loader, val_loader)
    trainer.save_checkpoint(str(dirs["checkpoints"] / "last.ckpt"))

    _save_reconstructions(model, val_loader, dirs["reconstructions"])


if __name__ == "__main__":
    main()
