"""Benchmark tests for the autoencoder on the Aesthetic4K dataset."""
from __future__ import annotations

from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch
from hydra import compose, initialize_config_dir

from src.autoencoders import get_default_config, get_model
from src.autoencoders.data import build_dataloaders
from src.autoencoders.utils.paths import resolve_path

pytestmark = pytest.mark.benchmark


@pytest.mark.slow
@pytest.mark.parametrize("limit_train_batches", [2])
def test_autoencoder_aesthetic4k(limit_train_batches: int) -> None:
    config_dir = Path(__file__).resolve().parents[2] / "src/autoencoders/conf"

    with initialize_config_dir(version_base="1.3", config_dir=str(config_dir)):
        cfg = compose(
            config_name="config",
            overrides=[
                "data=aesthetic4k",
                "data.params.batch_size=16",
                "data.params.val_split=128",
            ],
        )

    try:
        train_loader, val_loader = build_dataloaders(cfg.data)
    except Exception as e:
        pytest.skip(f"Skipping benchmark: could not load HF dataset ({e}). Try 'huggingface-cli login'.")

    if len(train_loader.dataset) == 0 or len(val_loader.dataset) == 0:
        pytest.skip("Aesthetic4K dataset is empty after preprocessing.")

    default_cfg = get_default_config(cfg.model.name)
    model = get_model(cfg.model.name, default_cfg)

    pl.seed_everything(cfg.seed)
    trainer = pl.Trainer(
        max_epochs=1,
        accelerator="auto",
        devices=1,
        enable_model_summary=False,
        enable_progress_bar=False,
        logger=False,
        limit_train_batches=limit_train_batches,
        limit_val_batches=limit_train_batches,
    )

    trainer.fit(model, train_loader, val_loader)
    metrics = trainer.validate(model, val_loader)
    val_loss = metrics[0]["val_loss"]

    assert torch.isfinite(torch.tensor(val_loss)), "Validation loss returned non-finite value"
    assert val_loss < 1.0, f"Validation loss too high: {val_loss}"
