"""Convolutional autoencoder Lightning module and configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
from torch import nn


@dataclass(frozen=True)
class AutoEncoderConfig:
    """Default hyperparameters for the convolutional autoencoder."""

    latent_dim: int = 32
    learning_rate: float = 1e-3


class LitAutoEncoder(pl.LightningModule):
    """Tiny convolutional autoencoder for 28x28 grayscale images."""

    def __init__(self, latent_dim: int = 32, lr: float = 1e-3) -> None:
        super().__init__()
        self.save_hyperparameters()

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, latent_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 32 * 7 * 7),
            nn.ReLU(),
            nn.Unflatten(1, (32, 7, 7)),
            nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid(),
        )
        self.criterion = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)

    def training_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _: int) -> torch.Tensor:
        x, _ = batch
        x_hat = self.forward(x)
        loss = self.criterion(x_hat, x)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch: Tuple[torch.Tensor, torch.Tensor], _: int) -> None:
        x, _ = batch
        x_hat = self.forward(x)
        val_loss = self.criterion(x_hat, x)
        self.log("val_loss", val_loss, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr)


DEFAULT_MODEL_CONFIG = AutoEncoderConfig()


def build_model(config: AutoEncoderConfig | Dict[str, Any] | None = None) -> LitAutoEncoder:
    """Construct the autoencoder using the provided configuration."""

    if config is None:
        config = DEFAULT_MODEL_CONFIG
    if isinstance(config, dict):
        config = AutoEncoderConfig(**config)
    if not isinstance(config, AutoEncoderConfig):
        raise TypeError("config must be AutoEncoderConfig, dict, or None")

    params = asdict(config)
    return LitAutoEncoder(latent_dim=params["latent_dim"], lr=params["learning_rate"])
