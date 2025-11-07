"""Convolutional autoencoder Lightning module and configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import pytorch_lightning as pl
import torch
from torch import nn

# Convention: model class is named 'Autoencoder' or endswith 'Autoencoder', config is 'Config' or endswith 'Config'

import helion
from helion._testing import run_example
import helion.language as hl

@dataclass
class Config:
    """Default hyperparameters for autoencoder."""
    latent_dim: int = 32
    learning_rate: float = 1e-3

@helion.kernel(autotune_effort="quick")
def h_plus(x: torch.Tensor) -> torch.Tensor:
    _b, _c, h, _w = x.size()
    out = torch.empty_like(x)
    for tile_h in hl.tile(h):
        out[:, :, tile_h, :] = (x[:, :, tile_h, :]) + 5
    return out

@helion.kernel(autotune_effort="quick")
def h_plus_grad(x: torch.Tensor) -> torch.Tensor:
    _b, _c, h, _w = x.size()
    out = torch.empty_like(x)
    for tile_h in hl.tile(h):
        out[:, :, tile_h, :] = x[:, :, tile_h, :]
    return out


class h_function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input):
        # ctx.save_for_backward(input)
        return h_plus(input)

    @staticmethod
    def backward(ctx, grad_output):
        # input, = ctx.saved_tensors
        grad_input = h_plus_grad(grad_output)
        return grad_input

class CUAutoencoder(pl.LightningModule):
    """Tiny convolutional autoencoder for 28x28 grayscale images."""
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        self.save_hyperparameters(config)
    
        latent_dim = config['latent_dim']
        self.learning_rate = config['learning_rate']

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
        z = h_function.apply(z)
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
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)