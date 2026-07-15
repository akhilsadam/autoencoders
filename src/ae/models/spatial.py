"""Spatial autoencoder Lightning module and configuration."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from torch import nn

from ae.modules.ae import BasicSpatialAutoencoder

# Convention: model class is named 'Autoencoder' or endswith 'Autoencoder', config is 'Config' or endswith 'Config'

@dataclass
class Config:
    """Default hyperparameters for spatial autoencoder."""
    
    in_channels: int = 1
    lift_steps: int = 3
    encode_layers: int = 3
    patch_size: int = 16
    factor: int = 2
    learning_rate: float = 1e-3


class SpatialAutoencoder(pl.LightningModule):
    """Spatial autoencoder using pixel shuffling and SIREN layers."""
    
    def __init__(self, config: Config) -> None:
        super().__init__()
        
        self.save_hyperparameters(config)
        
        in_channels = config['in_channels']
        lift_steps = config['lift_steps']
        encode_layers = config['encode_layers']
        patch_size = config['patch_size']
        factor = config['factor']
        self.learning_rate = config['learning_rate']
        
        self.model = BasicSpatialAutoencoder(
            in_dim=in_channels,
            lift_steps=lift_steps,
            encode_layers=encode_layers,
            p=patch_size,
            factor=factor
        )
        
        self.criterion = nn.MSELoss()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def training_step(self, batch: torch.Tensor, _: int) -> torch.Tensor:
        x = batch[0]  # allows for possibly more (y)
        x_hat = self.forward(x)
        loss = self.criterion(x_hat, x) / self.criterion(x, torch.mean(x, dim=(-2, -1), keepdim=True))  # relative MSE
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    def validation_step(self, batch: torch.Tensor, _: int) -> None:
        x = batch[0]
        x_hat = self.forward(x)
        val_loss = self.criterion(x_hat, x) / self.criterion(x, torch.mean(x, dim=(-2, -1), keepdim=True))  # relative MSE
        self.log("val_loss", val_loss, prog_bar=True)
    
    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
