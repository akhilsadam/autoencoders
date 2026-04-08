from __future__ import annotations

import os
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'
os.environ['TRITON_BACKEND'] = 'cuda'

from dataclasses import dataclass
from typing import Any, Dict

import pytorch_lightning as pl
import torch
from torch import nn
import torch.nn.functional as F

from . import operator_diffusion, llm

# Convention: model class ends with 'Diffusion', config is 'Config' or endswith 'Config'
@dataclass
class Config(operator_diffusion.Config, llm.Config):
    pass

class OptVLMDiffusion(pl.LightningModule):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        

        self.llm = llm.CRPNAutoencoder(config)
        self.opt = operator_diffusion.Diffusion(config)
        
        self.learning_rate = config['learning_rate']

    # ── Lightning ─────────────────────────────────────────────────────────

    def training_step(self, batch, step_id) -> torch.Tensor:
        rpn_batch, fused_batch = batch
        rpn_loss = self.llm.training_step(rpn_batch, step_id)
        
        rpns, images = fused_batch
        print(rpns, images.shape)
        
        
        loss = rpn_loss
        return loss

    def validation_step(self, batch, step_id) -> None:
        rpn_batch, fused_batch = batch
        rpn_loss = self.llm.validation_step(rpn_batch, step_id)
        
        
        
        
        loss = rpn_loss
        return loss
        
    def metrics(self, assistant, dirs):
        pass
        # val_loader = assistant #
        # MX.reconstruction(self, val_loader, dirs)
        # MX.generation(self, val_loader, dirs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)