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

from qg.solver.opt.operator.rpn import ContrastiveRPN

# Convention: model class ends with 'Diffusion', config is 'Config' or endswith 'Config'
@dataclass
class Config:
    seq_len: int = 100
    embed_dim: int = 32
    proj_dim: int = 64
    rules: bool = False
    
    # Training
    learning_rate: float = 1e-4


class CRPNAutoencoder(pl.LightningModule):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        
        # rules = None if not rules else 
        rules = None # TODO add

        self.crpn = ContrastiveRPN(seq_len=config['seq_len'],
                                   embed_dim=config['embed_dim'],
                                   proj_dim=config['proj_dim'],
                                   rules=rules)
        
        self.learning_rate = config['learning_rate']

    # ── Lightning ─────────────────────────────────────────────────────────

    def training_step(self, batch: torch.Tensor, _: int) -> torch.Tensor:
        loss, denoise_loss, rule_loss = self.crpn.loss(batch)
        self.log_dict({
            'llm_denoise_loss': denoise_loss,
            'llm_rule_loss': rule_loss,
        })
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, _: int) -> None:
        loss, denoise_loss, rule_loss = self.crpn.loss(batch)
        self.log_dict({
            'val_llm_denoise_loss': denoise_loss,
            'val_llm_rule_loss': rule_loss,
        })
        self.log('val_loss', loss, prog_bar=True)
        
    def metrics(self, assistant, dirs):
        pass
        # val_loader = assistant #
        # MX.reconstruction(self, val_loader, dirs)
        # MX.generation(self, val_loader, dirs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)