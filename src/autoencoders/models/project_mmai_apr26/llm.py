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

from autoencoders.metrics import text as MX

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
        

        self.crpn = ContrastiveRPN(seq_len=config['seq_len'],
                                   embed_dim=config['embed_dim'],
                                   proj_dim=config['proj_dim'],
                                   rules=config['rules'],
                                   ae_type=config['ae_type'])
        
        self.proj_dim = config['proj_dim']
        self.learning_rate = config['learning_rate']
        
        self.tokenize = self.crpn.tokenize
        self.detokenize = self.crpn.detokenize

    def encode(self, rpns):
        tokens, amps = self.crpn.tokenize(rpns)
        tokens = tokens.to(self.crpn.device)
        amps = amps.to(self.crpn.device)
        pooled = self.crpn.encode_token_batch(tokens, amps)
        return pooled
    
    def decode(self, pooled):
        tokens, amps = self.crpn.decode(pooled)
        rpns = self.crpn.detokenize(tokens, amps)
        return rpns
    
    # ── Lightning ─────────────────────────────────────────────────────────

    def training_step(self, batch: torch.Tensor, _: int, logger=None) -> torch.Tensor:
        logger = logger or self
        loss, masked_supcon_loss, denoise_distortion_loss_tk, denoise_distortion_loss_sc, denoise_perception_loss, syntax_loss, rule_loss = self.crpn.loss(batch)
        logger.log_dict({
            'llm_masked_supcon_loss': masked_supcon_loss,
            'llm_denoise_distortion_loss_token': denoise_distortion_loss_tk,
            'llm_denoise_distortion_loss_scalar': denoise_distortion_loss_sc,
            'llm_denoise_perception_loss': denoise_perception_loss,
            'llm_syntax_loss': syntax_loss,
            'llm_rule_loss': rule_loss,
            'train_llm_loss': loss,
        }, batch_size=len(batch))
        logger.log('train_loss', loss, prog_bar=True, batch_size=len(batch))
        return loss

    def validation_step(self, batch: torch.Tensor, _: int, logger=None) -> None:
        logger = logger or self
        loss, masked_supcon_loss, denoise_distortion_loss_tk, denoise_distortion_loss_sc, denoise_perception_loss, syntax_loss, rule_loss = self.crpn.loss(batch)
        logger.log_dict({
            'val_llm_denoise_distortion_loss_token': denoise_distortion_loss_tk,
            'val_llm_denoise_distortion_loss_scalar': denoise_distortion_loss_sc,
            'val_llm_denoise_perception_loss': denoise_perception_loss,
            'val_llm_syntax_loss': syntax_loss,
            'val_llm_rule_loss': rule_loss,
            'val_llm_loss': loss,
        }, batch_size=len(batch))
        logger.log('val_loss', loss, prog_bar=True, batch_size=len(batch))
        
        
    def metrics(self, assistant):
        # pass
        val_loader = assistant #
        # MX.reconstruction(self, val_loader, dirs)
        MX.generation(self, val_loader, self.dirs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)