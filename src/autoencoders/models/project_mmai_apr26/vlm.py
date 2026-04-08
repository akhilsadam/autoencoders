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

from . import vlm_diffusion, llm
from autoencoders.metrics import vlm_image_diffusion as MX

# Convention: model class ends with 'Diffusion', config is 'Config' or endswith 'Config'
@dataclass
class Config(vlm_diffusion.Config, llm.Config):
    pass

class OptVLMDiffusion(pl.LightningModule):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.save_hyperparameters(config)
        

        self.llm = llm.CRPNAutoencoder(config)
        self.opt = vlm_diffusion.Diffusion(config)
        
        self.proj_latent = nn.Sequential(
            nn.Linear(self.llm.proj_dim, 256),
            nn.LeakyReLU(),
            nn.Linear(256, self.opt.latent_dim),
            nn.Tanh(), # prevent explosion
        )
            
            # additive fusion doesn't work
        nn.init.zeros_(self.proj_latent[-1].weight)
        nn.init.zeros_(self.proj_latent[-1].bias)
        
        self.learning_rate = config['learning_rate']

    def compute_latent(self, rpns):
        return self.proj_latent(self.llm.encode(rpns))
    
    def gen(self, *args, **kwargs):
        return self.opt.gen(*args, **kwargs)
    # ── Lightning ─────────────────────────────────────────────────────────

    def training_step(self, batch, batch_id) -> torch.Tensor:
        rpn_batch, fused_batch = batch
        rpn_loss = self.llm.training_step(rpn_batch, batch_id, logger=self)
        
        rpns, seq = fused_batch
        x = seq[:,0]
        y = seq[:,1] # one timestep only
        latent = self.compute_latent(rpns)
        diffusion_loss = self.opt.loss(y, x, latent)
        self.log('diffusion_loss', diffusion_loss, prog_bar=True)
        
        loss = 0.1 * rpn_loss + diffusion_loss
        return loss

    def validation_step(self, batch, batch_id) -> None:
        rpn_batch, fused_batch = batch
        self.llm.validation_step(rpn_batch, batch_id, logger=self)
                
        rpns, seq = fused_batch
        x = seq[:,0]
        y = seq[:,1] # one timestep only
        latent = self.compute_latent(rpns)
        diffusion_loss = self.opt.loss(y, x, latent)
        self.log('val_diffusion_loss', diffusion_loss, prog_bar=True)
        
        MX.quick_reconstruction(self, rpns, seq, self.dirs, '', latent=latent)
            
    def metrics(self, assistant):
        # pass
        val_loader = assistant #
        MX.reconstruction(self, val_loader, dirs)
        # MX.generation(self, val_loader, dirs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)