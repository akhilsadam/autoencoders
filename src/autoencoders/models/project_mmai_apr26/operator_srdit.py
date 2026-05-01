"""Spatial latent diffusion model following Kaiming He's 'Just Denoise' approach."""
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

from autoencoders.models.modules.ae import BasicSpatialAutoencoder
from autoencoders.models.modules.siren import Siren
from autoencoders.models.modules.math.derivative import Derivative
from autoencoders.models.modules.diffusion.samplers.flow_matching import samplers
from autoencoders.models.modules.diffusion.embeddings import TimeEmbedding
from autoencoders.models.modules.act import Tri

from autoencoders.metrics import conditional_image_diffusion as MX

from autoencoders.models.external.speedrun_dit import LoRASpeedrunDiT

# ONLY SINGLE STEP right now
# Convention: model class ends with 'Diffusion', config is 'Config' or endswith 'Config'
@dataclass
class Config:
    # Data
    in_channels: int = 1
    condition_channels: int = 1
    shape: tuple = (512, 512)
    
    # AE architecture
    k: int = 8                    # pixel shuffle factor
    sdim: int = 2                 # spatial coord dims
    tdim: int = 16                # time embedding dims
    siren_width: int = None       # defaults to in_dim if None
    siren_layers: int = 5
    siren_w: float = 5.0
    encode_layers: int = 3
    
    # Sampler / schedule
    steps: int = 25
    sampler: str = 'AB2CN'        # 'Euler', 'AB2', 'AB2CN'
    L: float = 1.0                # domain half-length for coords / deriv
    
    # Training
    learning_rate: float = 1e-4


class Diffusion(pl.LightningModule):

    def __init__(self, config: Config) -> None:
        super().__init__()
        self.save_hyperparameters(config)

        dim = config['in_channels']
        cdim = config['condition_channels']
        self.cdim = cdim
        
        self.shape = config['shape']
        self.L = config['L']
        self.learning_rate = config['learning_rate']

        # self.k = k
        # self.shuffle = nn.PixelShuffle(k)
        # self.unshuffle = nn.PixelUnshuffle(k)

        # in_dim = (sdim + tdim + dim + cdim) * k ** 2
        # out_dim = dim * k ** 2
        # width = config['siren_width'] if config['siren_width'] is not None else in_dim

        self.model = LoRASpeedrunDiT(resolution=config['shape'][0])
        # self.ae = BasicSpatialAutoencoder(dim, 0, config['encode_layers'])
        # self.ae2 = BasicSpatialAutoencoder(dim, 0, config['encode_layers'])
        
        # self.t_emb = TimeEmbedding(tdim)

        self.steps = config['steps']
        self._init_buffers(self.shape, self.L)
        self.criterion = lambda x_hat, x: ((x_hat - x).pow(2).mean() / ((x - x.mean(dim=(-2,-1),keepdim=True)).pow(2).mean() + 1e-8))

        self.sampler = samplers[config['sampler']]()

    def _init_buffers(self, shape: tuple, L: float) -> None:
        shape_sm = tuple(s // self.k for s in shape)
        x  = torch.frac(torch.linspace(-L, L, shape_sm[-1]))
        x  = x[:, None].expand(*shape_sm)
        xy = torch.stack([x, x.mT], dim=0)[None, ...]
        self.register_buffer('xy', xy)
        self.deriv = Derivative(shape=shape_sm,
                                L=tuple(L for _ in range(len(shape))))
        self.register_buffer('t',  torch.linspace(0, 1, self.steps + 1))
        self.register_buffer('dt', self.t[1:] - self.t[:-1])

    # ── Core primitives ───────────────────────────────────────────────────

    def denoise(self, x: torch.Tensor, t: torch.Tensor, c: torch.Tensor = None) -> torch.Tensor:
        return self.model.denoise(x, t, c, None)

    def noise(self, x: torch.Tensor) -> torch.Tensor:
        return torch.randn_like(x)

    def mix(self, x: torch.Tensor, n: torch.Tensor,
            t: torch.Tensor) -> torch.Tensor:
        return x * t + n * (1 - t)

    def vel(self, x_pred: torch.Tensor, x_n: torch.Tensor,
            t: torch.Tensor) -> torch.Tensor:
        return (x_pred - x_n) / (1 - t)

    def loss(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        
        if x.shape != self.shape:
            self.shape = x.shape[-2:]
            self.L = x.shape[-1] * self.L / self.shape[-1]
            self._init_buffers(self.shape, self.L)
            self.to(x.device)
        
        t      = torch.rand(x.shape[0], device=x.device)[:, None, None, None]
        n      = self.noise(x)
        x_n    = self.mix(x, n, t)
        v_pred = self.vel(self.denoise(x_n, t, c=c), x_n, t) * (1 - t)
        v_true = self.vel(x,                    x_n, t) * (1 - t)
        return self.criterion(v_pred, v_true)

    # ── Generation ────────────────────────────────────────────────────────

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = torch.tensor(0.1, device=x.device)
        n = self.noise(x)
        return self.denoise(self.mix(x, n, t), t, c=n)

    def latent(self, x: torch.Tensor) -> torch.Tensor:
        return self.noise(x)

    def gen(self, x: torch.Tensor, c: torch.Tensor, shape=None, L=None) -> torch.Tensor:
        
        if shape is not None and L is not None:
            self._init_buffers(shape, L)
            self.to(x.device)
        
        self.sampler.reset()
        x_n = self.noise(x)
        for i in range(self.steps):
            x_n = self.sampler.step(self, x_n, i, self.t, self.dt, c=c)
        return x_n

    # ── Lightning ─────────────────────────────────────────────────────────

    def training_step(self, batch: torch.Tensor, _: int, logger=None) -> torch.Tensor:
        logger = logger or self
        loss = self.loss(batch[:, 1], batch[:, 0]) 
        logger.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch: torch.Tensor, batch_id: int, logger=None) -> None:
        logger = logger or self
        logger.log('val_loss', self.loss(batch[:, 1], batch[:, 0]), prog_bar=True)
        
        MX.quick_reconstruction(self, batch, self.dirs, '')
        
    def metrics(self, assistant):
        val_loader = assistant #
        MX.reconstruction(self, val_loader, self.dirs)
        # MX.generation(self, val_loader, self.dirs)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)