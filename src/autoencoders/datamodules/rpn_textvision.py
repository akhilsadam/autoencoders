"""Data utilities for forced QG turbulence."""
from __future__ import annotations

import os
import hashlib
import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple
from datetime import datetime

import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split, TensorDataset
from omegaconf import OmegaConf

from .rpn_encoder import RPNEncoderConfig, build_dataloaders as build_enc
from .rpn_turbulence import RPNTurbulenceConfig, build_dataloaders as build_rpnt

class CombinedLoader:
    def __init__(self, loader1, loader2):
        self.loader1 = loader1
        self.loader2 = loader2

    def __iter__(self):
        for b1, b2 in zip(self.loader1, self.loader2):
            yield b1, b2

    def __len__(self):
        return min(len(self.loader1), len(self.loader2))

@dataclass
class RPNETConfig:
    text_config: str = 'rpn_encoder'
    vision_config: str = 'rpn_turbulence'
    
def build_dataloaders(cfg):
        
    print(cfg)
    train_rpn_loader, val_rpn_loader = build_enc(RPNEncoderConfig(**cfg.text_config['params']))
    train_paired_loader, val_paired_loader, pred_paired_loader = build_rpnt(RPNTurbulenceConfig(**cfg.vision_config['params']))

    train_loader = CombinedLoader(train_rpn_loader, train_paired_loader)
    val_loader = CombinedLoader(val_rpn_loader, val_paired_loader)
    
    return train_loader, val_loader, pred_paired_loader