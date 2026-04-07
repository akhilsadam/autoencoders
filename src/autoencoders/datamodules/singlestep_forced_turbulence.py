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

from .forced_turbulence import ForcedTurbulenceConfig, get_dataset

class _TensorDatasetNoTuple(Dataset):
    """Wrapper that returns tensors directly without wrapping in tuples."""
    def __init__(self, tensor):
        self.tensor = tensor
    
    def __len__(self):
        return len(self.tensor)
    
    def __getitem__(self, idx):
        return self.tensor[idx]

def build_dataloaders(cfg: ForcedTurbulenceConfig) -> Tuple[DataLoader, DataLoader]:
    dataset_tensor = get_dataset(cfg)
    
    print("Creating TensorDataset...")
    dataset = TensorDataset(dataset_tensor)
    print(f"Dataset size: {len(dataset)} samples")
            
    # Split into train and validation
    val_split = cfg.val_split
    if val_split >= len(dataset):
        raise ValueError("Validation split must be smaller than dataset size")
    
    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(
        dataset, [len(dataset) - val_split, val_split], generator=generator
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader
