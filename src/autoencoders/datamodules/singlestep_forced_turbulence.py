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
from .timeseries import TimeSeriesDataset


def build_dataloaders(cfg: ForcedTurbulenceConfig) -> Tuple[DataLoader, DataLoader]:
    dataset_tensor = torch.from_numpy(get_dataset(cfg, 'forced_turbulence.npy', mmap=True)[:, cfg.spinup_frames:, 0:1, :, :])
            
    # Split into train and validation
    val_split = cfg.val_split
    if val_split >= (dataset_tensor.shape[1]):
        raise ValueError("Validation split must be smaller than dataset size")

    _train_dataset = dataset_tensor[:, :-val_split]
    _val_dataset = dataset_tensor[:, -val_split:]
    
    train_dataset = TimeSeriesDataset(
        data=_train_dataset,
        seq_length=2,  # single-step prediction,
        stride=4
    )
    
    val_dataset = TimeSeriesDataset(
        data=_val_dataset,
        seq_length=2,  # single-step prediction
        stride=4
    )
    
    test_dataset = TimeSeriesDataset(
        data=_val_dataset,
        seq_length=16,  # single-step prediction
        stride=4
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
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    
    return train_loader, val_loader, test_loader