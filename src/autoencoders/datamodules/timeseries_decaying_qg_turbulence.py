"""Data utilities for decaying QG turbulence time series."""
from __future__ import annotations

import os
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import List, Tuple
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from .load_timeseries_small import load_data


@dataclass
class TimeseriesDecayingQGTurbulenceConfig:
    """Configuration for decaying QG turbulence time series dataloaders."""
    root: str
    img_size: int = 256
    data_wavenumbers: List[float] = None
    dt: float = 0.001
    sim_steps: int = 100
    sim_time: float = 60.0
    sim_batch: int = 1
    batch_size: int = 32
    num_workers: int = 4
    train_memory_length: int = 20
    train_predict_length: int = 20
    test_memory_length: int = 20
    test_predict_length: int = 20
    downsample_factor: int = 1
    seed: int = 42
    version: int = 1

    def __post_init__(self):
        if self.data_wavenumbers is None:
            self.data_wavenumbers = [10.0, 32.0]


def _get_version_hash(cfg: TimeseriesDecayingQGTurbulenceConfig) -> str:
    """Generate hash of config parameters for versioning."""
    config_dict = asdict(cfg)
    # Exclude paths and non-physics parameters
    exclude_keys = {'root', 'batch_size', 'num_workers', 'seed', 'version'}
    physics_dict = {k: v for k, v in config_dict.items() if k not in exclude_keys}
    config_str = json.dumps(physics_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def build_dataloaders(cfg: TimeseriesDecayingQGTurbulenceConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test dataloaders for decaying QG turbulence."""
    # Create versioned cache directory
    version_hash = _get_version_hash(cfg)
    timestamp = datetime.now().strftime('%Y%m%d')
    version_dir = f"v{cfg.version}_{timestamp}_{version_hash}"
    cache_path = os.path.join(cfg.root, version_dir)
    os.makedirs(cache_path, exist_ok=True)
    
    # Save config for reproducibility
    config_path = os.path.join(cache_path, 'config.yaml')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            OmegaConf.save(cfg, f)
    
    save_path = os.path.join(cache_path, 'decaying_qg_turbulence_data.npy')

    def generate_data():
        from qg import config, QG

        _config = config()
        _config.logging.task_name = "data_generation_decaying_qg_turbulence"
        _config.logging.run_name = ""
        _config.forcing = None
        _config.grid.Nx = cfg.img_size
        _config.grid.Ny = cfg.img_size
        _config.ic.wavenumbers = cfg.data_wavenumbers
        _config.time.dt = cfg.dt
        _config.time.save_rate = cfg.sim_steps
        _config.time.T = cfg.sim_time
        _config.ic.n_batch = cfg.sim_batch
        
        qg = QG(_config)
        y = qg.solve(save_path=cache_path, name='decaying_qg_turbulence_data')[:,:,0:1,...]  # only vorticity
        return y
        
    if not os.path.exists(save_path):
        y = generate_data()
    else:
        y = torch.from_numpy(np.load(save_path)).to(torch.float32)[:,:,0:1,...]  # only vorticity
    
    y = y.to(torch.float32)
    nt = y.shape[1]
    data = y[:, :nt//2,...], y[:, nt//2:, ...], y[:, nt//2:,...]
    
    dataloader_kwargs = {
        'batch_size': cfg.batch_size,
        'num_workers': cfg.num_workers,
        'train_memory_length': cfg.train_memory_length,
        'train_predict_length': cfg.train_predict_length,
        'test_memory_length': cfg.test_memory_length,
        'test_predict_length': cfg.test_predict_length,
        'downsample_factor': cfg.downsample_factor,
    }
    
    loaders, shapes, datasets = load_data(data, **dataloader_kwargs)
    return loaders[0], loaders[1], loaders[2]  # train, val, test

