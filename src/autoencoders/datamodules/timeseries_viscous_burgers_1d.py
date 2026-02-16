"""Data utilities for 1D viscous Burgers equation time series."""
from __future__ import annotations

import os
import hashlib
import json
from dataclasses import dataclass, asdict
from typing import Tuple
from datetime import datetime

import numpy as np
import torch
from torch.utils.data import DataLoader
from omegaconf import OmegaConf

from .load_timeseries_small import load_data


@dataclass
class TimeseriesViscousBurgers1DConfig:
    """Configuration for viscous Burgers 1D time series dataloaders."""
    root: str
    Re: float = 1000.0
    T: float = 6.0
    batch_size: int = 32
    num_workers: int = 4
    train_memory_length: int = 20
    train_predict_length: int = 20
    test_memory_length: int = 20
    test_predict_length: int = 20
    downsample_factor: int = 1
    seed: int = 42
    version: int = 1


def _get_version_hash(cfg: TimeseriesViscousBurgers1DConfig) -> str:
    """Generate hash of config parameters for versioning."""
    config_dict = asdict(cfg)
    exclude_keys = {'root', 'batch_size', 'num_workers', 'seed', 'version'}
    physics_dict = {k: v for k, v in config_dict.items() if k not in exclude_keys}
    config_str = json.dumps(physics_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def build_dataloaders(cfg: TimeseriesViscousBurgers1DConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test dataloaders for viscous Burgers 1D."""
    from scipy.integrate import solve_ivp
    import matplotlib.pyplot as plt

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
    
    # Define the domain
    x = np.linspace(0, 1.0, 100)
    save_time = 0.01

    # Define the initial condition
    t0 = np.exp(cfg.Re/8)
    u0 = x / (1 + np.sqrt(1/t0) *(np.exp(cfg.Re * x**2 / 4)))

    # solve with RK45
    def f(t, u):
        du_dx = np.gradient(u, x, edge_order=2)
        d2u_d2x = np.gradient(du_dx, x, edge_order=2)
        du_dt = -u * du_dx + 1/cfg.Re * d2u_d2x
        return du_dt

    sol = solve_ivp(f, [0, cfg.T], u0, method='RK45', t_eval=np.arange(0, cfg.T, save_time), rtol=1e-6)

    y = sol.y.T
    
    # Save visualization
    plt.imshow(y, aspect='auto')
    plt.savefig(os.path.join(cache_path, 'viscous_burger_data.png'))
    plt.close()
    
    y = torch.from_numpy(y).to(torch.float32)
    nt = y.shape[0]
    data = y[None, :nt//3, None, :], y[None, nt//3:2*nt//3, None, :], y[None, 2*nt//3:, None, :]
    
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

