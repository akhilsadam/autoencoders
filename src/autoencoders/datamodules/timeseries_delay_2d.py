"""Data utilities for 2D delay differential equation time series."""
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
from tqdm import tqdm
from omegaconf import OmegaConf

from .load_timeseries_small import load_data


@dataclass
class TimeseriesDelay2DConfig:
    """Configuration for 2D delay time series dataloaders."""
    root: str
    w: int = 256
    T: float = 25.0
    R: float = 0.8
    c: float = 0.2
    batch_size: int = 32
    num_workers: int = 4
    train_memory_length: int = 20
    train_predict_length: int = 20
    test_memory_length: int = 20
    test_predict_length: int = 20
    downsample_factor: int = 1
    seed: int = 42
    version: int = 1


def _get_version_hash(cfg: TimeseriesDelay2DConfig) -> str:
    """Generate hash of config parameters for versioning."""
    config_dict = asdict(cfg)
    exclude_keys = {'root', 'batch_size', 'num_workers', 'seed', 'version'}
    physics_dict = {k: v for k, v in config_dict.items() if k not in exclude_keys}
    config_str = json.dumps(physics_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def build_dataloaders(cfg: TimeseriesDelay2DConfig) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Build train, validation, and test dataloaders for 2D delay system."""
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
    
    delay_2d_path = os.path.join(cache_path, 'delay_2d_data.npy')

    def generate_data():
        # Define the domain
        x = (torch.arange(-cfg.w//2, cfg.w//2)/cfg.w)[:,None]
        y = (torch.arange(-cfg.w//2, cfg.w//2)/cfg.w)[None,:]

        # Define the initial condition
        ux = 0.1*torch.sin(2 * torch.pi * x) * torch.sin(2 * torch.pi * y) * torch.exp(-0.5 * x**2)
        uy = 0.1*torch.cos(4 * torch.pi * x) * torch.cos(2 * torch.pi * y) * torch.exp(-0.5 * y**2)
        
        ux = ux[None, None, None, :, :]
        uy = uy[None, None, None, :, :]
        u0 = torch.cat([ux, uy], dim=2).repeat(1,2,1,1,1)

        @torch.compile
        def f(t, u_stack):
            u = u_stack[:,-1:,...]
            u_past = u_stack[:,-2:-1,...]
            
            du_dx, du_dy = torch.gradient(u*cfg.w, dim=(-2,-1), edge_order=2)
            rot_u_past = torch.stack([u_past[:,:,1], -u_past[:,:,0]], dim=2)
            
            du_dt = - cfg.c*(u[:,:,0:1].repeat(1,1,2,1,1) * du_dx) - cfg.c*(u[:,:,1:].repeat(1,1,2,1,1) * du_dy) \
                + cfg.R * rot_u_past
            return du_dt
        
        # euler timestepping
        save_time = 0.2
        dt = 0.002
        n_steps = int(cfg.T // save_time)
        n_save = int(save_time // dt)
        
        u_stack = u0
        
        for i in tqdm(range(n_steps)):
            u = u_stack[:,-2:]
            
            for j in range(n_save):
                du_dt = f(None, u)
                u_past = u[:,-1:]
                u = torch.cat([u_past, u_past + du_dt * dt],dim=1)
                
            u_stack = torch.cat([u_stack, u[:,-1:]], dim=1)

        y = u_stack
        np.save(delay_2d_path, y.cpu().numpy())
        return y
        
    if not os.path.exists(delay_2d_path):
        y = generate_data()
    else:
        y = torch.from_numpy(np.load(delay_2d_path)).to(torch.float32)
    
    y = y.to(torch.float32)
    nt = y.shape[1]
    data = y[:, :nt//3,...], y[:, nt//3:2*nt//3, ...], y[:, 2*nt//3:,...]
    
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

