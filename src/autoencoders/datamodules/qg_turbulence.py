"""QG turbulence dataset generator with automatic caching.

Generates datasets from QG (Quasi-Geostrophic) physics simulations and caches
them for fast reuse. Different configurations automatically create separate caches.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, TensorDataset
from omegaconf import OmegaConf

from mura.hydra import CacheManager


@dataclass
class QGDatasetConfig:
    """Configuration for QG-generated turbulence datasets."""
    
    batch_size: int
    num_workers: int = 4
    
    # QG simulation parameters
    grid_size: int = 128
    num_samples: int = 100  # Number of independent simulation batches
    time_steps: int = 200
    save_rate: int = 10  # Save every N timesteps
    dt: float = 1e-3
    
    # Physics parameters (QG solver)
    nu: float = 1.025e-5  # Viscosity
    mu: float = 0.0  # Linear drag
    beta: float = 0.0  # Beta plane
    
    # Caching
    cache_root: str = "${paths.data_root}/qg_cache"
    force_regenerate: bool = False
    
    # Train/val split
    val_split: float = 0.2
    
    seed: int = 42


def _generate_qg_data(config: dict, cache_path: Path) -> torch.Tensor:
    """Generate QG turbulence data using the qg package.
    
    Args:
        config: Configuration dictionary
        cache_path: Path to save artifacts
    
    Returns:
        Tensor of shape [B, T, C, H, W] where:
        - B: num_samples (number of independent simulations)
        - T: num_timesteps (time_steps // save_rate)
        - C: 4 (vorticity, u, v, streamfunction)
        - H, W: grid_size
    """
    try:
        from qg.solver.qg import QG
        from qg._input.validate_configuration import config as qg_config
    except ImportError:
        raise ImportError(
            "QG package not found. Install with: cd ../qg && uv pip install -e ."
        )
    
    print(f"Generating QG turbulence data with grid size {config['grid_size']}...")
    
    # Build QG configuration
    qg_cfg = qg_config()
    qg_cfg.grid.Nx = config['grid_size']
    qg_cfg.grid.Ny = config['grid_size']
    qg_cfg.time.dt = config['dt']
    qg_cfg.time.T = config['time_steps'] * config['dt']
    qg_cfg.time.save_rate = config['save_rate']
    qg_cfg.pde.nu = config['nu']
    qg_cfg.pde.mu = config['mu']
    qg_cfg.pde.B = config['beta']
    qg_cfg.seed = config['seed']
    
    # Initialize solver
    solver = QG(qg_cfg)
    
    # Generate data (this returns [B, T, C, H, W] tensor)
    data = solver._run()
    
    print(f"Generated data shape: {data.shape}")
    return data


def build_dataloaders(cfg: QGDatasetConfig) -> Tuple[DataLoader, DataLoader]:
    """Build train/val dataloaders from cached or generated QG data.
    
    Args:
        cfg: QG dataset configuration
    
    Returns:
        Tuple of (train_loader, val_loader)
    
    Example:
        In config.yaml:
            data:
              name: qg_turbulence
              params:
                batch_size: 32
                grid_size: 128
                num_samples: 100
    """
    # Resolve cache path (handle Hydra interpolations)
    cache_root = Path(OmegaConf.to_container(cfg.cache_root, resolve=True))
    
    # Create cache manager
    cache = CacheManager(cache_root=str(cache_root), version="v1")
    
    # Configuration dictionary for hashing (only include data-affecting params)
    config_dict = {
        'grid_size': cfg.grid_size,
        'num_samples': cfg.num_samples,
        'time_steps': cfg.time_steps,
        'save_rate': cfg.save_rate,
        'dt': cfg.dt,
        'nu': cfg.nu,
        'mu': cfg.mu,
        'beta': cfg.beta,
        'seed': cfg.seed,
    }
    
    # Load or generate data
    data = cache.load_or_generate(
        config=config_dict,
        generator_fn=_generate_qg_data,
        force_regenerate=cfg.force_regenerate,
        name="qg_turbulence",
    )
    
    # Data shape: [B, T, C, H, W]
    B, T, C, H, W = data.shape
    
    # Flatten batch and time dimensions: [B*T, C, H, W]
    data_flat = data.reshape(-1, C, H, W)
    
    # Split into train/val
    n_total = data_flat.shape[0]
    n_val = int(cfg.val_split * n_total)
    n_train = n_total - n_val
    
    # Use deterministic split
    generator = torch.Generator().manual_seed(cfg.seed)
    train_data, val_data = torch.utils.data.random_split(
        data_flat, [n_train, n_val], generator=generator
    )
    
    # Extract tensors from subsets
    train_tensor = torch.stack([train_data.dataset[i] for i in train_data.indices])
    val_tensor = torch.stack([val_data.dataset[i] for i in val_data.indices])
    
    # Create datasets (self-supervised: input = output for autoencoders)
    train_ds = TensorDataset(train_tensor, train_tensor)
    val_ds = TensorDataset(val_tensor, val_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    
    print(f"✅ QG dataset ready: {len(train_ds)} train, {len(val_ds)} val samples")
    print(f"   Data shape: C={C}, H={H}, W={W}")
    
    return train_loader, val_loader
