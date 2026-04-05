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
from tqdm import tqdm


class _TensorDatasetNoTuple(Dataset):
    """Wrapper that returns tensors directly without wrapping in tuples."""
    def __init__(self, tensor):
        self.tensor = tensor
    
    def __len__(self):
        return len(self.tensor)
    
    def __getitem__(self, idx):
        return self.tensor[idx]


@dataclass
class RPNTurbulenceConfig:
    """Configuration for forced QG turbulence dataloaders."""
    root: str
    Lx: float = 2 * np.pi
    Ly: float = 2 * np.pi
    grid_size: int = 128
    num_samples: int = 100
    time_steps: int = 200
    save_rate: int = 10
    dt: float = 0.001
    spinup_frames: int = 50  # Number of initial frames to skip
    
    # Physics parameters (from forced_turbulence scenario)
    mu: float = 0.02       # Linear drag
    nu: float = 1.025e-4   # Viscosity
    B: float = 1.0         # Beta plane
    ic_energy: float = 0.0
    ic_wavenumbers: Tuple[float, float] = (3.0, 5.0)
    
    # Forcing parameters
    forcing_A: float = -0.1
    forcing_B: float = 2.0
    forcing_C: float = 0.0
    forcing_D: float = 0.1
    forcing_E: float = 2.0
    forcing_F: float = 0.0
    
    batch_size: int = 32
    num_workers: int = 4
    val_split: int = 20
    seed: int = 42
    version: int = 1


def _get_version_hash(cfg: RPNTurbulenceConfig) -> str:
    """Generate hash of config parameters for versioning."""
    config_dict = asdict(cfg)
    # Exclude paths and non-physics parameters
    exclude_keys = {'root', 'batch_size', 'num_workers', 'seed', 'version', 'val_split'}
    physics_dict = {k: v for k, v in config_dict.items() if k not in exclude_keys}
    config_str = json.dumps(physics_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def build_dataloaders(cfg: RPNTurbulenceConfig) -> Tuple[DataLoader, DataLoader]:
    """Build train and validation dataloaders for forced turbulence."""
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
    
    save_path = lambda i : os.path.join(cache_path, f'rpn_turbulence_data_{i}.npy')
    
    print(f"Looking for cached data at: {save_path(0)}")
    print(f"Cache exists: {os.path.exists(save_path(0))}")
    
    n_datasets = 400
    
    def generate_data():
        """Generate rpn turbulence dataset using QG solver."""
        # Import only when generating (slow on network filesystem)
        from hydra import compose, initialize_config_dir
        from hydra.core.global_hydra import GlobalHydra
        from qg import QG
        
        # Initialize QG config from forced_turbulence scenario
        qg_config_dir = Path(__file__).parents[3] / "packages" / "qg" / "src" / "qg" / "conf"
        
        # Clear GlobalHydra if already initialized
        GlobalHydra.instance().clear()
        
        
        ### generate RPNs
        from qg.solver.opt.operator.rpn import batch_rpn_gen
        rpns = batch_rpn_gen(batch_size=n_datasets * 5, max_depth=5, max_nodes=20)
        # overgenerate and filter as needed?
        print(f"Generated RPNs: {rpns}")
        
        with open(os.path.join(cache_path, 'rpns.txt'), 'w') as f:   
            i = 0     
            for j, rpn in tqdm(enumerate(rpns)):   
                if i < n_datasets:     
                    try:
                        with initialize_config_dir(version_base='1.3', config_dir=str(qg_config_dir)):
                            qg_cfg = compose(config_name='config', overrides=['scenario=forced_turbulence'])
                            
                            # Override with our parameters
                            qg_cfg.qg.grid.Nx = cfg.grid_size
                            qg_cfg.qg.grid.Ny = cfg.grid_size
                            qg_cfg.qg.grid.Lx = cfg.Lx
                            qg_cfg.qg.grid.Ly = cfg.Ly
                            qg_cfg.qg.time.dt = cfg.dt
                            qg_cfg.qg.time.save_rate = cfg.save_rate
                            qg_cfg.qg.time.T = cfg.time_steps * cfg.dt
                            qg_cfg.qg.ic.n_batch = cfg.num_samples
                            qg_cfg.qg.ic.energy = cfg.ic_energy
                            qg_cfg.qg.ic.wavenumbers = list(cfg.ic_wavenumbers)
                            qg_cfg.qg.pde.mu = cfg.mu
                            qg_cfg.qg.pde.nu = cfg.nu
                            qg_cfg.qg.pde.B = cfg.B
                            qg_cfg.qg.forcing.A = cfg.forcing_A
                            qg_cfg.qg.forcing.B = cfg.forcing_B
                            qg_cfg.qg.forcing.C = cfg.forcing_C
                            qg_cfg.qg.forcing.D = cfg.forcing_D
                            qg_cfg.qg.forcing.E = cfg.forcing_E
                            qg_cfg.qg.forcing.F = cfg.forcing_F
                            
                            qg_cfg.qg.pde.rpn = rpn  # Pass RPN expression to config for use in solver
                            
                            # Don't pass logger - let QG use default
                            qg_solver = QG(qg_cfg.qg)
                            # Run simulation with visualization - saves videos and plots
                            result = qg_solver.solve(
                                save_path=cache_path,
                                name=f'rpn_turbulence_{i}',
                                clamp=0.3,
                                nan_check=True,
                                lim_check=100.0,
                            )
                            
                            variance = torch.mean(torch.var(result[:,:,0:1,...], dim=[-2,-1]))
                            if variance < 0.1:
                                print(f"Warning: Low variance ({variance:.2e}) in vorticity for RPN {j}: {rpn}")
                                raise ValueError("Low variance in generated data, skipping this RPN")
                            
                            # Skip spinup period and take all remaining frames
                            # result shape: [batch, time, channels, H, W]
                            vorticity = result[:, cfg.spinup_frames:, 0:1, :, :]  # [batch, time-spinup, 1, H, W]
                            
                            # Reshape to treat each timestep as a separate sample
                            # [batch, time-spinup, 1, H, W] -> [batch*(time-spinup), 1, H, W]
                            batch_size, n_frames, channels, height, width = vorticity.shape
                            vorticity = vorticity.reshape(batch_size * n_frames, channels, height, width)
                            
                            np.save(save_path(i), vorticity.cpu().numpy())
                            
                            f.write(rpn)
                            if i < n_datasets - 1:
                                f.write('\n')
                            i += 1
                            
                    except Exception as e:
                        print(f"Error generating data for RPN {j}: {rpn}")
                        print(f"Exception: {e}")
            
            # Re-initialize the original Hydra context after generating data
            GlobalHydra.instance().clear()
    
    # Load or generate data
    if not os.path.exists(save_path(0)):
        print("Generating new data...")
        generate_data()
    
    rpns = open(os.path.join(cache_path, 'rpns.txt')).read().splitlines()
    print(f"Loaded RPNs: {rpns}")
    n_datasets = len(rpns)
    print(f"Number of datasets: {n_datasets}")
    
    # TODO fix below to load all datasets and concatenate, currently just loads the first one for testing
    
    print(f"Loading cached data from {save_path(0)}...")
    file_size_mb = os.path.getsize(save_path(0)) / (1024**2)
    print(f"File size: {file_size_mb:.1f} MB")
    dataset_tensor = torch.from_numpy(np.load(save_path(0))).to(torch.float32)
    print(f"Data loaded: {dataset_tensor.shape}")

    # # Use memory mapping for large files to avoid loading entire file into RAM
    # print("Using memory-mapped array (mmap_mode='r')...")
    # data_np = np.load(save_path, mmap_mode='r')
    # print(f"Data shape: {data_np.shape}, dtype: {data_np.dtype}")
    
    # # Convert to torch tensor (this is lazy and won't load all data at once)
    # dataset_tensor = torch.from_numpy(data_np).float()
    # print(f"Tensor created (memory-mapped)")
    
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
