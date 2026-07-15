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

def _get_version_hash(cfg, extra_keys=[]) -> str:
    """Generate hash of config parameters for versioning."""
    config_dict = asdict(cfg) if not isinstance(cfg, dict) else cfg
    # Exclude paths and non-physics parameters
    exclude_keys = ['root', 'batch_size', 'num_workers', 'test_workers', 'seed', 'version', 'val_split', *extra_keys]
    physics_dict = {k: v for k, v in config_dict.items() if k not in exclude_keys}
    config_str = json.dumps(physics_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]

def get_cache(cfg, extra_keys=[]) -> str:
    version_hash = _get_version_hash(cfg, extra_keys=extra_keys)
    timestamp = datetime.now().strftime('%Y%m%d')
    
    cache_path = None
    os.makedirs(cfg.root, exist_ok=True)
    for folder in os.listdir(cfg.root):
        if folder.startswith(f"v{cfg.version}") and folder.endswith(f"_{version_hash}"):
            version_dir = folder
            cache_path = os.path.join(cfg.root, folder)
            print(f"Found existing cache directory: {cache_path}")
            break
    
    if cache_path is None:
        version_dir = f"v{cfg.version}_{timestamp}_{version_hash}"
        cache_path = os.path.join(cfg.root, version_dir)
        os.makedirs(cache_path, exist_ok=True)
    
    # Save config for reproducibility
    config_path = os.path.join(cache_path, 'config.yaml')
    if not os.path.exists(config_path):
        with open(config_path, 'w') as f:
            OmegaConf.save(cfg, f)
    return cache_path
