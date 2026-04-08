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

from .cache import get_cache
from qg.solver.opt.operator.rpn import create_vocab_from_embeddings, RPNGenerator

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx]

@dataclass
class RPNEncoderConfig:
    """Configuration for forced QG turbulence dataloaders."""
    root: str
    
    seq_len: int = 100    
    n_rpn: int = 4000
    
    batch_size: int = 32
    num_workers: int = 4
    val_split: int = 20
    seed: int = 42
    version: int = 1

def get_dataset(cfg: RPNEncoderConfig, name = 'rpns.txt') -> torch.Tensor:
    """Build train and validation dataloaders for forced turbulence."""
    cache_path = get_cache(cfg)
    
    save_path = os.path.join(cache_path, name)    
    print(f"Looking for cached data at: {save_path}")
    print(f"Cache exists: {os.path.exists(save_path)}")
    
    def generate_data():
        # Create components
        vocab = create_vocab_from_embeddings()
        generator = RPNGenerator(vocab, max_depth=25, max_nodes=cfg.seq_len-2)
        
        # rules = create_composite_ruleset(TOKEN_TO_ID, pad_token_id=SCALAR_TOKEN_ID)
        # trainer = ContrastiveRPN(rules=rules)
        # trainer.eval()

        # Tokenize using new API
        # token_ids, amp = batch_tokenize_rpn(rpns, max_len=100)

        # # Forward through trainer
        # z = trainer(rpns)
        # rpns = RPNGenerator.generate_rpns(n_rpn=cfg.n_rpn, max_len=cfg.seq_len)

        # Generate batch
        rpns = generator.generate_batch(cfg.n_rpn)
        
        # Save to text file (one RPN per line)
        with open(save_path, 'w') as f:
            f.write('\n'.join(rpns))
            
        return rpns     
     
    
    # Load or generate data
    if not os.path.exists(save_path):
        print("Generating new data...")
        dataset_tensor = generate_data()
    else:
        print(f"Loading cached data from {save_path}...")
        file_size_mb = os.path.getsize(save_path) / (1024**2)
        print(f"File size: {file_size_mb:.1f} MB")
        dataset_tensor = np.loadtxt(save_path, dtype=str)
        print(f"Data loaded (rpns): {len(dataset_tensor)}")
        
    return dataset_tensor

def build_dataloaders(cfg: RPNEncoderConfig) -> Tuple[DataLoader, DataLoader]:
    dataset_tensor = get_dataset(cfg)
    print("Creating TextDataset...")
            
    # Split into train and validation
    val_split = cfg.val_split
    if val_split >= len(dataset_tensor):
        raise ValueError("Validation split must be smaller than dataset size")
    
    train_dataset = TextDataset(dataset_tensor[:-val_split])
    print(f"Train dataset size: {len(train_dataset)} samples")
    val_dataset = TextDataset(dataset_tensor[-val_split:])
    
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
