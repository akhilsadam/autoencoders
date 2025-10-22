"""Data utilities for FashionMNIST."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

from ..utils.paths import resolve_path


@dataclass
class FashionMNISTConfig:
    """Configuration for the FashionMNIST dataloaders."""

    root: str
    batch_size: int
    num_workers: int = 4
    val_split: int = 5000
    download: bool = True
    seed: int = 42


def build_dataloaders(cfg: FashionMNISTConfig) -> Tuple[DataLoader, DataLoader]:
    root = resolve_path(cfg.root)
    root.mkdir(parents=True, exist_ok=True)

    transform = transforms.Compose([transforms.ToTensor()])
    dataset = datasets.FashionMNIST(root=str(root), download=cfg.download, transform=transform)

    val_split = cfg.val_split
    if val_split >= len(dataset):
        raise ValueError("Validation split must be smaller than dataset size")

    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(
        dataset, [len(dataset) - val_split, val_split], generator=generator
    )

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
