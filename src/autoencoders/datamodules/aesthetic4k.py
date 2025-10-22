"""Dataloaders for the Aesthetic4K image quality dataset via Hugging Face datasets."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


@dataclass
class Aesthetic4KConfig:
    """Configuration for loading the Aesthetic4K dataset from HF datasets."""

    batch_size: int
    num_workers: int = 8
    val_split: int = 512  # number of samples for validation if no split exists
    image_size: int = 256
    normalize: bool = True
    seed: int = 42
    repo_id: str = "zhang0jhon/Aesthetic-4K"


def _build_transform(cfg: Aesthetic4KConfig) -> transforms.Compose:
    pipeline = [transforms.Resize((cfg.image_size, cfg.image_size)), transforms.ToTensor()]
    if cfg.normalize:
        pipeline.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    return transforms.Compose(pipeline)


class HFDatasetWrapper(Dataset):
    def __init__(self, hf_ds, transform: transforms.Compose) -> None:
        self.ds = hf_ds
        self.transform = transform

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.ds)

    def __getitem__(self, idx: int):  # type: ignore[override]
        item = self.ds[idx]
        # 'image' feature returns a PIL Image object
        img = item["image"]
        x = self.transform(img) if self.transform else img
        y = item["label"] if "label" in item else 0
        return x, y


def build_dataloaders(cfg: Aesthetic4KConfig) -> Tuple[DataLoader, DataLoader]:
    transform = _build_transform(cfg)

    ds_dict = load_dataset(cfg.repo_id)

    if "train" in ds_dict and ("validation" in ds_dict or "val" in ds_dict):
        train_hf = ds_dict["train"]
        val_hf = ds_dict.get("validation", ds_dict.get("val"))
    elif "train" in ds_dict:
        # Create a validation split from the training set
        train_hf = ds_dict["train"]
        n = len(train_hf)
        val_size = cfg.val_split if cfg.val_split < n else max(1, n // 10)
        split = train_hf.train_test_split(test_size=val_size, seed=cfg.seed)
        train_hf, val_hf = split["train"], split["test"]
    else:
        # Fallback: if only a single split exists (e.g., 'default')
        first_key = next(iter(ds_dict.keys()))
        base = ds_dict[first_key]
        n = len(base)
        val_size = cfg.val_split if cfg.val_split < n else max(1, n // 10)
        split = base.train_test_split(test_size=val_size, seed=cfg.seed)
        train_hf, val_hf = split["train"], split["test"]

    train_ds = HFDatasetWrapper(train_hf, transform)
    val_ds = HFDatasetWrapper(val_hf, transform)

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
