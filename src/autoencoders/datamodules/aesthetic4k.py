"""Dataloaders for the Aesthetic4K image quality dataset."""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms

from ..utils.paths import resolve_path


@dataclass
class Aesthetic4KConfig:
    """Configuration for loading the Aesthetic4K dataset."""

    root: str
    batch_size: int
    num_workers: int = 8
    val_split: int = 512
    split: str = "train"
    image_size: int = 256
    normalize: bool = True
    seed: int = 42
    use_metadata: bool = False
    metadata_filename: str = "metadata.csv"
    filepath_column: str = "filepath"
    label_column: str = "label"
    split_column: str = "split"


class MetadataImageDataset(Dataset[Tuple[torch.Tensor, int]]):
    """Dataset that uses a metadata CSV to resolve file paths and labels."""

    def __init__(
        self,
        base_dir: Path,
        records: Iterable[Dict[str, str]],
        label_column: str,
        filepath_column: str,
        transform: transforms.Compose,
    ) -> None:
        self.base_dir = base_dir
        self.transform = transform

        labels = sorted({row[label_column] for row in records})
        self._label_to_idx = {label: idx for idx, label in enumerate(labels)}

        self.samples: List[Tuple[Path, int]] = []
        for row in records:
            rel_path = row[filepath_column]
            resolved = (base_dir / rel_path).resolve()
            if not resolved.exists():
                raise FileNotFoundError(f"Image referenced in metadata not found: {resolved}")
            label_idx = self._label_to_idx[row[label_column]]
            self.samples.append((resolved, label_idx))

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:  # type: ignore[override]
        image_path, label = self.samples[index]
        with Image.open(image_path) as img:
            image = img.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label


def _metadata_records(cfg: Aesthetic4KConfig, root: Path) -> List[Dict[str, str]]:
    metadata_path = root / cfg.metadata_filename
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {cfg.filepath_column, cfg.label_column}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Metadata missing required columns: {sorted(missing)}")

        split_column_present = cfg.split_column in (reader.fieldnames or [])
        records = []
        for row in reader:
            if split_column_present and row.get(cfg.split_column, cfg.split) != cfg.split:
                continue
            records.append(row)
    if not records:
        raise ValueError(f"No records found for split '{cfg.split}' in metadata {metadata_path}")
    return records


def _build_transform(cfg: Aesthetic4KConfig) -> transforms.Compose:
    pipeline = [transforms.Resize((cfg.image_size, cfg.image_size)), transforms.ToTensor()]
    if cfg.normalize:
        pipeline.append(
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        )
    return transforms.Compose(pipeline)


def _build_imagefolder_dataset(cfg: Aesthetic4KConfig, root: Path, transform: transforms.Compose) -> Dataset:
    candidates = [root / cfg.split, root / cfg.split.capitalize(), root]
    for candidate in candidates:
        if candidate.exists() and any(child.is_dir() for child in candidate.iterdir()):
            dataset_root = candidate
            break
    else:
        raise FileNotFoundError(
            f"Could not locate class folders under {root}. Expected directories for ImageFolder."
        )
    return datasets.ImageFolder(root=str(dataset_root), transform=transform)


def build_dataloaders(cfg: Aesthetic4KConfig) -> Tuple[DataLoader, DataLoader]:
    root = resolve_path(cfg.root)
    if not root.exists():
        raise FileNotFoundError(
            f"Aesthetic4K root not found at {root}. Use download script to retrieve the dataset."
        )

    transform = _build_transform(cfg)

    if cfg.use_metadata:
        records = _metadata_records(cfg, root)
        dataset = MetadataImageDataset(
            root,
            records,
            label_column=cfg.label_column,
            filepath_column=cfg.filepath_column,
            transform=transform,
        )
    else:
        dataset = _build_imagefolder_dataset(cfg, root, transform)

    if len(dataset) <= cfg.val_split:
        raise ValueError("Validation split must be smaller than dataset size")

    generator = torch.Generator().manual_seed(cfg.seed)
    train_dataset, val_dataset = random_split(
        dataset,
        [len(dataset) - cfg.val_split, cfg.val_split],
        generator=generator,
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
