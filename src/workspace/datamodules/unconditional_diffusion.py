import torch
import numpy as np
from torch.utils.data import DataLoader, random_split, Dataset


class TurbulenceDataset(Dataset):
    """Dataset wrapper for turbulence data."""

    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def create_turbulence_loaders(
    data_path, batch_size=16, val_split=16, num_workers=16, seed=86, pin_memory=True
):
    """Create train/val dataloaders for turbulence data.

    Args:
        data_path: Path to .npy file with turbulence data
        batch_size: Training batch size
        val_split: Number of samples for validation split
        num_workers: DataLoader workers
        seed: Random seed for split
        pin_memory: Pin memory for faster GPU transfer

    Returns:
        (train_loader, val_loader, full_dataset)
    """
    dataset = torch.from_numpy(np.load(data_path, mmap_mode="r")[:-1, :, :, :]).clone()

    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = random_split(
        dataset, [len(dataset) - val_split, val_split], generator=generator
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=128,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, val_loader, dataset
