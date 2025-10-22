"""Dataset registry and helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, Type

from torch.utils.data import DataLoader

from .aesthetic4k import Aesthetic4KConfig, build_dataloaders as build_aesthetic4k
from .fashion_mnist import FashionMNISTConfig, build_dataloaders as build_fashion_mnist

DataLoaderPair = Tuple[DataLoader, DataLoader]


@dataclass(frozen=True)
class DatasetEntry:
    config_cls: Type[Any]
    builder: Callable[[Any], DataLoaderPair]


DATASET_REGISTRY: Dict[str, DatasetEntry] = {
    "fashion_mnist": DatasetEntry(config_cls=FashionMNISTConfig, builder=build_fashion_mnist),
    "aesthetic4k": DatasetEntry(config_cls=Aesthetic4KConfig, builder=build_aesthetic4k),
}


def list_datasets() -> Tuple[str, ...]:
    return tuple(DATASET_REGISTRY.keys())