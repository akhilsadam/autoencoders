"""Dataset registry and helpers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Tuple, Type

from torch.utils.data import DataLoader

from .aesthetic4k import Aesthetic4KConfig, build_dataloaders as build_aesthetic4k
from .fashion_mnist import FashionMNISTConfig, build_dataloaders as build_fashion_mnist
from .qg_turbulence import QGDatasetConfig, build_dataloaders as build_qg_turbulence

# Time series datasets
from .timeseries_decaying_qg_turbulence import TimeseriesDecayingQGTurbulenceConfig, build_dataloaders as build_timeseries_decaying_qg_turbulence
from .timeseries_delay_2d import TimeseriesDelay2DConfig, build_dataloaders as build_timeseries_delay_2d
from .timeseries_viscous_burgers_1d import TimeseriesViscousBurgers1DConfig, build_dataloaders as build_timeseries_viscous_burgers_1d

# Single-image forced turbulence dataset
from .forced_turbulence import ForcedTurbulenceConfig, build_dataloaders as build_forced_turbulence
from .rpn_turbulence import RPNTurbulenceConfig, build_dataloaders as build_rpn_turbulence


from .load_timeseries_small import load_data, TimeSeriesDataset, Normalize

DataLoaderPair = Tuple[DataLoader, DataLoader]


@dataclass(frozen=True)
class DatasetEntry:
    config_cls: Type[Any]
    builder: Callable[[Any], DataLoaderPair]


DATASET_REGISTRY: Dict[str, DatasetEntry] = {
    "fashion_mnist": DatasetEntry(config_cls=FashionMNISTConfig, builder=build_fashion_mnist),
    "aesthetic4k": DatasetEntry(config_cls=Aesthetic4KConfig, builder=build_aesthetic4k),
    "qg_turbulence": DatasetEntry(config_cls=QGDatasetConfig, builder=build_qg_turbulence),
    "forced_turbulence": DatasetEntry(config_cls=ForcedTurbulenceConfig, builder=build_forced_turbulence),
    "rpn_turbulence": DatasetEntry(config_cls=RPNTurbulenceConfig, builder=build_rpn_turbulence),
    "timeseries_decaying_qg_turbulence": DatasetEntry(config_cls=TimeseriesDecayingQGTurbulenceConfig, builder=build_timeseries_decaying_qg_turbulence),
    "timeseries_delay_2d": DatasetEntry(config_cls=TimeseriesDelay2DConfig, builder=build_timeseries_delay_2d),
    "timeseries_viscous_burgers_1d": DatasetEntry(config_cls=TimeseriesViscousBurgers1DConfig, builder=build_timeseries_viscous_burgers_1d),
}


def list_datasets() -> Tuple[str, ...]:
    return tuple(DATASET_REGISTRY.keys())