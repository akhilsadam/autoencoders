"""Path utilities shared across the autoencoder package."""
from __future__ import annotations

from pathlib import Path

from hydra.utils import get_original_cwd

_PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path_str: str) -> Path:
    """Resolve paths relative to Hydra's original cwd or the project root."""

    path = Path(path_str).expanduser()
    if path.is_absolute():
        return path

    try:
        hydra_root = Path(get_original_cwd())
    except ValueError:
        hydra_root = _PROJECT_ROOT

    return (hydra_root / path).resolve()
