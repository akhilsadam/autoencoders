"""Backward-compatibility shim for the Lightning autoencoder."""
from __future__ import annotations

from .models.conv_autoencoder import AutoEncoderConfig, LitAutoEncoder, build_model

__all__ = ["AutoEncoderConfig", "LitAutoEncoder", "build_model"]
