"""Generate a lightweight stand-in for the Aesthetic4K dataset.

This helper produces a deterministic image folder layout that mimics the
Aesthetic4K structure so benchmark tests can run in CI environments that
cannot access the full dataset. For real experiments, replace this script
with an authenticated downloader that pulls the official images.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import torch
from torchvision.utils import save_image

CLASSES = ("high_quality", "low_quality")
SPLITS = ("train", "val")


def _generate(split: str, label: str, count: int, output_dir: Path, image_size: int) -> None:
    class_dir = output_dir / split / label
    class_dir.mkdir(parents=True, exist_ok=True)

    for idx in range(count):
        seed = hash((split, label, idx)) & 0xFFFFFFFF
        torch.manual_seed(seed)
        image = torch.rand(3, image_size, image_size)
        save_image(image, class_dir / f"{idx:05d}.png")


def build_dataset(output_dir: Path, images_per_class: int, image_size: int) -> None:
    for split in SPLITS:
        count = images_per_class if split == "train" else max(1, images_per_class // 4)
        for label in CLASSES:
            _generate(split, label, count, output_dir, image_size)

    (output_dir / "README.txt").write_text(
        "Synthetic Aesthetic4K-style dataset generated for tests.\n",
        encoding="utf-8",
    )


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a synthetic Aesthetic4K sample dataset")
    parser.add_argument("--output", type=Path, default=Path("data/aesthetic4k"))
    parser.add_argument("--images-per-class", type=int, default=64)
    parser.add_argument("--image-size", type=int, default=256)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir: Path = args.output.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    build_dataset(output_dir, args.images_per_class, args.image_size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
