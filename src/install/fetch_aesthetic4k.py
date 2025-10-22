"""Fetch the real Aesthetic4K dataset into the local data directory.

This helper downloads or stages the official Aesthetic4K dataset so that
benchmarks run against the real images instead of a synthetic sample. It
supports three sourcing strategies:

1. `--source-dir` to copy an existing local dataset folder.
2. `--archive` / `--url` to extract from a tar/zip archive (local or remote).
3. `--repo-id` to snapshot a Hugging Face dataset repository.

For routes (2) and (3) the script will create a canonical folder structure
under the requested output directory. Existing contents are left untouched
unless `--force` is supplied.
"""
from __future__ import annotations

import argparse
import shutil
import tarfile
import tempfile
import zipfile
from pathlib import Path
from typing import Iterable, Optional
from urllib.parse import urlparse
from urllib.request import urlopen

DEFAULT_OUTPUT = Path("data/aesthetic4k")


def _parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Destination directory")
    parser.add_argument(
        "--source-dir",
        type=Path,
        help="Existing Aesthetic4K directory to copy instead of downloading",
    )
    parser.add_argument(
        "--archive",
        type=Path,
        help="Local archive (.zip/.tar[.gz|.bz2|.xz]) to extract",
    )
    parser.add_argument(
        "--url",
        type=str,
        help="Remote URL to download an archive from before extracting",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        default=None,
        help=(
            "Hugging Face dataset repo to snapshot (e.g. 'cafeai/aesthetic-visual-quality'). "
            "Requires the huggingface_hub package and valid authentication for gated repos."
        ),
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional Hugging Face revision (tag/commit) when using --repo-id",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output directory if it already exists",
    )
    return parser.parse_args(argv)


def _ensure_clean_dir(path: Path, force: bool) -> None:
    if path.exists():
        if not force:
            return
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def _copy_source_dir(source: Path, dest: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Source directory not found: {source}")
    shutil.copytree(source, dest, dirs_exist_ok=True)


def _download_url(url: str, dest_dir: Path) -> Path:
    parsed = urlparse(url)
    filename = Path(parsed.path).name or "aesthetic4k"
    archive_path = dest_dir / filename
    with urlopen(url) as response, archive_path.open("wb") as handle:  # type: ignore[attr-defined]
        shutil.copyfileobj(response, handle)
    return archive_path


def _extract_archive(archive_path: Path, dest_dir: Path) -> None:
    if zipfile.is_zipfile(archive_path):
        with zipfile.ZipFile(archive_path, "r") as zip_file:
            zip_file.extractall(dest_dir)
    elif tarfile.is_tarfile(archive_path):
        with tarfile.open(archive_path, "r:*") as tar:
            tar.extractall(dest_dir)
    else:
        raise ValueError(f"Unsupported archive format for {archive_path}")


def _download_huggingface(repo_id: str, revision: Optional[str], dest_dir: Path) -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "huggingface_hub is required to download from Hugging Face. Install it via "
            "`pip install huggingface_hub`."
        ) from exc

    snapshot_download(
        repo_id=repo_id,
        repo_type="dataset",
        revision=revision,
        local_dir=str(dest_dir),
        local_dir_use_symlinks=False,
    )


def main(argv: Optional[Iterable[str]] = None) -> int:
    args = _parse_args(argv)
    output_dir = args.output.expanduser().resolve()

    if args.repo_id:
        _ensure_clean_dir(output_dir, args.force)
        _download_huggingface(args.repo_id, args.revision, output_dir)
        return 0

    if args.source_dir:
        _ensure_clean_dir(output_dir, args.force)
        _copy_source_dir(args.source_dir.expanduser().resolve(), output_dir)
        return 0

    temp_dir = tempfile.mkdtemp(prefix="aesthetic4k-")
    temp_path = Path(temp_dir)
    try:
        archive = args.archive
        if args.url:
            archive = _download_url(args.url, temp_path)
        if not archive:
            raise ValueError("Provide --repo-id, --source-dir, --archive, or --url to acquire the dataset")

        archive = archive.expanduser().resolve()
        if not archive.exists():
            raise FileNotFoundError(f"Archive not found: {archive}")

        extract_dir = temp_path / "extracted"
        extract_dir.mkdir(parents=True, exist_ok=True)
        _extract_archive(archive, extract_dir)

        _ensure_clean_dir(output_dir, args.force)
        shutil.copytree(extract_dir, output_dir, dirs_exist_ok=True)
    finally:
        shutil.rmtree(temp_path, ignore_errors=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
