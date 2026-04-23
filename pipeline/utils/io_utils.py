"""I/O helpers: directory management, CSV writing, path construction."""

from __future__ import annotations

import os
import logging

import pandas as pd

logger = logging.getLogger(__name__)


def ensure_dir(path: str) -> str:
    """Create *path* (and parents) if it does not exist. Returns *path*."""
    os.makedirs(path, exist_ok=True)
    return path


def video_output_dir(output_root: str, video_path: str) -> str:
    """Return ``<output_root>/<video_stem>`` and ensure it exists."""
    stem = os.path.splitext(os.path.basename(video_path))[0]
    d = os.path.join(output_root, stem)
    return ensure_dir(d)


def safe_write_csv(df: pd.DataFrame, path: str) -> None:
    """Write *df* to CSV, creating parent dirs if needed."""
    ensure_dir(os.path.dirname(path))
    df.to_csv(path, index=False)
    logger.info("Wrote %d rows → %s", len(df), path)


def video_stem(video_path: str) -> str:
    """Return the filename without extension."""
    return os.path.splitext(os.path.basename(video_path))[0]
