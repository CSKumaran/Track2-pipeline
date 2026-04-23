"""FFmpeg helpers for audio extraction and frame grabbing."""

from __future__ import annotations

import os
import subprocess
import logging

logger = logging.getLogger(__name__)


def extract_audio(video_path: str, output_dir: str,
                  sample_rate: int = 16000) -> str:
    """Extract mono WAV audio from *video_path*.

    Returns the path to the extracted WAV file.
    """
    out_path = os.path.join(output_dir, "audio.wav")
    if os.path.exists(out_path):
        logger.info("Audio already extracted: %s", out_path)
        return out_path

    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vn",                       # no video
        "-acodec", "pcm_s16le",      # 16-bit PCM
        "-ar", str(sample_rate),     # target sample rate
        "-ac", "1",                  # mono
        out_path,
    ]
    logger.info("Extracting audio: %s", " ".join(cmd))
    subprocess.run(cmd, check=True, capture_output=True)
    return out_path


def extract_frame_at_time(video_path: str, time_sec: float,
                          output_path: str) -> str:
    """Grab a single JPEG frame from *video_path* at *time_sec*.

    Returns *output_path*.
    """
    if os.path.exists(output_path):
        return output_path

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    cmd = [
        "ffmpeg", "-y",
        "-ss", f"{time_sec:.3f}",
        "-i", video_path,
        "-frames:v", "1",
        "-q:v", "2",
        output_path,
    ]
    result = subprocess.run(cmd, capture_output=True)
    if result.returncode != 0:
        logger.warning("ffmpeg frame extraction failed at t=%.3f: %s",
                       time_sec, result.stderr.decode("utf-8", errors="replace")[:200])
    return output_path
