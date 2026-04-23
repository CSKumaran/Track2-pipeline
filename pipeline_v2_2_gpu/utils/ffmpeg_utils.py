"""Audio and frame extraction via ffmpeg."""

import subprocess
import os
import logging

logger = logging.getLogger(__name__)


def extract_audio(video_path: str, output_dir: str, sr: int = 16000) -> str:
    """Extract 16kHz mono WAV from video."""
    os.makedirs(output_dir, exist_ok=True)
    wav_path = os.path.join(output_dir, "audio.wav")
    if os.path.exists(wav_path):
        logger.info("Audio already extracted: %s", wav_path)
        return wav_path
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", str(sr), "-vn",
        wav_path
    ]
    logger.info("Extracting audio: %s", " ".join(cmd))
    subprocess.run(cmd, capture_output=True, check=True)
    return wav_path


def extract_frames(video_path: str, output_dir: str, interval: float = 0.5) -> list:
    """Extract frames at given interval (seconds). Returns list of (time, path)."""
    frames_dir = os.path.join(output_dir, "sampled_frames")
    os.makedirs(frames_dir, exist_ok=True)

    # Check if already extracted
    existing = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    if existing:
        result = []
        for f in existing:
            t = float(f.replace("frame_", "").replace(".jpg", ""))
            result.append((t, os.path.join(frames_dir, f)))
        logger.info("Frames already extracted: %d frames", len(result))
        return result

    # Get video duration
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, check=True
    )
    duration = float(probe.stdout.strip())

    fps = 1.0 / interval
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        os.path.join(frames_dir, "frame_%010.4f.jpg")
    ]

    # ffmpeg doesn't support decimal in filename pattern easily, use sequential
    # then rename based on timestamp
    seq_pattern = os.path.join(frames_dir, "frame_%06d.jpg")
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-vf", f"fps={fps}",
        "-q:v", "2",
        seq_pattern
    ]
    logger.info("Extracting frames at %.2f fps", fps)
    subprocess.run(cmd, capture_output=True, check=True)

    # Rename to timestamp-based names
    result = []
    seq_files = sorted([f for f in os.listdir(frames_dir) if f.startswith("frame_")])
    for i, f in enumerate(seq_files):
        t = i * interval
        if t > duration + interval:
            break
        new_name = f"frame_{t:010.4f}.jpg"
        old_path = os.path.join(frames_dir, f)
        new_path = os.path.join(frames_dir, new_name)
        if old_path != new_path:
            os.rename(old_path, new_path)
        result.append((t, new_path))

    logger.info("Extracted %d frames over %.1fs", len(result), duration)
    return result


def get_video_duration(video_path: str) -> float:
    """Get video duration in seconds."""
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", video_path],
        capture_output=True, text=True, check=True
    )
    return float(probe.stdout.strip())


def extract_keyframe(video_path: str, timestamp: float, output_path: str) -> str:
    """Extract a single frame at a given timestamp."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        return output_path
    cmd = [
        "ffmpeg", "-y", "-ss", str(timestamp),
        "-i", video_path, "-frames:v", "1",
        "-q:v", "2", output_path
    ]
    subprocess.run(cmd, capture_output=True, check=True)
    return output_path
