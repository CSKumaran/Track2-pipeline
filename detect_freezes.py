#!/usr/bin/env python3
"""
Detect freeze-frame segments in delayed video variants.

Compares consecutive frames within each video to find segments where
frames are repeated (frozen). Works by piping grayscale frames from
ffmpeg and computing mean absolute difference between consecutive frames.

Usage:
    python detect_freezes.py A1.mp4 A3.mp4 A5.mp4 -o outputs/freeze_detection/
    python detect_freezes.py A1.mp4 A3.mp4 A5.mp4 --baseline A0.mp4 -o outputs/freeze_detection/
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import numpy as np

# Frame extraction resolution (downscaled for speed)
FRAME_W = 320
FRAME_H = 240
FRAME_BYTES = FRAME_W * FRAME_H


def get_video_info(path):
    """Get fps, frame count, and duration using ffprobe."""
    cmd = [
        "ffprobe", "-v", "quiet",
        "-print_format", "json",
        "-show_streams", "-show_format",
        path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    fps_num, fps_den = map(int, stream["r_frame_rate"].split("/"))
    fps = fps_num / fps_den
    nb_frames = int(stream.get("nb_frames", 0))
    duration = float(info["format"]["duration"])
    return fps, nb_frames, duration


def detect_freezes_in_video(video_path, mad_threshold=0.5, min_duration_sec=0.5):
    """
    Detect freeze-frame segments by comparing consecutive frames.

    Returns list of dicts with freeze segment info.
    """
    fps, nb_frames, duration = get_video_info(video_path)
    print(f"  {os.path.basename(video_path)}: {nb_frames} frames, {fps:.1f} fps, {duration:.1f}s")

    cmd = [
        "ffmpeg", "-i", video_path,
        "-f", "rawvideo", "-pix_fmt", "gray",
        "-s", f"{FRAME_W}x{FRAME_H}",
        "-v", "quiet",
        "pipe:1"
    ]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    # Read first frame
    prev_frame_data = proc.stdout.read(FRAME_BYTES)
    if len(prev_frame_data) < FRAME_BYTES:
        print(f"  ERROR: Could not read first frame from {video_path}")
        proc.terminate()
        return []

    prev_frame = np.frombuffer(prev_frame_data, dtype=np.uint8).reshape(FRAME_H, FRAME_W)

    frozen_runs = []  # list of (start_frame, end_frame)
    current_run_start = None
    frame_idx = 0
    mad_values = []  # for diagnostics

    while True:
        frame_data = proc.stdout.read(FRAME_BYTES)
        if len(frame_data) < FRAME_BYTES:
            break
        frame_idx += 1
        curr_frame = np.frombuffer(frame_data, dtype=np.uint8).reshape(FRAME_H, FRAME_W)

        mad = np.mean(np.abs(curr_frame.astype(np.int16) - prev_frame.astype(np.int16)))

        if mad < mad_threshold:
            # Frame is frozen (identical to previous)
            if current_run_start is None:
                current_run_start = frame_idx - 1  # the first frame of the frozen pair
        else:
            # Frame changed
            if current_run_start is not None:
                frozen_runs.append((current_run_start, frame_idx - 1))
                current_run_start = None

        prev_frame = curr_frame

    # Close any open run
    if current_run_start is not None:
        frozen_runs.append((current_run_start, frame_idx))

    proc.terminate()
    proc.wait()

    # Filter by minimum duration and convert to segments
    min_frames = int(min_duration_sec * fps)
    segments = []
    for start_f, end_f in frozen_runs:
        n_frozen = end_f - start_f + 1
        if n_frozen >= min_frames:
            segments.append({
                "freeze_start_frame": start_f,
                "freeze_end_frame": end_f,
                "freeze_start_sec": round(start_f / fps, 3),
                "freeze_end_sec": round(end_f / fps, 3),
                "duration_sec": round(n_frozen / fps, 3),
                "frozen_frame_count": n_frozen,
            })

    return segments


def print_results(all_results):
    """Print a formatted table of results."""
    print("\n" + "=" * 80)
    print("FREEZE DETECTION RESULTS")
    print("=" * 80)

    for video_name, segments in all_results.items():
        print(f"\n--- {video_name} ---")
        if not segments:
            print("  No freeze segments detected.")
            continue

        total_freeze = sum(s["duration_sec"] for s in segments)
        print(f"  Found {len(segments)} freeze segment(s), total frozen: {total_freeze:.1f}s\n")
        print(f"  {'#':<4} {'Start':>8} {'End':>8} {'Duration':>10} {'Frames':>8}")
        print(f"  {'-'*4} {'-'*8} {'-'*8} {'-'*10} {'-'*8}")
        for i, seg in enumerate(segments, 1):
            print(f"  {i:<4} {seg['freeze_start_sec']:>7.1f}s {seg['freeze_end_sec']:>7.1f}s "
                  f"{seg['duration_sec']:>9.1f}s {seg['frozen_frame_count']:>8}")

    # Cross-video comparison
    if len(all_results) > 1:
        print(f"\n{'=' * 80}")
        print("CROSS-VIDEO COMPARISON")
        print("=" * 80)

        # Find common freeze points by clustering start times
        all_starts = {}
        for video_name, segments in all_results.items():
            for seg in segments:
                all_starts.setdefault(video_name, []).append(seg["freeze_start_sec"])

        # Show aligned table
        videos = list(all_results.keys())
        header = f"  {'Point':<8}"
        for v in videos:
            header += f" {v:>20}"
        print(f"\n{header}")
        print(f"  {'-' * (8 + 21 * len(videos))}")

        max_segs = max(len(segs) for segs in all_results.values()) if all_results else 0
        for i in range(max_segs):
            row = f"  {i+1:<8}"
            for v in videos:
                segs = all_results[v]
                if i < len(segs):
                    s = segs[i]
                    row += f" {s['freeze_start_sec']:>7.1f}s ({s['duration_sec']:.1f}s)"
                else:
                    row += f" {'---':>20}"
            print(row)


def save_results(all_results, output_dir):
    """Save results to CSV files."""
    os.makedirs(output_dir, exist_ok=True)

    # Detailed segments CSV
    segments_path = os.path.join(output_dir, "freeze_segments.csv")
    with open(segments_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "segment_id", "freeze_start_frame", "freeze_end_frame",
                         "freeze_start_sec", "freeze_end_sec", "duration_sec", "frozen_frame_count"])
        for video_name, segments in all_results.items():
            for i, seg in enumerate(segments, 1):
                writer.writerow([video_name, i, seg["freeze_start_frame"], seg["freeze_end_frame"],
                                 seg["freeze_start_sec"], seg["freeze_end_sec"],
                                 seg["duration_sec"], seg["frozen_frame_count"]])

    # Summary CSV
    summary_path = os.path.join(output_dir, "delay_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "num_freeze_points", "total_freeze_duration",
                         "freeze_timestamps", "freeze_durations"])
        for video_name, segments in all_results.items():
            timestamps = ";".join(f"{s['freeze_start_sec']:.1f}" for s in segments)
            durations = ";".join(f"{s['duration_sec']:.1f}" for s in segments)
            total = sum(s["duration_sec"] for s in segments)
            writer.writerow([video_name, len(segments), round(total, 1), timestamps, durations])

    print(f"\nResults saved to:")
    print(f"  {segments_path}")
    print(f"  {summary_path}")


def main():
    parser = argparse.ArgumentParser(description="Detect freeze-frame delays in video variants")
    parser.add_argument("videos", nargs="+", help="Delayed video files to analyze")
    parser.add_argument("-o", "--output", default="outputs/freeze_detection",
                        help="Output directory (default: outputs/freeze_detection)")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="MAD threshold for freeze detection (default: 0.5)")
    parser.add_argument("--min-duration", type=float, default=0.5,
                        help="Minimum freeze duration in seconds (default: 0.5)")
    args = parser.parse_args()

    print("Freeze-Frame Delay Detection")
    print("=" * 40)
    print(f"Threshold: MAD < {args.threshold}")
    print(f"Min duration: {args.min_duration}s")
    print(f"Resolution: {FRAME_W}x{FRAME_H} grayscale")
    print()

    all_results = {}
    for video_path in args.videos:
        if not os.path.exists(video_path):
            print(f"WARNING: {video_path} not found, skipping.")
            continue
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        print(f"Processing {video_name}...")
        segments = detect_freezes_in_video(video_path, args.threshold, args.min_duration)
        all_results[video_name] = segments

    print_results(all_results)
    save_results(all_results, args.output)


if __name__ == "__main__":
    main()
