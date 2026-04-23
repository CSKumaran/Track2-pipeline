#!/usr/bin/env python3
"""
Compare delayed video variants against baseline (A0) to find delay injection points.

Extracts frames at regular intervals from both videos, computes SSIM/MAD at each
timestamp, and identifies transitions from "matching" to "mismatched" content.
Each such transition marks a delay injection point.

Usage:
    python compare_with_baseline.py A0.mp4 A1.mp4 A3.mp4 A5.mp4 -o outputs/freeze_detection/
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import numpy as np

FRAME_W = 320
FRAME_H = 240
FRAME_BYTES = FRAME_W * FRAME_H


def get_video_info(path):
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
    duration = float(info["format"]["duration"])
    return fps, duration


def extract_frame_at_time(video_path, timestamp):
    """Extract a single grayscale frame at a specific timestamp."""
    cmd = [
        "ffmpeg", "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-f", "rawvideo", "-pix_fmt", "gray",
        "-s", f"{FRAME_W}x{FRAME_H}",
        "-v", "quiet",
        "pipe:1"
    ]
    result = subprocess.run(cmd, capture_output=True)
    if len(result.stdout) < FRAME_BYTES:
        return None
    return np.frombuffer(result.stdout, dtype=np.uint8).reshape(FRAME_H, FRAME_W)


def compute_mad(frame1, frame2):
    """Mean absolute difference between two frames."""
    return np.mean(np.abs(frame1.astype(np.int16) - frame2.astype(np.int16)))


def compare_videos(baseline_path, delayed_path, sample_interval=0.5):
    """
    Compare baseline vs delayed video at regular intervals.
    Returns list of (timestamp, mad) tuples.
    """
    _, duration = get_video_info(baseline_path)
    video_name = os.path.splitext(os.path.basename(delayed_path))[0]

    timestamps = np.arange(0, duration, sample_interval)
    results = []

    print(f"  Comparing {video_name} vs baseline at {len(timestamps)} points ({sample_interval}s intervals)...")

    for i, t in enumerate(timestamps):
        base_frame = extract_frame_at_time(baseline_path, t)
        delayed_frame = extract_frame_at_time(delayed_path, t)

        if base_frame is None or delayed_frame is None:
            continue

        mad = compute_mad(base_frame, delayed_frame)
        results.append((round(t, 2), mad))

        if (i + 1) % 50 == 0:
            print(f"    {i+1}/{len(timestamps)} timestamps processed...")

    return results


def find_delay_points(comparison_results, match_threshold=2.0):
    """
    Find delay injection points from comparison timeseries.

    A delay point is where the video transitions from matching (MAD < threshold)
    to mismatched (MAD > threshold). After a delay, content stays shifted until
    it either hits another static segment or the next delay.
    """
    delay_points = []
    was_matching = True  # assume start is matching

    for i, (t, mad) in enumerate(comparison_results):
        is_matching = mad < match_threshold
        if was_matching and not is_matching:
            # Transition from matching to mismatched = delay injection point
            delay_points.append(t)
        was_matching = is_matching

    return delay_points


def estimate_delay_duration(baseline_path, delayed_path, delay_timestamp, search_range=8.0, step=0.1):
    """
    At a delay point, estimate the delay duration by finding the time offset
    that makes the delayed video match the baseline again.

    At time T (delay point), baseline shows frame X.
    The delayed video froze at T, so after the freeze, it shows content from
    T + delay_duration onwards. So delayed[T + delay_duration + d] should match
    baseline[T + d] for small d values after the freeze.

    We search for the offset that minimizes MAD between:
      baseline[T + offset] vs delayed[T + offset]  -- won't work, content stays shifted

    Better approach: find offset D such that baseline[T] matches delayed[T + D]
    That means D seconds of content were frozen/skipped.
    """
    base_frame = extract_frame_at_time(baseline_path, delay_timestamp)
    if base_frame is None:
        return None

    best_mad = float('inf')
    best_offset = 0

    # The delayed video at T+D should show what baseline shows at T (if D is the delay)
    # Actually: after delay, delayed video is BEHIND by D seconds.
    # So delayed[T] != baseline[T], but delayed[T] might == baseline[T - D]
    # Or equivalently: baseline[T + D] might match delayed[T + D] again if a static segment
    # bridges the gap.

    # Simpler: search forward in the delayed video for where the baseline frame at T+epsilon
    # appears. The offset = cumulative delay up to this point.
    # But we only want the INCREMENTAL delay at this point.

    # Best approach: compare baseline[T-1] (just before delay) with delayed[T+offset]
    # to find when the delayed video moves past the freeze.
    pre_delay = max(0, delay_timestamp - 0.5)
    ref_frame = extract_frame_at_time(baseline_path, pre_delay)
    if ref_frame is None:
        ref_frame = base_frame

    # Check: what does the delayed video show at T? It should be the frozen frame.
    # Search forward in delayed video for when it stops showing the frozen frame.
    # That gives us the freeze end point.

    # But we already know from self-comparison where freezes are.
    # Let's instead use a different approach: compare baseline vs delayed at
    # T + small_offset for increasing offsets. When they match again, the offset
    # equals the delay duration (only works if there's recognizable content change after T).

    return None  # Will use differential approach instead


def differential_delay_estimation(comparison_results, delay_points, match_threshold=2.0):
    """
    Estimate delay durations by analyzing the MAD timeseries pattern.

    After each delay point, the content is shifted. The next time content matches
    again is when a static segment in the baseline bridges the offset gap.
    The delay duration can be inferred from the pattern of matches/mismatches.
    """
    # For now, return the delay points without duration estimates.
    # Duration estimation requires cross-video frame search.
    return [(t, None) for t in delay_points]


def main():
    parser = argparse.ArgumentParser(description="Compare delayed videos against baseline")
    parser.add_argument("baseline", help="Baseline video (A0)")
    parser.add_argument("delayed", nargs="+", help="Delayed video(s) (A1, A3, A5)")
    parser.add_argument("-o", "--output", default="outputs/freeze_detection",
                        help="Output directory")
    parser.add_argument("--interval", type=float, default=0.5,
                        help="Sampling interval in seconds (default: 0.5)")
    parser.add_argument("--threshold", type=float, default=2.0,
                        help="MAD threshold for match/mismatch (default: 2.0)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Cross-Video Baseline Comparison")
    print("=" * 50)
    print(f"Baseline: {args.baseline}")
    print(f"Interval: {args.interval}s")
    print(f"Threshold: MAD < {args.threshold}")
    print()

    all_delay_points = {}

    for delayed_path in args.delayed:
        video_name = os.path.splitext(os.path.basename(delayed_path))[0]
        print(f"\nProcessing {video_name}...")

        # Compare at regular intervals
        comparison = compare_videos(args.baseline, delayed_path, args.interval)

        # Save timeseries
        ts_path = os.path.join(args.output, f"mad_timeseries_{video_name}.csv")
        with open(ts_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "mad"])
            for t, mad in comparison:
                writer.writerow([t, round(mad, 4)])

        # Find delay points (match -> mismatch transitions)
        delay_pts = find_delay_points(comparison, args.threshold)
        all_delay_points[video_name] = delay_pts

        print(f"\n  {video_name} delay injection points (match->mismatch transitions):")
        for i, t in enumerate(delay_pts, 1):
            print(f"    Point {i}: {t:.1f}s")

        # Also show where content re-syncs (mismatch -> match)
        resync_pts = []
        was_matching = True
        for t, mad in comparison:
            is_matching = mad < args.threshold
            if not was_matching and is_matching:
                resync_pts.append(t)
            was_matching = is_matching

        if resync_pts:
            print(f"  Re-sync points (mismatch->match): {[f'{t:.1f}s' for t in resync_pts]}")

    # Summary
    print("\n" + "=" * 60)
    print("DELAY INJECTION POINT SUMMARY")
    print("=" * 60)
    for video_name, pts in all_delay_points.items():
        print(f"\n  {video_name}: {len(pts)} delay point(s)")
        for i, t in enumerate(pts, 1):
            print(f"    {i}. {t:.1f}s")

    # Save summary
    summary_path = os.path.join(args.output, "delay_injection_points.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "delay_point", "timestamp"])
        for video_name, pts in all_delay_points.items():
            for i, t in enumerate(pts, 1):
                writer.writerow([video_name, i, t])

    print(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main()
