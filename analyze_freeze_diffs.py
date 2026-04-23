#!/usr/bin/env python3
"""
Analyze freeze segment differences between baseline (A0) and delayed versions
to identify delay injection points and estimate delay durations.

Also performs fast dual-pipe frame comparison between A0 and each delayed version.

Usage:
    python analyze_freeze_diffs.py A0.mp4 A1.mp4 A3.mp4 A5.mp4 -o outputs/freeze_detection/
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
        "ffprobe", "-v", "quiet", "-print_format", "json",
        "-show_streams", "-show_format", path
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    info = json.loads(result.stdout)
    stream = info["streams"][0]
    fps_num, fps_den = map(int, stream["r_frame_rate"].split("/"))
    return fps_num / fps_den, float(info["format"]["duration"])


def dual_pipe_compare(baseline_path, delayed_path, sample_fps=2):
    """
    Compare two videos frame-by-frame using dual ffmpeg pipes.
    Much faster than individual frame extraction.

    sample_fps: frames per second to compare (2 = one frame every 0.5s)
    """
    fps, duration = get_video_info(baseline_path)
    video_name = os.path.splitext(os.path.basename(delayed_path))[0]

    def make_cmd(path):
        return [
            "ffmpeg", "-i", path,
            "-vf", f"fps={sample_fps}",
            "-f", "rawvideo", "-pix_fmt", "gray",
            "-s", f"{FRAME_W}x{FRAME_H}",
            "-v", "quiet",
            "pipe:1"
        ]

    proc_base = subprocess.Popen(make_cmd(baseline_path), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    proc_delay = subprocess.Popen(make_cmd(delayed_path), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    results = []
    frame_idx = 0

    while True:
        base_data = proc_base.stdout.read(FRAME_BYTES)
        delay_data = proc_delay.stdout.read(FRAME_BYTES)

        if len(base_data) < FRAME_BYTES or len(delay_data) < FRAME_BYTES:
            break

        base_frame = np.frombuffer(base_data, dtype=np.uint8).reshape(FRAME_H, FRAME_W)
        delay_frame = np.frombuffer(delay_data, dtype=np.uint8).reshape(FRAME_H, FRAME_W)

        mad = np.mean(np.abs(base_frame.astype(np.int16) - delay_frame.astype(np.int16)))
        timestamp = frame_idx / sample_fps
        results.append((round(timestamp, 3), round(float(mad), 4)))
        frame_idx += 1

    for proc in [proc_base, proc_delay]:
        proc.terminate()
        proc.wait()

    print(f"  {video_name}: compared {len(results)} frame pairs")
    return results


def find_transitions(comparison, match_threshold=3.0):
    """
    Find match->mismatch and mismatch->match transitions.
    Returns (delay_points, resync_points).
    """
    delay_points = []
    resync_points = []
    was_matching = True

    for t, mad in comparison:
        is_matching = mad < match_threshold
        if was_matching and not is_matching:
            delay_points.append(t)
        elif not was_matching and is_matching:
            resync_points.append(t)
        was_matching = is_matching

    return delay_points, resync_points


def estimate_delay_at_point(comparison, delay_time, next_resync_time, match_threshold=3.0):
    """
    Estimate the delay duration at a specific injection point.

    After a delay D is injected at time T:
    - The delayed video at T+d shows what baseline shows at T+d-D (for small d after the delay)
    - Content stays mismatched for a while until a static segment bridges the gap
    - The mismatch duration is related to but not equal to D

    Better approach: the mismatched region length correlates with delay duration.
    """
    # Find length of mismatched region
    mismatch_duration = next_resync_time - delay_time if next_resync_time else None
    return mismatch_duration


def main():
    parser = argparse.ArgumentParser(description="Analyze delays via dual-pipe frame comparison")
    parser.add_argument("baseline", help="Baseline video (A0)")
    parser.add_argument("delayed", nargs="+", help="Delayed video(s)")
    parser.add_argument("-o", "--output", default="outputs/freeze_detection",
                        help="Output directory")
    parser.add_argument("--sample-fps", type=float, default=2,
                        help="Comparison sampling rate in fps (default: 2 = every 0.5s)")
    parser.add_argument("--threshold", type=float, default=3.0,
                        help="MAD threshold for match vs mismatch (default: 3.0)")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    print("Dual-Pipe Frame Comparison")
    print("=" * 50)
    print(f"Baseline: {args.baseline}")
    print(f"Sample rate: {args.sample_fps} fps (every {1/args.sample_fps:.2f}s)")
    print(f"Threshold: MAD < {args.threshold}")
    print()

    all_results = {}

    for delayed_path in args.delayed:
        if not os.path.exists(delayed_path):
            print(f"WARNING: {delayed_path} not found, skipping.")
            continue

        video_name = os.path.splitext(os.path.basename(delayed_path))[0]
        print(f"Comparing {video_name} vs baseline...")

        comparison = dual_pipe_compare(args.baseline, delayed_path, args.sample_fps)

        # Save timeseries
        ts_path = os.path.join(args.output, f"comparison_{video_name}.csv")
        with open(ts_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "mad"])
            for t, mad in comparison:
                writer.writerow([t, mad])

        # Find transitions
        delay_pts, resync_pts = find_transitions(comparison, args.threshold)
        all_results[video_name] = {
            "comparison": comparison,
            "delay_points": delay_pts,
            "resync_points": resync_pts,
        }

    # Print results
    print("\n" + "=" * 80)
    print("DELAY DETECTION RESULTS")
    print("=" * 80)

    for video_name, data in all_results.items():
        delay_pts = data["delay_points"]
        resync_pts = data["resync_points"]

        print(f"\n--- {video_name} vs baseline ---")
        print(f"  Delay injection points (content diverges): {len(delay_pts)}")
        print(f"  Re-sync points (content re-aligns): {len(resync_pts)}")
        print()

        # Pair up delay and resync points
        all_transitions = [(t, "DELAY") for t in delay_pts] + [(t, "RESYNC") for t in resync_pts]
        all_transitions.sort()

        print(f"  {'Time':>8}  {'Event':>8}  {'MAD at point':>12}")
        print(f"  {'-'*8}  {'-'*8}  {'-'*12}")
        for t, event in all_transitions:
            # Find MAD at this timestamp
            mad_at_t = None
            for ts, mad in data["comparison"]:
                if abs(ts - t) < 0.01:
                    mad_at_t = mad
                    break
            print(f"  {t:>7.1f}s  {event:>8}  {mad_at_t:>11.2f}" if mad_at_t else f"  {t:>7.1f}s  {event:>8}")

    # Cross-video comparison of delay points
    if len(all_results) > 1:
        print(f"\n{'=' * 80}")
        print("CROSS-VIDEO DELAY POINT COMPARISON")
        print("=" * 80)

        videos = list(all_results.keys())
        header = f"  {'#':<4}"
        for v in videos:
            header += f"  {v + ' delay':>15}  {v + ' resync':>15}"
        print(header)

        max_pts = max(len(d["delay_points"]) for d in all_results.values())
        for i in range(max_pts):
            row = f"  {i+1:<4}"
            for v in videos:
                d = all_results[v]
                dp = d["delay_points"][i] if i < len(d["delay_points"]) else None
                rp = d["resync_points"][i] if i < len(d["resync_points"]) else None
                row += f"  {dp:>14.1f}s" if dp else f"  {'---':>15}"
                row += f"  {rp:>14.1f}s" if rp else f"  {'---':>15}"
            print(row)

    # MAD statistics for threshold tuning
    print(f"\n{'=' * 80}")
    print("MAD DISTRIBUTION (for threshold tuning)")
    print("=" * 80)
    for video_name, data in all_results.items():
        mads = [mad for _, mad in data["comparison"]]
        mads_arr = np.array(mads)
        matched = mads_arr[mads_arr < args.threshold]
        mismatched = mads_arr[mads_arr >= args.threshold]
        print(f"\n  {video_name}:")
        print(f"    Total points: {len(mads_arr)}")
        print(f"    Matched (MAD < {args.threshold}): {len(matched)} "
              f"(mean={np.mean(matched):.2f}, max={np.max(matched):.2f})" if len(matched) > 0 else "")
        print(f"    Mismatched (MAD >= {args.threshold}): {len(mismatched)} "
              f"(mean={np.mean(mismatched):.2f}, min={np.min(mismatched):.2f})" if len(mismatched) > 0 else "")

    # Save summary
    summary_path = os.path.join(args.output, "delay_analysis_summary.csv")
    with open(summary_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["video", "event_type", "event_num", "timestamp"])
        for video_name, data in all_results.items():
            for i, t in enumerate(data["delay_points"], 1):
                writer.writerow([video_name, "delay", i, t])
            for i, t in enumerate(data["resync_points"], 1):
                writer.writerow([video_name, "resync", i, t])

    print(f"\nResults saved to {args.output}/")


if __name__ == "__main__":
    main()
