#!/usr/bin/env python3
"""
Get precise delay timestamps and extract visual frames at each delay point.
Runs at native 30fps around known delay regions for sub-frame accuracy.
Also extracts PNG frames for visual inspection.
"""

import csv
import json
import os
import subprocess
import sys
import numpy as np

FRAME_W = 320
FRAME_H = 240
FRAME_BYTES = FRAME_W * FRAME_H

# Known delay regions from previous analysis (approximate)
DELAY_REGIONS = [
    (68, 80),     # Delay point 1 around 72s
    (112, 122),   # Delay point 2 around 115s (hidden in static)
    (149, 160),   # Delay point 3 around 152.5s
    (170, 180),   # Delay point 4 around 174s
]


def dual_pipe_compare_region(video_a, video_b, start_sec, end_sec, fps=30):
    """Compare two videos frame-by-frame at native fps within a time region."""
    duration = end_sec - start_sec

    def make_cmd(path):
        return [
            "ffmpeg", "-ss", str(start_sec), "-t", str(duration),
            "-i", path,
            "-f", "rawvideo", "-pix_fmt", "gray",
            "-s", f"{FRAME_W}x{FRAME_H}",
            "-v", "quiet",
            "pipe:1"
        ]

    proc_a = subprocess.Popen(make_cmd(video_a), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    proc_b = subprocess.Popen(make_cmd(video_b), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    results = []
    frame_idx = 0

    while True:
        data_a = proc_a.stdout.read(FRAME_BYTES)
        data_b = proc_b.stdout.read(FRAME_BYTES)
        if len(data_a) < FRAME_BYTES or len(data_b) < FRAME_BYTES:
            break

        fa = np.frombuffer(data_a, dtype=np.uint8).reshape(FRAME_H, FRAME_W)
        fb = np.frombuffer(data_b, dtype=np.uint8).reshape(FRAME_H, FRAME_W)
        mad = float(np.mean(np.abs(fa.astype(np.int16) - fb.astype(np.int16))))
        timestamp = start_sec + frame_idx / fps
        results.append((round(timestamp, 4), round(mad, 4)))
        frame_idx += 1

    for p in [proc_a, proc_b]:
        p.terminate()
        p.wait()

    return results


def extract_frame_png(video_path, timestamp, output_path):
    """Extract a single frame as PNG at a specific timestamp."""
    cmd = [
        "ffmpeg", "-ss", str(timestamp),
        "-i", video_path,
        "-frames:v", "1",
        "-y", "-v", "quiet",
        output_path
    ]
    subprocess.run(cmd, capture_output=True)


def find_precise_transition(mad_series, threshold):
    """Find exact frame where MAD crosses threshold (low->high)."""
    transitions = []
    for i in range(1, len(mad_series)):
        prev_t, prev_mad = mad_series[i-1]
        curr_t, curr_mad = mad_series[i]
        if prev_mad < threshold and curr_mad >= threshold:
            transitions.append(("diverge", prev_t, curr_t, curr_mad))
        elif prev_mad >= threshold and curr_mad < threshold:
            transitions.append(("resync", prev_t, curr_t, prev_mad))
    return transitions


def main():
    output_dir = "outputs/freeze_detection/precise"
    os.makedirs(output_dir, exist_ok=True)
    frames_dir = os.path.join(output_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    videos = {
        "A1": "A1.mp4",
        "A3": "A3.mp4",
        "A5": "A5.mp4",
    }
    baseline = "A1.mp4"  # Use A1 as reference for inter-variant comparison

    print("PRECISE DELAY POINT ANALYSIS")
    print("=" * 70)

    all_delay_info = []

    for region_idx, (start, end) in enumerate(DELAY_REGIONS, 1):
        print(f"\n--- Delay Region {region_idx}: {start}-{end}s ---")

        # Compare A1 vs A3 and A1 vs A5 at 30fps
        for target_name, target_path in [("A3", "A3.mp4"), ("A5", "A5.mp4")]:
            mad_series = dual_pipe_compare_region(baseline, target_path, start, end, fps=30)

            # Determine threshold - for region 2 (hidden), use lower threshold
            if region_idx == 2:
                threshold = 0.1  # Very low - the signal is subtle (MAD ~0.47)
            else:
                threshold = 1.0

            transitions = find_precise_transition(mad_series, threshold)

            for event, t_before, t_after, mad_val in transitions:
                timestamp = (t_before + t_after) / 2
                print(f"  A1 vs {target_name}: {event} at {timestamp:.3f}s (MAD={mad_val:.4f})")

                info = {
                    "delay_point": region_idx,
                    "comparison": f"A1_vs_{target_name}",
                    "event": event,
                    "timestamp_before": round(t_before, 4),
                    "timestamp_after": round(t_after, 4),
                    "precise_timestamp": round(timestamp, 4),
                    "mad": round(mad_val, 4),
                }
                all_delay_info.append(info)

        # Also compare A3 vs A5
        mad_series_35 = dual_pipe_compare_region("A3.mp4", "A5.mp4", start, end, fps=30)
        threshold_35 = 0.1 if region_idx == 2 else 1.0
        transitions_35 = find_precise_transition(mad_series_35, threshold_35)
        for event, t_before, t_after, mad_val in transitions_35:
            timestamp = (t_before + t_after) / 2
            print(f"  A3 vs A5: {event} at {timestamp:.3f}s (MAD={mad_val:.4f})")
            all_delay_info.append({
                "delay_point": region_idx,
                "comparison": "A3_vs_A5",
                "event": event,
                "timestamp_before": round(t_before, 4),
                "timestamp_after": round(t_after, 4),
                "precise_timestamp": round(timestamp, 4),
                "mad": round(mad_val, 4),
            })

    # Extract frames at each delay point from A0 (baseline) for visual context
    print("\n\n--- Extracting visual frames at delay points ---")

    # Derive delay timestamps in A0's timeline
    # The delay in A1 at time T means A0 also has the transition at T (before any shift)
    # For point 1: diverge at ~72s in A1 timeline = same in A0 (first delay)
    # For point 2: ~115.3s in A1 timeline, but A1 has 1s shift from point 1, so A0 time ≈ 115.3 - 1 = 114.3s
    # For point 3: ~152.5s in A1 timeline, A1 has 2s shift, so A0 time ≈ 152.5 - 2 = 150.5s
    # For point 4: ~173.9s in A1 timeline, A1 has 3s shift, so A0 time ≈ 173.9 - 3 = 170.9s

    # Extract frames from all 4 versions at each delay point
    delay_timestamps_a1 = []
    for info in all_delay_info:
        if info["comparison"] == "A1_vs_A3" and info["event"] == "diverge":
            delay_timestamps_a1.append((info["delay_point"], info["precise_timestamp"]))

    for dp_num, t_a1 in delay_timestamps_a1:
        # Cumulative A1 delay before this point = (dp_num - 1) * 1s
        t_a0 = t_a1 - (dp_num - 1) * 1.0

        print(f"\n  Delay Point {dp_num}:")
        print(f"    A1 timeline: {t_a1:.2f}s")
        print(f"    A0 timeline (approx): {t_a0:.2f}s")

        # Extract frames from each video at multiple timestamps around the delay
        for offset_label, offset in [("2s_before", -2), ("at_delay", 0), ("2s_after", 2)]:
            for vid_name in ["A0", "A1", "A3", "A5"]:
                t = t_a1 + offset  # Use A1 timeline for all (they're synced before the delay)
                fname = f"dp{dp_num}_{offset_label}_{vid_name}.png"
                fpath = os.path.join(frames_dir, fname)
                extract_frame_png(f"{vid_name}.mp4", t, fpath)

            print(f"    Extracted {offset_label} frames (t={t_a1 + offset:.2f}s)")

    # Save detailed results
    csv_path = os.path.join(output_dir, "precise_delay_points.csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["delay_point", "comparison", "event",
                                                "timestamp_before", "timestamp_after",
                                                "precise_timestamp", "mad"])
        writer.writeheader()
        writer.writerows(all_delay_info)

    print(f"\n\nResults saved to {csv_path}")
    print(f"Frames saved to {frames_dir}/")

    # Final summary
    print("\n" + "=" * 70)
    print("SUMMARY: PRECISE DELAY INJECTION POINTS")
    print("=" * 70)

    for dp_num, t_a1 in delay_timestamps_a1:
        # Find resync times for A3 and A5
        resync_a3 = None
        resync_a5 = None
        resync_a3a5 = None
        for info in all_delay_info:
            if info["delay_point"] == dp_num and info["event"] == "resync":
                if info["comparison"] == "A1_vs_A3":
                    resync_a3 = info["precise_timestamp"]
                elif info["comparison"] == "A1_vs_A5":
                    resync_a5 = info["precise_timestamp"]
                elif info["comparison"] == "A3_vs_A5":
                    resync_a3a5 = info["precise_timestamp"]

        dur_a3 = round(resync_a3 - t_a1, 2) if resync_a3 else "?"
        dur_a5 = round(resync_a5 - t_a1, 2) if resync_a5 else "?"

        t_a0 = t_a1 - (dp_num - 1) * 1.0

        print(f"\n  Point {dp_num}: {t_a1:.2f}s (A1 timeline) / ~{t_a0:.1f}s (A0 timeline)")
        print(f"    A3-A1 mismatch: {dur_a3}s (expect 2.0)")
        print(f"    A5-A1 mismatch: {dur_a5}s (expect 4.0)")
        if resync_a3a5:
            # A3 vs A5 diverge point
            div_a3a5 = None
            for info in all_delay_info:
                if info["delay_point"] == dp_num and info["event"] == "diverge" and info["comparison"] == "A3_vs_A5":
                    div_a3a5 = info["precise_timestamp"]
            if div_a3a5:
                dur_a3a5 = round(resync_a3a5 - div_a3a5, 2)
                print(f"    A5-A3 mismatch: {dur_a3a5}s (expect 2.0)")


if __name__ == "__main__":
    main()
