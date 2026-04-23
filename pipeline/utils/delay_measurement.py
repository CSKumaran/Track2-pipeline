"""Frame-level visual delay measurement between related videos.

Uses same-timestamp SSIM comparison to detect where videos diverge visually,
then estimates the actual delay duration at each divergence point.

Two modes:
  1. Same-timestamp comparison: extracts frames at identical timestamps from
     reference and target videos, computes SSIM → dip regions = delay points.
  2. Delay estimation (--estimate-delays): at each detected dip, searches for
     the time offset that maximises SSIM → the delay duration.

Usage:
    # Basic: find divergence points between A1 and A3/A5
    python -m pipeline.utils.delay_measurement \
        --reference A1.mp4 --targets A3.mp4 A5.mp4 \
        --expected-delays 2 4 --output-dir outputs/delay_measurement

    # With delay estimation at each dip
    python -m pipeline.utils.delay_measurement \
        --reference A1.mp4 --targets A3.mp4 A5.mp4 \
        --expected-delays 2 4 --estimate-delays
"""
from __future__ import annotations

import argparse
import csv
import logging
import os

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

logger = logging.getLogger(__name__)


# ── frame extraction ─────────────────────────────────────────────────────────

def extract_frame_at_time(cap: cv2.VideoCapture, t: float,
                          size: tuple[int, int] = (320, 240)) -> np.ndarray | None:
    """Extract a single grayscale frame at time t (seconds)."""
    cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000)
    ret, frame = cap.read()
    if not ret:
        return None
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, size)


def _video_duration(cap: cv2.VideoCapture) -> float:
    fps = cap.get(cv2.CAP_PROP_FPS)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return n / fps if fps > 0 else 0.0


# ── Mode 1: same-timestamp SSIM comparison ───────────────────────────────────

def compare_at_same_timestamps(
    ref_path: str,
    target_paths: list[str],
    target_names: list[str],
    sample_interval: float = 1.0,
) -> list[dict]:
    """Compare reference vs targets at identical timestamps.

    At each timestamp t, extracts a frame from reference and each target,
    then computes SSIM(ref_frame, target_frame). Regions where SSIM drops
    indicate visual divergence (delay points).
    """
    ref_cap = cv2.VideoCapture(ref_path)
    fps = ref_cap.get(cv2.CAP_PROP_FPS)
    duration = _video_duration(ref_cap)
    logger.info("Reference: %s (%.1ffps, %.1fs)", ref_path, fps, duration)

    target_caps = []
    for tp in target_paths:
        cap = cv2.VideoCapture(tp)
        target_caps.append(cap)
        tfps = cap.get(cv2.CAP_PROP_FPS)
        logger.info("  Target: %s (%.1ffps)", tp, tfps)

    times = np.arange(0, duration - 1, sample_interval)
    results = []

    for t in times:
        ref_frame = extract_frame_at_time(ref_cap, t)
        if ref_frame is None:
            continue

        row = {"time": round(t, 2)}
        for tname, tcap in zip(target_names, target_caps):
            tgt_frame = extract_frame_at_time(tcap, t)
            if tgt_frame is None:
                row[f"{tname}_ssim"] = None
            else:
                row[f"{tname}_ssim"] = round(ssim(ref_frame, tgt_frame), 4)
        results.append(row)

    ref_cap.release()
    for cap in target_caps:
        cap.release()

    return results


# ── Mode 1 cont: detect dip regions ─────────────────────────────────────────

def detect_dip_regions(
    results: list[dict],
    target_name: str,
    threshold: float = 0.95,
    merge_gap: float = 2.0,
    min_duration: float = 0.5,
) -> list[dict]:
    """Find contiguous regions where SSIM < threshold for a target.

    Returns list of dicts: {start, end, duration, min_ssim, mean_ssim}.
    """
    # Extract (time, ssim) pairs
    points = []
    for row in results:
        s = row.get(f"{target_name}_ssim")
        if s is not None:
            points.append((row["time"], s))

    if not points:
        return []

    # Find sub-threshold points
    dip_times = [(t, s) for t, s in points if s < threshold]
    if not dip_times:
        return []

    # Group contiguous dips (merge within merge_gap seconds)
    regions = []
    current = [dip_times[0]]
    for t, s in dip_times[1:]:
        if t - current[-1][0] <= merge_gap:
            current.append((t, s))
        else:
            regions.append(current)
            current = [(t, s)]
    regions.append(current)

    # Build region summaries
    dip_list = []
    for region in regions:
        times = [t for t, _ in region]
        ssims = [s for _, s in region]
        dur = times[-1] - times[0]
        if dur < min_duration and len(region) < 2:
            continue  # skip single-point noise
        dip_list.append({
            "start": round(times[0], 2),
            "end": round(times[-1], 2),
            "duration": round(dur, 2),
            "min_ssim": round(min(ssims), 4),
            "mean_ssim": round(np.mean(ssims), 4),
            "center": round(np.mean(times), 2),
            "n_points": len(region),
        })

    return dip_list


# ── Mode 2: estimate delay at dip ───────────────────────────────────────────

def estimate_delay_at_dip(
    ref_path: str,
    target_path: str,
    dip_start: float,
    dip_duration: float,
    max_delay: float = 8.0,
    step: float = 0.1,
) -> dict:
    """Estimate the delay duration at a specific dip location.

    Primary method: dip duration — the dip lasts exactly as long as the
    relative delay between the two videos (proven reliable for animated content).

    Secondary method: frame offset search — takes the reference frame at
    dip midpoint and searches the target for where it appears.
    This can be noisy for animated content but provides a cross-check.
    """
    # Primary: dip duration IS the delay
    primary_delay = round(dip_duration, 1)

    # Secondary: frame offset search at dip midpoint
    dip_mid = dip_start + dip_duration / 2.0
    ref_cap = cv2.VideoCapture(ref_path)
    tgt_cap = cv2.VideoCapture(target_path)

    ref_frame = extract_frame_at_time(ref_cap, dip_mid)
    if ref_frame is None:
        ref_cap.release()
        tgt_cap.release()
        return {"delay_by_duration": primary_delay, "delay_by_offset": 0.0, "offset_ssim": 0.0}

    best_offset = 0.0
    best_ssim = -1.0

    # Search forward and backward
    for offset in np.arange(-max_delay, max_delay + step, step):
        tgt_frame = extract_frame_at_time(tgt_cap, dip_mid + offset)
        if tgt_frame is None:
            continue
        s = ssim(ref_frame, tgt_frame)
        if s > best_ssim:
            best_ssim = s
            best_offset = offset

    ref_cap.release()
    tgt_cap.release()

    return {
        "delay_by_duration": primary_delay,
        "delay_by_offset": round(best_offset, 2),
        "offset_ssim": round(best_ssim, 4),
    }


# ── output ───────────────────────────────────────────────────────────────────

def save_results(
    results: list[dict],
    dip_regions: dict[str, list[dict]],
    target_names: list[str],
    output_dir: str,
    expected_delays: list[float] | None = None,
    delay_estimates: dict[str, list[dict]] | None = None,
):
    """Save results to CSV and print summary."""
    os.makedirs(output_dir, exist_ok=True)

    # Full SSIM timeseries CSV
    if results:
        keys = results[0].keys()
        with open(os.path.join(output_dir, "ssim_timeseries.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(results)

    # Dip regions CSV (one per target)
    all_dips = []
    for tn in target_names:
        for dip in dip_regions.get(tn, []):
            row = {"target": tn, **dip}
            if delay_estimates and tn in delay_estimates:
                # Find matching estimate
                for est in delay_estimates[tn]:
                    if abs(est["dip_center"] - dip["center"]) < 1.0:
                        row["delay_by_duration"] = est["delay_by_duration"]
                        row["delay_by_offset"] = est["delay_by_offset"]
                        row["offset_ssim"] = est["offset_ssim"]
                        break
            all_dips.append(row)

    if all_dips:
        keys = all_dips[0].keys()
        with open(os.path.join(output_dir, "dip_regions.csv"), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            w.writerows(all_dips)

    # Console summary
    print("\n" + "=" * 80)
    print("VISUAL DELAY MEASUREMENT RESULTS (same-timestamp SSIM)")
    print("=" * 80)
    print(f"Sampled {len(results)} timepoints")

    for tn_idx, tn in enumerate(target_names):
        dips = dip_regions.get(tn, [])
        expected = expected_delays[tn_idx] if expected_delays and tn_idx < len(expected_delays) else None
        exp_str = f" (expected delay: {expected}s)" if expected is not None else ""
        print(f"\n{tn}: {len(dips)} dip region(s){exp_str}")

        if dips:
            for di, dip in enumerate(dips):
                est_str = ""
                if delay_estimates and tn in delay_estimates:
                    for est in delay_estimates[tn]:
                        if abs(est["dip_center"] - dip["center"]) < 1.0:
                            match_tag = ""
                            if expected is not None:
                                if abs(est["delay_by_duration"] - expected) < 1.5:
                                    match_tag = " [OK]"
                                else:
                                    match_tag = " [MISMATCH]"
                            est_str = (f"  delay={est['delay_by_duration']:.1f}s"
                                       f" (offset_check={est['delay_by_offset']:+.1f}s,"
                                       f" ssim={est['offset_ssim']:.3f}){match_tag}")
                            break
                print(f"  Dip {di+1}: {dip['start']:.1f}s - {dip['end']:.1f}s "
                      f"(dur={dip['duration']:.1f}s, min_ssim={dip['min_ssim']:.3f}, "
                      f"mean_ssim={dip['mean_ssim']:.3f}){est_str}")

        # Overall SSIM stats
        all_ssim = [r.get(f"{tn}_ssim") for r in results if r.get(f"{tn}_ssim") is not None]
        if all_ssim:
            print(f"  Overall: mean_ssim={np.mean(all_ssim):.3f}, "
                  f"median={np.median(all_ssim):.3f}, "
                  f"min={min(all_ssim):.3f}, max={max(all_ssim):.3f}")

    print("=" * 80)


# ── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Visual delay measurement via same-timestamp SSIM comparison"
    )
    parser.add_argument(
        "--reference", required=True,
        help="Reference video path (e.g., A1.mp4)"
    )
    parser.add_argument(
        "--targets", nargs="+", required=True,
        help="Target video paths (e.g., A3.mp4 A5.mp4)"
    )
    parser.add_argument(
        "--target-names", nargs="+", default=None,
        help="Names for targets (default: derived from filenames)"
    )
    parser.add_argument(
        "--expected-delays", type=float, nargs="+", default=None,
        help="Expected delays in seconds (e.g., 2 4)"
    )
    parser.add_argument(
        "--output-dir", default="outputs/delay_measurement",
        help="Output directory"
    )
    parser.add_argument(
        "--sample-interval", type=float, default=1.0,
        help="Sample interval in seconds (default: 1.0)"
    )
    parser.add_argument(
        "--ssim-threshold", type=float, default=0.95,
        help="SSIM threshold for dip detection (default: 0.95)"
    )
    parser.add_argument(
        "--estimate-delays", action="store_true",
        help="Estimate actual delay duration at each dip (slower)"
    )
    parser.add_argument(
        "--max-delay", type=float, default=8.0,
        help="Max delay to search for in estimation mode (default: 8.0s)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    target_names = args.target_names or [
        os.path.splitext(os.path.basename(p))[0] for p in args.targets
    ]

    # Mode 1: same-timestamp SSIM comparison
    logger.info("Mode 1: Same-timestamp SSIM comparison (interval=%.1fs, threshold=%.2f)",
                args.sample_interval, args.ssim_threshold)
    results = compare_at_same_timestamps(
        args.reference, args.targets, target_names,
        sample_interval=args.sample_interval,
    )

    # Detect dip regions per target
    dip_regions = {}
    for tn in target_names:
        dips = detect_dip_regions(
            results, tn,
            threshold=args.ssim_threshold,
            merge_gap=2.0,
            min_duration=0.5,
        )
        dip_regions[tn] = dips
        logger.info("%s: %d dip regions detected", tn, len(dips))

    # Mode 2: estimate delay at each dip (optional)
    delay_estimates = None
    if args.estimate_delays:
        logger.info("Mode 2: Estimating delay at each dip (max_delay=%.1fs)", args.max_delay)
        delay_estimates = {}
        for tn_idx, tn in enumerate(target_names):
            estimates = []
            for dip in dip_regions[tn]:
                est = estimate_delay_at_dip(
                    args.reference, args.targets[tn_idx],
                    dip["start"],
                    dip["duration"],
                    max_delay=args.max_delay,
                )
                est["dip_center"] = dip["center"]
                estimates.append(est)
                logger.info("  %s dip at %.1fs: delay=%.1fs (by duration), "
                            "offset=%+.1fs (ssim=%.3f)",
                            tn, dip["center"],
                            est["delay_by_duration"],
                            est["delay_by_offset"], est["offset_ssim"])
            delay_estimates[tn] = estimates

    save_results(results, dip_regions, target_names, args.output_dir,
                 args.expected_delays, delay_estimates)


if __name__ == "__main__":
    main()
