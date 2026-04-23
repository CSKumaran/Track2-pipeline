"""Automated delay detection pipeline for instructional video experiments.

Discovers visual delay points across video sets by comparing delay variants
using same-timestamp SSIM analysis. No prior knowledge of delay locations needed.

Usage:
    python -m pipeline.utils.detect_delays --video-dir . --sets A B C D
    python -m pipeline.utils.detect_delays --video-dir . --sets A --no-baseline
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import re
from dataclasses import dataclass, field

import numpy as np

from pipeline.utils.delay_measurement import (
    compare_at_same_timestamps,
    detect_dip_regions,
    estimate_delay_at_dip,
)

logger = logging.getLogger(__name__)

VALID_DELAY_LEVELS = {0, 1, 3, 5}


# ── data structures ──────────────────────────────────────────────────────────

@dataclass
class VideoVariant:
    set_name: str
    delay_level: int
    path: str

@dataclass
class VideoSet:
    set_name: str
    variants: dict[int, VideoVariant] = field(default_factory=dict)

    @property
    def has_baseline(self) -> bool:
        return 0 in self.variants

    @property
    def has_x1(self) -> bool:
        return 1 in self.variants

@dataclass
class ComparisonPair:
    ref_path: str
    ref_name: str
    target_path: str
    target_name: str
    expected_dip_duration: float
    strategy: str  # "inter-variant" or "baseline"
    label: str     # e.g., "X1vsX3"

@dataclass
class DelayPoint:
    timestamp: float
    delay_per_variant: dict[int, float]
    confidence: str
    evidence: list[str]


# ── video discovery ──────────────────────────────────────────────────────────

def discover_video_sets(
    video_dir: str,
    set_names: list[str] | None = None,
) -> dict[str, VideoSet]:
    """Scan video_dir for files matching {Letter}{Digit}.mp4 pattern."""
    pattern = re.compile(r'^([A-Za-z]+)(\d+)\.mp4$')
    sets: dict[str, VideoSet] = {}

    for fname in sorted(os.listdir(video_dir)):
        m = pattern.match(fname)
        if not m:
            continue
        prefix = m.group(1).upper()
        level = int(m.group(2))
        if level not in VALID_DELAY_LEVELS:
            continue
        if set_names and prefix not in [s.upper() for s in set_names]:
            continue

        if prefix not in sets:
            sets[prefix] = VideoSet(set_name=prefix)
        sets[prefix].variants[level] = VideoVariant(
            set_name=prefix,
            delay_level=level,
            path=os.path.join(video_dir, fname),
        )

    for sname, vset in sets.items():
        levels = sorted(vset.variants.keys())
        logger.info("Set %s: variants %s", sname, levels)

    return sets


# ── comparison planning ──────────────────────────────────────────────────────

def plan_comparisons(video_set: VideoSet) -> list[ComparisonPair]:
    """Build ordered comparison pairs for a video set."""
    pairs = []
    v = video_set.variants
    sn = video_set.set_name

    # Primary: inter-variant (cleanest signal)
    if 1 in v and 3 in v:
        pairs.append(ComparisonPair(
            ref_path=v[1].path, ref_name=f"{sn}1",
            target_path=v[3].path, target_name=f"{sn}3",
            expected_dip_duration=2.0, strategy="inter-variant",
            label=f"{sn}1vs{sn}3",
        ))
    if 1 in v and 5 in v:
        pairs.append(ComparisonPair(
            ref_path=v[1].path, ref_name=f"{sn}1",
            target_path=v[5].path, target_name=f"{sn}5",
            expected_dip_duration=4.0, strategy="inter-variant",
            label=f"{sn}1vs{sn}5",
        ))
    # Fallback: if X1 missing, use X3 vs X5
    if 1 not in v and 3 in v and 5 in v:
        pairs.append(ComparisonPair(
            ref_path=v[3].path, ref_name=f"{sn}3",
            target_path=v[5].path, target_name=f"{sn}5",
            expected_dip_duration=2.0, strategy="inter-variant",
            label=f"{sn}3vs{sn}5",
        ))

    # Secondary: baseline comparisons (cumulative but gives absolute reference)
    if 0 in v:
        for lvl in [1, 3, 5]:
            if lvl in v:
                pairs.append(ComparisonPair(
                    ref_path=v[0].path, ref_name=f"{sn}0",
                    target_path=v[lvl].path, target_name=f"{sn}{lvl}",
                    expected_dip_duration=float(lvl),
                    strategy="baseline",
                    label=f"{sn}0vs{sn}{lvl}",
                ))

    return pairs


# ── adaptive threshold ───────────────────────────────────────────────────────

def compute_adaptive_threshold(
    ssim_values: list[float],
    min_threshold: float = 0.90,
    max_threshold: float = 0.999,
) -> float:
    """Derive SSIM dip threshold from data distribution.

    For inter-variant comparisons (median SSIM = 1.000), uses a tight
    threshold (0.999) because identical encoding means any SSIM < 1.0
    is a real divergence. For noisier comparisons, adapts based on the
    percentile spread.
    """
    arr = np.array([v for v in ssim_values if v is not None])
    if len(arr) == 0:
        return 0.95

    median_val = float(np.median(arr))
    p1 = float(np.percentile(arr, 1))

    # If median is very close to 1.0 (inter-variant, identical encoding),
    # any sub-1.0 SSIM is meaningful → use tight threshold
    if median_val >= 0.999:
        threshold = max_threshold  # 0.999
    else:
        # Noisier comparison (baseline, different encoding)
        threshold = median_val - 0.3 * (median_val - p1)

    return float(np.clip(threshold, min_threshold, max_threshold))


# ── per-set analysis ─────────────────────────────────────────────────────────

def analyze_single_set(
    video_set: VideoSet,
    sample_interval: float = 0.5,
    estimate_delays: bool = True,
    skip_baseline: bool = False,
) -> dict:
    """Run full delay detection for one video set."""
    sn = video_set.set_name
    logger.info("=" * 60)
    logger.info("Analyzing Set %s", sn)
    logger.info("=" * 60)

    all_pairs = plan_comparisons(video_set)
    if skip_baseline:
        all_pairs = [p for p in all_pairs if p.strategy != "baseline"]

    comparisons = {}
    thresholds_used = {}
    ssim_data = {}

    for pair in all_pairs:
        logger.info("  %s (%s)...", pair.label, pair.strategy)

        # Same-timestamp SSIM comparison
        results = compare_at_same_timestamps(
            pair.ref_path, [pair.target_path], [pair.target_name],
            sample_interval=sample_interval,
        )

        # Extract SSIM values
        ssim_col = f"{pair.target_name}_ssim"
        ssim_vals = [r.get(ssim_col) for r in results if r.get(ssim_col) is not None]

        # Adaptive threshold
        threshold = compute_adaptive_threshold(ssim_vals)
        thresholds_used[pair.label] = threshold
        logger.info("    threshold=%.4f (median=%.4f)", threshold,
                     float(np.median(ssim_vals)) if ssim_vals else 0)

        # Detect dip regions
        merge_gap = pair.expected_dip_duration + 1.0
        min_dur = max(0.5, pair.expected_dip_duration * 0.3)
        dips = detect_dip_regions(
            results, pair.target_name,
            threshold=threshold,
            merge_gap=merge_gap,
            min_duration=min_dur,
        )
        logger.info("    %d dip regions detected", len(dips))

        # Estimate delay at each dip
        dip_estimates = []
        for dip in dips:
            est = {"dip": dip}
            if estimate_delays:
                delay_est = estimate_delay_at_dip(
                    pair.ref_path, pair.target_path,
                    dip["start"], dip["duration"],
                    max_delay=8.0,
                )
                est["delay_estimate"] = delay_est
            dip_estimates.append(est)

        comparisons[pair.label] = {
            "pair": pair,
            "dips": dips,
            "dip_estimates": dip_estimates,
            "n_samples": len(results),
        }

        # Store SSIM timeseries for report
        ssim_data[pair.label] = [
            {"time": r["time"], "ssim": r.get(ssim_col)}
            for r in results
        ]

    # Cross-validate
    delay_points = cross_validate_dips(comparisons, video_set)
    logger.info("Set %s: %d delay points detected", sn, len(delay_points))
    for dp in delay_points:
        logger.info("  t=%.1fs delays=%s confidence=%s evidence=%s",
                     dp.timestamp, dp.delay_per_variant,
                     dp.confidence, dp.evidence)

    return {
        "set_name": sn,
        "comparisons": comparisons,
        "delay_points": delay_points,
        "thresholds_used": thresholds_used,
        "ssim_data": ssim_data,
    }


# ── cross-validation ────────────────────────────────────────────────────────

def cross_validate_dips(
    comparisons: dict,
    video_set: VideoSet,
    time_tolerance: float = 3.0,
) -> list[DelayPoint]:
    """Merge dip detections across comparison pairs into unified delay points."""
    # Collect dip centers from inter-variant comparisons
    all_centers = []  # (center_time, label, dip_duration, strategy)
    for label, comp in comparisons.items():
        pair = comp["pair"]
        for dip in comp["dips"]:
            all_centers.append((
                dip["center"],
                label,
                dip["duration"],
                pair.strategy,
            ))

    if not all_centers:
        return []

    # Sort by time
    all_centers.sort(key=lambda x: x[0])

    # Cluster by proximity
    clusters = []
    current_cluster = [all_centers[0]]
    for item in all_centers[1:]:
        if item[0] - current_cluster[-1][0] <= time_tolerance:
            current_cluster.append(item)
        else:
            clusters.append(current_cluster)
            current_cluster = [item]
    clusters.append(current_cluster)

    # Build DelayPoints from clusters
    delay_points = []
    for cluster in clusters:
        timestamp = float(np.mean([c[0] for c in cluster]))
        evidence = list(set(c[1] for c in cluster))

        # Count inter-variant confirmations
        n_intervar = sum(1 for c in cluster if c[3] == "inter-variant")
        n_baseline = sum(1 for c in cluster if c[3] == "baseline")

        if n_intervar >= 2:
            confidence = "high"
        elif n_intervar >= 1 and n_baseline >= 1:
            confidence = "medium"
        elif n_intervar >= 1:
            confidence = "medium"
        else:
            confidence = "low"

        # Derive per-variant delay from dip durations
        delay_per_variant = {}
        for center, label, duration, strategy in cluster:
            if strategy == "inter-variant":
                # X1vsX3 dip of duration d → X3 has d more delay than X1
                # X1vsX5 dip of duration d → X5 has d more delay than X1
                # The individual delay per point for each variant:
                # X1 gets 1s, X3 gets 3s, X5 gets 5s per point
                # duration ≈ |delay_target - delay_ref| per point
                # For X1vsX3: duration ≈ 2s → confirms 1s and 3s
                # For X1vsX5: duration ≈ 4s → confirms 1s and 5s
                pass  # Delay values derived below

        # Use dip durations to infer per-variant delays
        # Collect inter-variant dip durations
        iv_durations = {}
        for center, label, duration, strategy in cluster:
            if strategy == "inter-variant":
                iv_durations[label] = duration

        # Derive from known delay structure:
        # X1vsX3 duration ≈ 2 → (3-1)=2, so per-point: X1=1, X3=3
        # X1vsX5 duration ≈ 4 → (5-1)=4, so per-point: X1=1, X5=5
        sn = video_set.set_name
        x1x3_label = f"{sn}1vs{sn}3"
        x1x5_label = f"{sn}1vs{sn}5"
        x3x5_label = f"{sn}3vs{sn}5"

        if x1x3_label in iv_durations:
            d13 = iv_durations[x1x3_label]
            # d13 ≈ 2 → X1_delay = d13/2, X3_delay = d13/2 * 3
            # More directly: per-point delay = d13 / 2 * level
            delay_per_variant[1] = round(d13 / 2, 1)
            delay_per_variant[3] = round(d13 / 2 * 3, 1)

        if x1x5_label in iv_durations:
            d15 = iv_durations[x1x5_label]
            delay_per_variant[1] = round(d15 / 4, 1)
            delay_per_variant[5] = round(d15 / 4 * 5, 1)

        if x3x5_label in iv_durations:
            d35 = iv_durations[x3x5_label]
            delay_per_variant[3] = round(d35 / 2 * 3, 1)
            delay_per_variant[5] = round(d35 / 2 * 5, 1)

        # If we have both X1vsX3 and X1vsX5, average the X1 estimate
        if 1 in delay_per_variant:
            pass  # already set

        # Add baseline info if available
        for center, label, duration, strategy in cluster:
            if strategy == "baseline":
                # X0vsX1 dip at this point = cumulative, less reliable
                # Just note it as evidence, don't override delay values
                pass

        delay_points.append(DelayPoint(
            timestamp=round(timestamp, 1),
            delay_per_variant=delay_per_variant,
            confidence=confidence,
            evidence=evidence,
        ))

    # Sort by timestamp
    delay_points.sort(key=lambda dp: dp.timestamp)
    return delay_points


# ── output ───────────────────────────────────────────────────────────────────

def save_set_results(set_name: str, analysis: dict, output_dir: str) -> None:
    """Save per-set CSV outputs."""
    set_dir = os.path.join(output_dir, set_name)
    os.makedirs(set_dir, exist_ok=True)

    # Delay points CSV
    dps = analysis["delay_points"]
    if dps:
        rows = []
        for i, dp in enumerate(dps):
            rows.append({
                "delay_point": i + 1,
                "timestamp": dp.timestamp,
                "X1_delay": dp.delay_per_variant.get(1, ""),
                "X3_delay": dp.delay_per_variant.get(3, ""),
                "X5_delay": dp.delay_per_variant.get(5, ""),
                "confidence": dp.confidence,
                "evidence": ";".join(dp.evidence),
            })
        path = os.path.join(set_dir, f"{set_name}_delay_points.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    # Dip regions CSV (all comparisons)
    dip_rows = []
    for label, comp in analysis["comparisons"].items():
        for dip in comp["dips"]:
            row = {"comparison": label, "strategy": comp["pair"].strategy, **dip}
            dip_rows.append(row)
    if dip_rows:
        path = os.path.join(set_dir, f"{set_name}_dip_regions.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=dip_rows[0].keys())
            w.writeheader()
            w.writerows(dip_rows)

    # Thresholds JSON
    path = os.path.join(set_dir, f"{set_name}_thresholds.json")
    with open(path, "w") as f:
        json.dump(analysis["thresholds_used"], f, indent=2)

    logger.info("Saved Set %s results to %s", set_name, set_dir)


def save_summary(all_analyses: dict[str, dict], output_dir: str) -> None:
    """Save cross-set summary CSV and print console summary."""
    os.makedirs(output_dir, exist_ok=True)
    rows = []
    for sn, analysis in sorted(all_analyses.items()):
        dps = analysis["delay_points"]
        timestamps = [dp.timestamp for dp in dps]
        rows.append({
            "set": sn,
            "n_delay_points": len(dps),
            "timestamps": ";".join(f"{t:.1f}" for t in timestamps),
            "all_high_confidence": all(dp.confidence == "high" for dp in dps),
        })

    if rows:
        path = os.path.join(output_dir, "summary.csv")
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=rows[0].keys())
            w.writeheader()
            w.writerows(rows)

    # Console summary
    print("\n" + "=" * 70)
    print("DELAY DETECTION SUMMARY")
    print("=" * 70)
    for sn, analysis in sorted(all_analyses.items()):
        dps = analysis["delay_points"]
        ts = [f"{dp.timestamp:.1f}s" for dp in dps]
        conf = [dp.confidence for dp in dps]
        print(f"\n  Set {sn}: {len(dps)} delay point(s) at [{', '.join(ts)}]")
        for i, dp in enumerate(dps):
            delays = ", ".join(f"X{k}={v:.1f}s" for k, v in sorted(dp.delay_per_variant.items()))
            print(f"    Point {i+1} (t={dp.timestamp:.1f}s): {delays} "
                  f"[{dp.confidence}] ({', '.join(dp.evidence)})")
    print("=" * 70)


def generate_html_report(
    all_analyses: dict[str, dict],
    output_path: str,
) -> None:
    """Generate HTML report with Chart.js SSIM plots."""
    n_sets = len(all_analyses)
    total_points = sum(len(a["delay_points"]) for a in all_analyses.values())

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8">
<title>Delay Detection Report</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
<style>
  body {{ font-family: 'Segoe UI', sans-serif; margin: 20px; background: #f5f5f5; color: #333; }}
  h1 {{ color: #1a237e; }}
  h2 {{ color: #283593; border-bottom: 2px solid #3f51b5; padding-bottom: 5px; }}
  .summary {{ background: white; padding: 15px 25px; border-radius: 8px; margin: 15px 0;
              box-shadow: 0 2px 4px rgba(0,0,0,0.1); display: flex; gap: 40px; }}
  .stat {{ text-align: center; }}
  .stat .value {{ font-size: 2em; font-weight: bold; color: #1a237e; }}
  .stat .label {{ font-size: 0.9em; color: #666; }}
  .set-section {{ background: white; padding: 20px; border-radius: 8px; margin: 20px 0;
                  box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
  table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
  th {{ background: #1a237e; color: white; padding: 8px 12px; font-size: 0.85em; text-align: center; }}
  td {{ padding: 8px 12px; border-bottom: 1px solid #e0e0e0; text-align: center; font-size: 0.9em; }}
  tr:hover {{ background: #e8eaf6; }}
  .high {{ color: #2e7d32; font-weight: bold; }}
  .medium {{ color: #f57f17; font-weight: bold; }}
  .low {{ color: #c62828; font-weight: bold; }}
  canvas {{ max-height: 250px; margin: 15px 0; }}
</style>
</head><body>
<h1>Automated Delay Detection Report</h1>
<div class="summary">
  <div class="stat"><div class="value">{n_sets}</div><div class="label">Video Sets</div></div>
  <div class="stat"><div class="value">{total_points}</div><div class="label">Total Delay Points</div></div>
</div>
"""

    for sn, analysis in sorted(all_analyses.items()):
        dps = analysis["delay_points"]
        html += f'<div class="set-section">\n'
        html += f'<h2>Set {sn} &mdash; {len(dps)} Delay Point(s)</h2>\n'

        # Delay points table
        if dps:
            html += '<table><tr><th>#</th><th>Timestamp</th>'
            html += '<th>X1 Delay</th><th>X3 Delay</th><th>X5 Delay</th>'
            html += '<th>Confidence</th><th>Evidence</th></tr>\n'
            for i, dp in enumerate(dps):
                conf_cls = dp.confidence
                html += f'<tr><td>{i+1}</td><td>{dp.timestamp:.1f}s</td>'
                html += f'<td>{dp.delay_per_variant.get(1, "—")}</td>'
                html += f'<td>{dp.delay_per_variant.get(3, "—")}</td>'
                html += f'<td>{dp.delay_per_variant.get(5, "—")}</td>'
                html += f'<td class="{conf_cls}">{dp.confidence.upper()}</td>'
                html += f'<td>{", ".join(dp.evidence)}</td></tr>\n'
            html += '</table>\n'

        # SSIM chart
        chart_id = f"chart_{sn}"
        html += f'<canvas id="{chart_id}"></canvas>\n'

        # Build Chart.js datasets
        colors = {
            "inter-variant": ["#1565C0", "#0D47A1"],
            "baseline": ["#E65100", "#BF360C", "#8D6E63"],
        }
        iv_idx = 0
        bl_idx = 0
        datasets_js = []
        annotations_js = []

        for label, ssim_series in analysis["ssim_data"].items():
            comp = analysis["comparisons"][label]
            strategy = comp["pair"].strategy
            if strategy == "inter-variant":
                color = colors["inter-variant"][iv_idx % 2]
                iv_idx += 1
                width = 2
            else:
                color = colors["baseline"][bl_idx % 3]
                bl_idx += 1
                width = 1

            times = [p["time"] for p in ssim_series]
            ssims = [p["ssim"] if p["ssim"] is not None else None for p in ssim_series]

            # Downsample for performance (max 500 points per trace)
            step = max(1, len(times) // 500)
            times_ds = times[::step]
            ssims_ds = ssims[::step]

            datasets_js.append({
                "label": label,
                "data": [{"x": t, "y": s} for t, s in zip(times_ds, ssims_ds) if s is not None],
                "borderColor": color,
                "borderWidth": width,
                "pointRadius": 0,
                "fill": False,
            })

        # Delay point annotations (vertical lines)
        for dp in dps:
            annotations_js.append({
                "type": "line",
                "xMin": dp.timestamp,
                "xMax": dp.timestamp,
                "borderColor": "#F44336",
                "borderWidth": 2,
                "borderDash": [5, 5],
                "label": {"display": True, "content": f"Delay @ {dp.timestamp:.0f}s",
                          "position": "start", "font": {"size": 10}},
            })

        # Threshold line
        thresholds = list(analysis["thresholds_used"].values())
        if thresholds:
            avg_thresh = float(np.mean(thresholds))
            datasets_js.append({
                "label": f"Threshold (~{avg_thresh:.3f})",
                "data": [{"x": 0, "y": avg_thresh},
                         {"x": max(t["time"] for s in analysis["ssim_data"].values() for t in s), "y": avg_thresh}],
                "borderColor": "#999",
                "borderWidth": 1,
                "borderDash": [3, 3],
                "pointRadius": 0,
                "fill": False,
            })

        data_json = json.dumps(datasets_js)
        html += f"""<script>
new Chart(document.getElementById('{chart_id}'), {{
  type: 'scatter',
  data: {{ datasets: {data_json} }},
  options: {{
    responsive: true,
    scales: {{
      x: {{ title: {{ display: true, text: 'Time (s)' }}, type: 'linear' }},
      y: {{ title: {{ display: true, text: 'SSIM' }}, min: 0.5, max: 1.02 }}
    }},
    plugins: {{
      legend: {{ position: 'top', labels: {{ font: {{ size: 11 }} }} }},
      title: {{ display: true, text: 'Set {sn}: SSIM Timeseries', font: {{ size: 14 }} }}
    }}
  }}
}});
</script>\n"""

        html += '</div>\n'

    html += '</body></html>'

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    logger.info("Wrote HTML report -> %s", output_path)


# ── CLI ──────────────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Automated delay detection across video sets"
    )
    p.add_argument("--video-dir", default=".",
                   help="Directory containing video files (default: .)")
    p.add_argument("--sets", nargs="*", default=None,
                   help="Set names to process (default: auto-detect all)")
    p.add_argument("--output-dir", default="outputs/delay_detection",
                   help="Output directory (default: outputs/delay_detection)")
    p.add_argument("--sample-interval", type=float, default=0.5,
                   help="Seconds between SSIM samples (default: 0.5)")
    p.add_argument("--no-estimate", action="store_true",
                   help="Skip offset-search delay estimation (faster)")
    p.add_argument("--no-baseline", action="store_true",
                   help="Skip X0-based comparisons (inter-variant only)")
    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    # Discover video sets
    video_sets = discover_video_sets(args.video_dir, args.sets)
    if not video_sets:
        logger.error("No video sets found in %s", args.video_dir)
        return

    logger.info("Found %d video set(s): %s", len(video_sets),
                ", ".join(sorted(video_sets.keys())))

    # Analyze each set sequentially
    all_analyses = {}
    for sn in sorted(video_sets.keys()):
        analysis = analyze_single_set(
            video_sets[sn],
            sample_interval=args.sample_interval,
            estimate_delays=not args.no_estimate,
            skip_baseline=args.no_baseline,
        )
        save_set_results(sn, analysis, args.output_dir)
        all_analyses[sn] = analysis

    # Summary
    save_summary(all_analyses, args.output_dir)

    # HTML report
    report_path = os.path.join(args.output_dir, "delay_detection_report.html")
    generate_html_report(all_analyses, report_path)

    logger.info("Done. Results in %s", args.output_dir)


if __name__ == "__main__":
    main()
