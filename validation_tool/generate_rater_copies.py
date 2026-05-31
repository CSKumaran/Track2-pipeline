"""
generate_rater_copies.py
Generates personalised rater HTML files from rater_template.html + config/.

Per spec tool_design_spec.md (locked 2026-05-31):
  - Each rater gets a unique randomised video order across both stimulus sets
    (manipulated + fresh). Same-topic videos (manipulated dose-set) are kept
    ≥ MIN_SPACING positions apart.
  - Videos are relabeled neutrally as Video_R01..Video_RNN to preserve blinding.
  - For each CPIP we compute a viewing window per §8:
      manipulated set: [t_narr - 5, t_narr + 7]   (12 s)
      fresh set:       [t_narr - 5, t_narr + 10]  (15 s)
    A CPIP entry may also override these with explicit windowStartS / windowEndS.

Outputs:
    output/rater_R{n}.html            -- file to email to each rater
    output/rater_R{n}_order.csv       -- offline de-blinding audit trail

Usage:
    python generate_rater_copies.py --raters R1 R2 R3 R4 R5
    python generate_rater_copies.py --raters R1 --seed 42
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

BASE = Path(__file__).resolve().parent
CFG = BASE / "config"
TEMPLATE = BASE / "rater_template.html"
OUT = BASE / "output"
MIN_SPACING = 4  # min positions between two videos sharing the same topicKey

# Per spec §8
WINDOW_RULES = {
    "manipulated": (-5.0, 7.0),
    "fresh":       (-5.0, 10.0),
}

# Per spec §5 — CTML principles listed for the Q4 checklist
DEFAULT_PRINCIPLES = [
    "Temporal contiguity",
    "Spatial contiguity",
    "Modality",
    "Redundancy",
    "Coherence",
    "Signaling",
    "Segmenting",
    "Personalisation",
]

CONFIDENCE_LEVELS = ["very_low", "low", "medium", "high", "very_high"]
CONFIDENCE_LABELS = {
    "very_low":  "Very low",
    "low":       "Low",
    "medium":    "Medium",
    "high":      "High",
    "very_high": "Very high",
}


def load_inputs():
    with open(CFG / "videos.json", encoding="utf-8") as f:
        videos = json.load(f)["videos"]
    with open(CFG / "cpips.json", encoding="utf-8") as f:
        cpips = json.load(f)["cpips"]
    scale_card_html = (CFG / "rating_scale_reference.html").read_text(encoding="utf-8")
    apps_script_url = (CFG / "apps_script_url.txt").read_text(encoding="utf-8").strip()
    template = TEMPLATE.read_text(encoding="utf-8")
    return videos, cpips, scale_card_html, apps_script_url, template


def seed_for(rater_id: str, base_seed: int) -> int:
    h = hashlib.sha256(f"{base_seed}_{rater_id}".encode()).hexdigest()
    return int(h[:8], 16)


def compute_window(video_set: str, t_narr: float, override_start, override_end):
    if override_start is not None and override_end is not None:
        return float(override_start), float(override_end)
    if video_set not in WINDOW_RULES:
        raise ValueError(
            f"Unknown video set '{video_set}'. Expected one of: {list(WINDOW_RULES)}"
        )
    offset_start, offset_end = WINDOW_RULES[video_set]
    start = max(0.0, t_narr + offset_start)
    end = t_narr + offset_end
    return start, end


def enrich_cpips(video, raw_cpips):
    """Compute viewing windows for each CPIP and return the rater-facing dicts."""
    out = []
    for c in raw_cpips:
        # Tolerate legacy field name 'timestamp' as a fallback for t_narr_s
        t_narr = c.get("t_narr_s")
        if t_narr is None:
            t_narr = c.get("timestamp")
        if t_narr is None:
            raise ValueError(
                f"CPIP {c.get('cpipId')!r} in {video['internalId']} is missing t_narr_s"
            )
        ws, we = compute_window(
            video["set"], float(t_narr),
            c.get("windowStartS"), c.get("windowEndS"),
        )
        out.append({
            "cpipId": c["cpipId"],
            "t_narr_s": float(t_narr),
            "conceptLabel": c.get("conceptLabel", ""),
            "window_start_s": round(ws, 3),
            "window_end_s": round(we, 3),
        })
    return out


def spacing_ok(order, min_spacing=MIN_SPACING):
    for i in range(len(order)):
        for j in range(i + 1, min(i + min_spacing + 1, len(order))):
            if order[i]["topicKey"] == order[j]["topicKey"]:
                return False
    return True


def randomise_with_spacing(videos, rng, max_tries=50000):
    """Round-robin placement of multi-item topics, then fill singletons,
    then randomised valid swaps to shuffle while keeping the constraint."""
    by_topic = defaultdict(list)
    for v in videos:
        by_topic[v["topicKey"]].append(v)
    for t in by_topic:
        rng.shuffle(by_topic[t])

    topics_sorted = sorted(by_topic.keys(), key=lambda t: -len(by_topic[t]))
    n = len(videos)
    order = [None] * n
    step = max(MIN_SPACING + 1, 1)
    used = set()

    for t in topics_sorted:
        items = by_topic[t]
        k = len(items)
        if k == 1:
            continue
        bases = list(range(n))
        rng.shuffle(bases)
        placed = False
        for base in bases:
            positions = [(base + i * step) % n for i in range(k)]
            if len(set(positions)) == k and all(p not in used for p in positions):
                for i, p in enumerate(positions):
                    order[p] = items[i]
                    used.add(p)
                placed = True
                break
        if not placed:
            for alt_step in range(step + 1, n):
                for base in bases:
                    positions = [(base + i * alt_step) % n for i in range(k)]
                    if len(set(positions)) == k and all(p not in used for p in positions):
                        for i, p in enumerate(positions):
                            order[p] = items[i]
                            used.add(p)
                        placed = True
                        break
                if placed:
                    break
        if not placed:
            raise RuntimeError(f"Cannot place topic {t} with spacing constraint")

    singletons = [by_topic[t][0] for t in topics_sorted if len(by_topic[t]) == 1]
    rng.shuffle(singletons)
    free_positions = [i for i in range(n) if order[i] is None]
    for p, item in zip(free_positions, singletons):
        order[p] = item

    if not spacing_ok(order):
        for _ in range(max_tries):
            i, j = rng.randrange(n), rng.randrange(n)
            if i == j:
                continue
            order[i], order[j] = order[j], order[i]
            if spacing_ok(order):
                break
            order[i], order[j] = order[j], order[i]
        else:
            raise RuntimeError("Could not satisfy spacing constraint after repair")

    for _ in range(500):
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            continue
        order[i], order[j] = order[j], order[i]
        if not spacing_ok(order):
            order[i], order[j] = order[j], order[i]

    return order


def build_config_for_rater(rater_id, videos, cpips, scale_card_html, apps_script_url, base_seed):
    rng = random.Random(seed_for(rater_id, base_seed))
    ordered = randomise_with_spacing(videos, rng)

    rater_videos = []
    rater_cpips = {}
    for idx, v in enumerate(ordered, start=1):
        rater_videos.append({
            "displayLabel": f"Video_R{idx:02d}",
            "internalId":   v["internalId"],
            "youtubeId":    v["youtubeId"],
            "duration":     v.get("duration", 0),
            "set":          v["set"],
            "topic":        v.get("topic", ""),
            "delay_s":      v.get("delay_s"),
        })
        raw = cpips.get(v["internalId"], [])
        rater_cpips[v["internalId"]] = enrich_cpips(v, raw)

    return {
        "raterId": rater_id,
        "appsScriptUrl": apps_script_url,
        "principles": DEFAULT_PRINCIPLES,
        "confidenceLevels": CONFIDENCE_LEVELS,
        "confidenceLabels": CONFIDENCE_LABELS,
        "videos": rater_videos,
        "cpips": rater_cpips,
        "ratingScaleHtml": scale_card_html,
    }


def inject_config(template: str, rater_cfg: dict) -> str:
    cfg_json = json.dumps(rater_cfg, ensure_ascii=False)
    cfg_json_safe = cfg_json.replace("</", "<\\/")
    injection = (
        '<script id="rater-config">\n'
        f"window.RATER_CONFIG = {cfg_json_safe};\n"
        "</script>"
    )
    start = template.find('<script id="rater-config">')
    if start == -1:
        raise RuntimeError("rater-config placeholder not found in template")
    end = template.find("</script>", start) + len("</script>")
    return template[:start] + injection + template[end:]


def write_audit_csv(path: Path, rater_videos, rater_cpips):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow([
            "order_index", "display_label", "internal_id", "youtube_id",
            "set", "topic", "delay_s", "n_cpips",
        ])
        for i, v in enumerate(rater_videos, start=1):
            n_cpips = len(rater_cpips.get(v["internalId"], []))
            w.writerow([
                i, v["displayLabel"], v["internalId"], v["youtubeId"],
                v["set"], v.get("topic", ""), v.get("delay_s", ""), n_cpips,
            ])


def validate_inputs(videos, cpips, apps_script_url):
    missing = [v["internalId"] for v in videos if v["internalId"] not in cpips]
    if missing:
        raise SystemExit(f"Videos missing CPIPs in cpips.json: {missing}")
    bad_set = [v["internalId"] for v in videos if v.get("set") not in WINDOW_RULES]
    if bad_set:
        raise SystemExit(
            f"Videos with unknown/missing 'set' field (must be one of "
            f"{list(WINDOW_RULES)}): {bad_set}"
        )
    if not apps_script_url or apps_script_url.startswith("PASTE_") or apps_script_url == "REPLACE_ME":
        print("WARNING: apps_script_url.txt still has placeholder — HTML will fail to submit.")
    bad_yt = [v["internalId"] for v in videos if v["youtubeId"].startswith("REPLACE_")]
    if bad_yt:
        print(f"WARNING: videos still have placeholder YouTube IDs: {bad_yt}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raters", nargs="+", default=["R1", "R2", "R3", "R4"])
    ap.add_argument("--seed", type=int, default=20260531)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    videos, cpips, scale_card_html, apps_script_url, template = load_inputs()
    validate_inputs(videos, cpips, apps_script_url)

    for rid in args.raters:
        cfg = build_config_for_rater(rid, videos, cpips, scale_card_html, apps_script_url, args.seed)
        html = inject_config(template, cfg)
        out_html = OUT / f"rater_{rid}.html"
        out_csv = OUT / f"rater_{rid}_order.csv"
        out_html.write_text(html, encoding="utf-8")
        write_audit_csv(out_csv, cfg["videos"], cfg["cpips"])
        print(f"  wrote {out_html.name} ({len(cfg['videos'])} videos) + audit csv")

    print(f"\nDone. Files written to: {OUT}")
    print("Email each rater ONLY their own rater_R{n}.html file.")
    print("Keep rater_R{n}_order.csv files offline — they contain the de-blinding mapping.")


if __name__ == "__main__":
    main()
