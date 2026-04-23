"""
generate_rater_copies.py
Generates personalised rater HTML files from rater_template.html + config/.

Each rater gets a unique randomised video order with the constraint that
two same-topic videos (same topicKey) are at least 4 positions apart.
Videos are relabeled neutrally as Video_R01..Video_R20 to preserve blinding.

Outputs:
    output/rater_R{n}.html            — file to email to each rater
    output/rater_R{n}_order.csv       — offline audit trail (display_label -> internal_id)

Usage:
    python generate_rater_copies.py --raters R1 R2 R3 R4
    python generate_rater_copies.py --raters R1 R2 R3 R4 --seed 42
"""

from __future__ import annotations
import argparse
import csv
import hashlib
import json
import random
from pathlib import Path

BASE = Path(__file__).resolve().parent
CFG = BASE / "config"
TEMPLATE = BASE / "rater_template.html"
OUT = BASE / "output"
MIN_SPACING = 4  # minimum positions between same-topic videos


def load_inputs():
    with open(CFG / "videos.json", encoding="utf-8") as f:
        videos = json.load(f)["videos"]
    with open(CFG / "cpips.json", encoding="utf-8") as f:
        cpips = json.load(f)["cpips"]
    zone_card_html = (CFG / "zones_reference_card.html").read_text(encoding="utf-8")
    apps_script_url = (CFG / "apps_script_url.txt").read_text(encoding="utf-8").strip()
    template = TEMPLATE.read_text(encoding="utf-8")
    return videos, cpips, zone_card_html, apps_script_url, template


def seed_for(rater_id: str, base_seed: int) -> int:
    h = hashlib.sha256(f"{base_seed}_{rater_id}".encode()).hexdigest()
    return int(h[:8], 16)


def spacing_ok(order, min_spacing=MIN_SPACING):
    """Check no two same-topic videos appear within min_spacing positions."""
    for i in range(len(order)):
        for j in range(i + 1, min(i + min_spacing + 1, len(order))):
            if order[i]["topicKey"] == order[j]["topicKey"]:
                return False
    return True


def randomise_with_spacing(videos, rng, max_tries=50000):
    """Construct a valid order using round-robin over topics, then random-walk
    through valid swaps to randomise further while preserving the constraint."""
    # Group by topic
    from collections import defaultdict
    by_topic = defaultdict(list)
    for v in videos:
        by_topic[v["topicKey"]].append(v)
    for t in by_topic:
        rng.shuffle(by_topic[t])

    # Sort topics by size (largest first — most constrained)
    topics_sorted = sorted(by_topic.keys(), key=lambda t: -len(by_topic[t]))

    # Constructive placement: round-robin through the large topics, then
    # interleave singletons in the remaining gaps.
    n = len(videos)
    order = [None] * n
    # Place items from each multi-item topic spaced approximately n/k apart
    pos_cursor = 0
    step = max(MIN_SPACING + 1, 1)  # spacing between same-topic items
    # Use an index schedule: for each topic with k items, positions are base, base+step, base+2*step, ... mod n
    used_positions = set()
    for t in topics_sorted:
        items = by_topic[t]
        k = len(items)
        if k == 1:
            continue
        # find a base offset where all k positions (base + i*step) are free
        found = False
        bases = list(range(n))
        rng.shuffle(bases)
        for base in bases:
            positions = [(base + i * step) % n for i in range(k)]
            if len(set(positions)) == k and all(p not in used_positions for p in positions):
                for i, p in enumerate(positions):
                    order[p] = items[i]
                    used_positions.add(p)
                found = True
                break
        if not found:
            # Try different step sizes
            for alt_step in range(step + 1, n):
                for base in bases:
                    positions = [(base + i * alt_step) % n for i in range(k)]
                    if len(set(positions)) == k and all(p not in used_positions for p in positions):
                        for i, p in enumerate(positions):
                            order[p] = items[i]
                            used_positions.add(p)
                        found = True
                        break
                if found:
                    break
        if not found:
            raise RuntimeError(f"Cannot place topic {t} with spacing constraint")

    # Fill remaining positions with singleton-topic items
    singletons = [by_topic[t][0] for t in topics_sorted if len(by_topic[t]) == 1]
    rng.shuffle(singletons)
    free_positions = [i for i in range(n) if order[i] is None]
    for p, item in zip(free_positions, singletons):
        order[p] = item

    if not spacing_ok(order):
        # Try random swaps to fix any residual violation (rare)
        for _ in range(max_tries):
            i, j = rng.randrange(n), rng.randrange(n)
            if i == j:
                continue
            order[i], order[j] = order[j], order[i]
            if spacing_ok(order):
                return order
            order[i], order[j] = order[j], order[i]
        raise RuntimeError("Could not satisfy spacing constraint after repair")

    # Do randomised valid swaps to shuffle while maintaining validity
    for _ in range(500):
        i, j = rng.randrange(n), rng.randrange(n)
        if i == j:
            continue
        order[i], order[j] = order[j], order[i]
        if not spacing_ok(order):
            order[i], order[j] = order[j], order[i]

    return order


def build_config_for_rater(rater_id, videos, cpips, zone_card_html, apps_script_url, base_seed):
    rng = random.Random(seed_for(rater_id, base_seed))
    ordered = randomise_with_spacing(videos, rng)

    # Attach neutral display labels
    rater_videos = []
    for idx, v in enumerate(ordered, start=1):
        rater_videos.append({
            "displayLabel": f"Video_R{idx:02d}",
            "internalId": v["internalId"],
            "youtubeId": v["youtubeId"],
            "duration": v.get("duration", 0),
        })

    # Only include CPIPs for videos this rater has
    rater_cpips = {v["internalId"]: cpips[v["internalId"]] for v in rater_videos}

    return {
        "raterId": rater_id,
        "appsScriptUrl": apps_script_url,
        "videos": rater_videos,
        "cpips": rater_cpips,
        "zoneCardHtml": zone_card_html,
    }


def inject_config(template: str, rater_cfg: dict) -> str:
    # JSON-encode once, then escape for safe embedding inside a <script> tag
    cfg_json = json.dumps(rater_cfg, ensure_ascii=False)
    # Replace </ to avoid premature script termination
    cfg_json_safe = cfg_json.replace("</", "<\\/")
    injection = (
        '<script id="rater-config">\n'
        f"window.RATER_CONFIG = {cfg_json_safe};\n"
        "</script>"
    )
    # Replace the placeholder block in the template
    start = template.find('<script id="rater-config">')
    end = template.find("</script>", start) + len("</script>")
    if start == -1 or end == -1:
        raise RuntimeError("rater-config placeholder not found in template")
    return template[:start] + injection + template[end:]


def write_audit_csv(path: Path, rater_videos):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["order_index", "display_label", "internal_id", "youtube_id"])
        for i, v in enumerate(rater_videos, start=1):
            w.writerow([i, v["displayLabel"], v["internalId"], v["youtubeId"]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raters", nargs="+", default=["R1", "R2", "R3", "R4"])
    ap.add_argument("--seed", type=int, default=20260405)
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    videos, cpips, zone_card_html, apps_script_url, template = load_inputs()

    # Validate inputs
    missing_cpips = [v["internalId"] for v in videos if v["internalId"] not in cpips]
    if missing_cpips:
        raise SystemExit(f"Videos missing CPIPs in cpips.json: {missing_cpips}")
    if apps_script_url.startswith("PASTE_YOUR") or not apps_script_url:
        print("WARNING: apps_script_url.txt still has placeholder — HTML will fail to submit.")
    bad_yt = [v["internalId"] for v in videos if v["youtubeId"].startswith("REPLACE_")]
    if bad_yt:
        print(f"WARNING: videos still have placeholder YouTube IDs: {bad_yt}")

    for rid in args.raters:
        cfg = build_config_for_rater(rid, videos, cpips, zone_card_html, apps_script_url, args.seed)
        html = inject_config(template, cfg)
        out_html = OUT / f"rater_{rid}.html"
        out_csv = OUT / f"rater_{rid}_order.csv"
        out_html.write_text(html, encoding="utf-8")
        write_audit_csv(out_csv, cfg["videos"])
        print(f"  wrote {out_html.name} ({len(cfg['videos'])} videos) + audit csv")

    print(f"\nDone. Files written to: {OUT}")
    print("Email each rater ONLY their own rater_R{n}.html file.")
    print("Keep rater_R{n}_order.csv files offline — they contain the de-blinding mapping.")


if __name__ == "__main__":
    main()
