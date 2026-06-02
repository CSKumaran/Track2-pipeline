"""
generate_rater_copies.py
Generates personalised rater HTML files from rater_template.html + config/.

Per spec tool_design_spec.md (locked 2026-05-31) + 2026-06-02 session design:

  - The 23 videos per rater are split across 3 SESSIONS:
      * Each session contains 6 manipulated videos (one per topic A–F).
      * Within each session the 6 topics each contribute exactly one delay
        variant — chosen so the session has a 2-2-2 delay balance across
        {0, 1.5, 5} s. Across the 3 sessions, each topic is seen at each
        delay exactly once (within-rater complete; cross-rater
        counterbalanced via per-rater seed).
      * Fresh videos are distributed 2 + 2 + 1 across sessions (or 4 + 3 + 3
        when ≥10 fresh videos are present), random which session gets the
        "short" allocation. Fresh and manipulated interleave randomly
        within each session.
  - Soft session boundaries: after a session is complete the rater sees
    an interstitial in the HTML; "Continue now" or "Resume later".
  - Videos are relabeled neutrally as Video_R01..Video_RNN.
  - For each CPIP we compute a viewing window per spec §8:
      manipulated set: [t_narr - 5, t_narr + 7]   (12 s)
      fresh set:       [t_narr - 5, t_narr + 10]  (15 s)
    A CPIP entry may override these with explicit windowStartS / windowEndS.

Outputs:
    output/rater_R{n}.html            -- file to email to each rater
    output/rater_R{n}_order.csv       -- offline de-blinding audit trail
                                         (gitignored; includes session columns)

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

# Session design (2026-06-02 decision)
N_SESSIONS = 3
DELAYS = [0, 1.5, 5]  # the 3 manipulated delays used by spec §3
EXPECTED_TOPICS = 6   # A–F

# Delay-pattern cyclic permutations across N_SESSIONS sessions.
# With 6 manipulated topics and 3 patterns × 2 topics per pattern, each
# session ends up with exactly 2 of each delay (2-2-2 balance), and each
# topic gets each delay exactly once across the 3 sessions.
DELAY_PATTERNS = [
    [DELAYS[(i + offset) % len(DELAYS)] for i in range(N_SESSIONS)]
    for offset in range(len(DELAYS))
]
# DELAY_PATTERNS == [[0, 1.5, 5], [1.5, 5, 0], [5, 0, 1.5]]


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


def assign_sessions(manipulated, fresh, rng):
    """Build N_SESSIONS session-blocks of (6 manipulated + N fresh) videos
    with the 2-2-2 delay balance per session and within-session shuffle.

    Returns a list of N_SESSIONS lists, each containing 7 or 8 video dicts.
    """
    # Group manipulated by topic
    by_topic = defaultdict(list)
    for v in manipulated:
        by_topic[v["topicKey"]].append(v)
    topics = sorted(by_topic.keys())

    if len(topics) != EXPECTED_TOPICS:
        raise SystemExit(
            f"Session design requires exactly {EXPECTED_TOPICS} manipulated "
            f"topics; got {len(topics)}: {topics}"
        )
    for t in topics:
        delays_present = sorted(v["delay_s"] for v in by_topic[t])
        if delays_present != DELAYS:
            raise SystemExit(
                f"Topic {t!r} must have delays {DELAYS}; got {delays_present}"
            )

    # Shuffle topic order so different raters get different topic→pattern
    # assignments. Patterns are cyclic permutations of DELAYS; assigning two
    # topics to each pattern guarantees the 2-2-2 per-session balance.
    rng.shuffle(topics)
    n_patterns = len(DELAY_PATTERNS)
    topics_per_pattern = len(topics) // n_patterns  # 6 / 3 = 2
    if topics_per_pattern * n_patterns != len(topics):
        raise SystemExit(
            f"Topic count {len(topics)} must be divisible by the number of "
            f"delay patterns ({n_patterns})."
        )

    sessions = [[] for _ in range(N_SESSIONS)]
    for idx, t in enumerate(topics):
        pattern = DELAY_PATTERNS[idx // topics_per_pattern]
        for s in range(N_SESSIONS):
            target_delay = pattern[s]
            video = next(v for v in by_topic[t] if v["delay_s"] == target_delay)
            sessions[s].append(video)

    # Distribute fresh videos across sessions: base + remainder-to-random.
    fresh_shuffled = list(fresh)
    rng.shuffle(fresh_shuffled)
    n_fresh = len(fresh_shuffled)
    base = n_fresh // N_SESSIONS
    remainder = n_fresh % N_SESSIONS
    extra_session_indices = list(range(N_SESSIONS))
    rng.shuffle(extra_session_indices)
    extras = set(extra_session_indices[:remainder])

    cursor = 0
    for s in range(N_SESSIONS):
        count = base + (1 if s in extras else 0)
        for _ in range(count):
            sessions[s].append(fresh_shuffled[cursor])
            cursor += 1

    # Sanity check: every fresh consumed
    if cursor != n_fresh:
        raise RuntimeError(f"Fresh distribution mismatch: {cursor} placed of {n_fresh}")

    # Shuffle the videos within each session so manipulated + fresh interleave
    for s in range(N_SESSIONS):
        rng.shuffle(sessions[s])

    return sessions


def build_config_for_rater(rater_id, videos, cpips, scale_card_html, apps_script_url, base_seed):
    rng = random.Random(seed_for(rater_id, base_seed))

    manipulated = [v for v in videos if v.get("set") == "manipulated"]
    fresh = [v for v in videos if v.get("set") == "fresh"]
    sessions = assign_sessions(manipulated, fresh, rng)

    rater_videos = []
    rater_cpips = {}
    global_order = 0
    for s_idx, session_videos in enumerate(sessions, start=1):
        for pos_in_session, v in enumerate(session_videos, start=1):
            global_order += 1
            rater_videos.append({
                "displayLabel":       f"Video_R{global_order:02d}",
                "internalId":         v["internalId"],
                "youtubeId":          v["youtubeId"],
                "duration":           v.get("duration", 0),
                "set":                v["set"],
                "topic":              v.get("topic", ""),
                "delay_s":            v.get("delay_s"),
                "session_index":      s_idx,
                "position_in_session": pos_in_session,
            })
            raw = cpips.get(v["internalId"], [])
            rater_cpips[v["internalId"]] = enrich_cpips(v, raw)

    return {
        "raterId":          rater_id,
        "appsScriptUrl":    apps_script_url,
        "principles":       DEFAULT_PRINCIPLES,
        "confidenceLevels": CONFIDENCE_LEVELS,
        "confidenceLabels": CONFIDENCE_LABELS,
        "nSessions":        N_SESSIONS,
        "videos":           rater_videos,
        "cpips":            rater_cpips,
        "ratingScaleHtml":  scale_card_html,
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
            "order_index", "session_index", "position_in_session",
            "display_label", "internal_id", "youtube_id",
            "set", "topic", "delay_s", "n_cpips",
        ])
        for i, v in enumerate(rater_videos, start=1):
            n_cpips = len(rater_cpips.get(v["internalId"], []))
            w.writerow([
                i, v["session_index"], v["position_in_session"],
                v["displayLabel"], v["internalId"], v["youtubeId"],
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


def session_summary(rater_videos):
    """Build a compact text summary of the per-rater session layout for logging."""
    by_session = defaultdict(list)
    for v in rater_videos:
        by_session[v["session_index"]].append(v)
    lines = []
    for s_idx in sorted(by_session.keys()):
        items = by_session[s_idx]
        delay_counts = defaultdict(int)
        for v in items:
            if v["set"] == "manipulated":
                delay_counts[v["delay_s"]] += 1
        n_fresh = sum(1 for v in items if v["set"] == "fresh")
        bal = " / ".join(f"{d}s={delay_counts[d]}" for d in DELAYS)
        lines.append(
            f"    Session {s_idx}: {len(items)} videos "
            f"({len(items)-n_fresh} manipulated [{bal}], {n_fresh} fresh)"
        )
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raters", nargs="+", default=["R1", "R2", "R3", "R4"])
    ap.add_argument("--seed", type=int, default=20260602)
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
        print(f"  wrote {out_html.name} ({len(cfg['videos'])} videos):")
        print(session_summary(cfg["videos"]))

    print(f"\nDone. Files written to: {OUT}")
    print("Email each rater ONLY their own rater_R{n}.html file.")
    print("Keep rater_R{n}_order.csv files offline — they contain the de-blinding mapping.")


if __name__ == "__main__":
    main()
