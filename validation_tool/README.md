# Expert Evaluation Tool — Multimedia Learning

Browser-based rating tool for the IEEE TLT expert validation of the
temporal-contiguity pipeline.

**Authoritative spec:** `../CompanionPaper_ValidationTool/tool_design_spec.md`
(locked 2026-05-31). This rebuild **supersedes the Sep-2025 protocol** (CPIP
zone + delay-seconds + confidence) and the bare 2026-05-29 supervisor sketch
(per-video score + confidence only).

Per spec: ≥4 raters (5 preferred) × ≥28 videos (18 manipulated Fold 1 +
≥10 fresh Fold 2). Multi-session via Google-Sheets-backed HTML form, ~3 h
per rater.

## What's in here

```
validation_tool/
├── rater_template.html              # UI template (demographics gate + per-CPIP + per-video form)
├── generate_rater_copies.py         # builds personalised rater_R{n}.html
├── config/
│   ├── videos.json                  # video manifest (manipulated + fresh sets)
│   ├── cpips.json                   # per-video CPIPs (t_narr_s + conceptLabel)
│   ├── rating_scale_reference.html  # 1-5 perception anchors (no timing language)
│   └── apps_script_url.txt          # Google Apps Script webhook URL
├── apps_script/
│   ├── Code.gs                      # doPost handler — writes the §10 schema
│   └── sheet_schema.md              # tab/column reference + deployment steps
└── output/
    ├── rater_R{n}.html              # files to email to each rater
    └── rater_R{n}_order.csv         # offline de-blinding audit trail
```

## Instrument summary (per spec §4–§7)

- **Demographics gate** (once at session start): institution, years of EdTech
  experience, CTML familiarity (1–5), highest qualification, prior exposure
  to evaluating automated tools (yes/no).
- **Per CPIP** — single question (§4): *"At this moment, how well does the
  visual support what's being said?"* — one click on a 1–5 button bar
  (Very poorly → Very well). **No** "delay" or "temporal contiguity"
  wording in the rater UI. Time-on-CPIP and rewinds are auto-tracked.
- **Per video** (§5):
  - Q2 — overall presentation quality (0–100 slider)
  - Q3 — confidence in Q2 (5-point Likert: very low → very high)
  - Q4 — multimedia principles violated (checkbox list, multi-select)
  - Q5 — one specific moment that drove the score (optional ≤200-char text)
- **Viewing windows** (§8): clicking "Watch this moment" seeks the YouTube
  player to `t_narr - 5 s` and auto-pauses at the end of the window:
  - manipulated set: window length **12 s** (t-5 to t+7)
  - fresh set: window length **15 s** (t-5 to t+10)

## What YOU need to do (action checklist)

### Before running the generator

- [ ] **Lock the video list.** Spec §3 + §13 open items: 18 manipulated
  (6 base topics × 3 delays: 0 s / 1.5 s / 5 s) + ≥10 fresh (with ≥2 non-AI/ML).
- [ ] **Upload all videos to YouTube as UNLISTED.** Grab the 11-character ID
  from each URL (`https://youtu.be/XXXXXXXXXXX`).
- [ ] **Fill in `config/videos.json`** — replace every `REPLACE_YT_ID_*` with
  the real id. Update `duration`, `topic`, and `delay_s` fields. Each entry
  must declare `"set": "manipulated"` or `"set": "fresh"`. `topicKey` is
  used by the generator to enforce ≥4-video spacing for same-topic items.
- [ ] **Fill in `config/cpips.json`** — for each video, list its curated
  CPIPs. Each entry: `cpipId` (unique), `t_narr_s` (seconds), `conceptLabel`
  (3–8 word label shown to the rater). Optional `windowStartS` /
  `windowEndS` override the §8 defaults.
- [ ] **Curate CPIPs per spec §9.** Apply the pre-committed remove rules
  (technical-error only), log every decision in the `CurationLog` sheet tab,
  flag a 20% random sub-sample for the auditor.
- [ ] **Deploy the Apps Script backend** — follow `apps_script/sheet_schema.md`
  step-by-step (~10 minutes). Paste the deployed Web App URL into
  `config/apps_script_url.txt`.
- [ ] **Review `config/rating_scale_reference.html`** — confirm the 1–5
  perception anchors match what you want raters to see (the modal is
  accessible from the header of the rater UI).

### Run the generator

> ⚠️ All Python in this project runs on the IIT Jodhpur HPC, not Windows.
> Push the folder to HPC and run inside the `tc_pipeline` conda env:

```bash
# on HPC
cd /iitjhome/senthil1/validation_tool
conda activate tc_pipeline
python generate_rater_copies.py --raters R1 R2 R3 R4 R5
```

Outputs `output/rater_R{n}.html` + audit CSVs. Each rater's HTML has a
unique seed-derived randomised order with ≥4-position spacing between
same-`topicKey` videos.

### Before emailing raters

- [ ] **Smoke-test one rater file.** Open `output/rater_R1.html` in Chrome.
  Confirm:
  - Demographics screen appears first; cannot proceed without it.
  - YouTube player loads the first video; title shows `Video_R01`.
  - "Watch this moment" seeks to `t_narr - 5 s` and auto-pauses at the
    correct end (12 s for manipulated, 15 s for fresh).
  - 1–5 button bar registers exactly one click per CPIP.
  - Per-video form (Q2–Q5) unlocks once every CPIP has a rating;
    Submit unlocks once Q3 (confidence) is also picked.
  - Submitting POSTs to the Sheet — the `Ratings` tab gains
    (#CPIPs) new rows with the full §10 column set, and `RaterProgress`
    auto-pivots to show this rater × this video.
  - Closing and re-opening resumes at the next video without re-prompting
    for demographics.
- [ ] **Confirm blinding.** `grep -E "M_A|M_B|F_AI" output/rater_R1.html`
  should only match `internalId` entries inside the embedded config, never
  user-facing strings.
- [ ] **Collect consent** via the separate Google Form (spec §12) before
  sending the HTML.
- [ ] **Email each rater only their own file** (`rater_R1.html` → Rater 1).
  Include the rating-scale reference (also bundled inside the HTML modal)
  and the ~3 h workload estimate, plus the multi-session note.
- [ ] **Keep `rater_R{n}_order.csv` files offline** — they hold the
  display-label → internal-id de-blinding map.

### During data collection

- [ ] Monitor `RaterProgress` for live (rater × video) submission state.
- [ ] After all raters finish: export `Ratings` as CSV, run the spec §11
  analysis plan in order (per-rater stringency → reliability → pipeline-vs-panel
  → SROCC → confidence-stratified → curation-robustness → principle concordance).

## Rater-facing UX

1. Open the single HTML file in any modern browser → demographics gate (~2 min).
2. After submit, main view: YouTube player (left 60%) + rating panel (right 40%).
3. For each "moment":
   - Click **Watch this moment** → player seeks to `t_narr − 5 s` and
     auto-pauses at the window end.
   - Click one of the five buttons (1–5) — single click, no follow-ups.
   - Re-watch as many times as needed; re-watches are auto-counted.
4. After every moment has a rating, the per-video Q2–Q5 form unlocks:
   - 0–100 overall slider
   - 5-point confidence
   - CTML-principle checkbox list (multi-select, may leave blank)
   - Optional ≤200-char free-text moment anchor
5. Submit → row(s) appended to the Google Sheet → next video loads.
6. localStorage keeps draft progress; raters can close and resume.

## Design decisions

| Decision | Source |
|---|---|
| Single 1–5 click per CPIP (no zone/delay/confidence trio) | Spec §4 |
| No "delay" / "temporal contiguity" wording in rater UI | Spec §4 |
| Demographics gate at session start | Spec §6 |
| Per-video Q2 (slider) + Q3 (confidence) + Q4 (CTML checklist) + Q5 (anchor) | Spec §5 |
| Implicit calibration via 0 s / 5 s extremes — no separate calibration phase | Spec §7 |
| Asymmetric viewing windows by set | Spec §8 |
| Blinded `Video_R##` labels + no pipeline scores in UI | Spec §2/§3 (blinding) |
| ≥4-video same-`topicKey` spacing | Reduces dose-set order effects |
| Long-format Sheet (1 row per CPIP, demographics repeated) | Spec §10 |
| Author-curated CPIPs via pre-committed technical-error rules + 20% audit | Spec §9 |

## Known limitations

- `fetch` uses `mode: 'no-cors'`, so the HTML cannot read the Apps Script
  response. The UI assumes success on a thrown-free POST. Verify each
  rater's progress in the Sheet's `RaterProgress` tab.
- Chrome / Edge recommended. Safari `localStorage` behaves slightly
  differently for `file://` URLs — host on a static URL if raters use Safari.
- If a rater clears browser storage mid-session, draft answers for the
  in-progress video are lost (submitted videos are safe on the Sheet).
- The auto-pause uses a 250 ms `setInterval` polling `getCurrentTime()`,
  so the actual pause may land 0–250 ms past `window_end_s`. Acceptable
  for the construct.

## Migration from the Sep-2025 schema

If a previous Sheet exists with the old `zone` / `delay_s` / `confidence`
columns: rename that tab to `Ratings_legacy` before running the new
`setupSheets()`, so the new 28-column `Ratings` tab can coexist without
disturbing legacy data.
