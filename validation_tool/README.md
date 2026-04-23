# Expert Validation Tool — Temporal Contiguity Pipeline

Browser-based rating tool for 4 expert raters × 20 videos × 4–6 CPIPs. Implements the Sep-2025-revised Expert Validation Protocol.

## What's in here

```
validation_tool/
├── rater_template.html          # UI template (YouTube player + rating form)
├── generate_rater_copies.py     # builds personalised rater_R{1..4}.html
├── config/
│   ├── videos.json              # 20-video master list (Set C + Set D)
│   ├── cpips.json               # CPIP timestamps + concept labels per video
│   ├── zones_reference_card.html  # cognitive zone definitions modal
│   └── apps_script_url.txt      # Google Apps Script webhook URL
├── apps_script/
│   ├── Code.gs                  # doPost handler — appends rows to Sheet
│   └── sheet_schema.md          # Sheet setup guide + column schema
└── output/
    ├── rater_R{1..4}.html       # files to email to each rater
    └── rater_R{1..4}_order.csv  # offline de-blinding audit trail
```

---

## What YOU need to do (action checklist)

### Before running the generator

- [ ] **Record Set D videos (4 new ones).** Per your decision, Set D = 4 fresh unedited YouTube instructional videos (~4 min each), distinct content, no prior exposure to raters. These can be either your own clips or existing educational content you have rights to re-upload as unlisted.

- [ ] **Upload all 20 videos to YouTube as UNLISTED.** In YouTube Studio: Visibility → Unlisted. Grab the 11-character video ID from the URL (`https://youtu.be/XXXXXXXXXXX`).

- [ ] **Fill in `config/videos.json`** — replace every `REPLACE_YT_ID_*` with the real YouTube ID for that video. Update `duration` (seconds) for each.

- [ ] **Fill in `config/cpips.json`** — this is the big one. For each of the 20 videos, identify **4–6 CPIPs** (Critical Pedagogical Integration Points). A CPIP = a moment where a new concept is introduced and the learner must see the visual while hearing the narration. For each CPIP give:
  - `timestamp` — seconds into the video (float)
  - `conceptLabel` — 3–8 word label shown to the rater (e.g. "Diagram of force vectors")
  - `cpipId` — unique ID per video (already pre-filled, you can keep the pattern)

  The placeholder file has sensible starter CPIPs for all 20 videos — you just need to adjust timestamps and concept labels to match each actual video's content. **For Set C variants of the same topic (e.g. A0/A1/A3/A5), CPIPs should be at the same conceptual moments but their absolute timestamps may differ** since the manipulation shifts narration-visual alignment.

- [ ] **Deploy the Google Apps Script backend** — follow `apps_script/sheet_schema.md` step-by-step (≈10 minutes). Paste the deployed Web App URL into `config/apps_script_url.txt`.

- [ ] **Review `config/zones_reference_card.html`** — confirm the cognitive zone definitions match what you want raters to see. These are deliberately *not* numerical (no seconds mentioned) per protocol §6.

### Run the generator

```bash
cd validation_tool
python generate_rater_copies.py --raters R1 R2 R3 R4
```

Outputs `output/rater_R{1..4}.html` + audit CSVs. Each rater's HTML has a unique randomised order (reproducible via rater-ID-seeded PRNG) with ≥4-video spacing between same-topic videos.

### Before emailing raters

- [ ] **Smoke-test one rater file yourself.** Open `output/rater_R1.html` in Chrome. Confirm:
  - YouTube player loads the first video
  - CPIP timestamp buttons jump the player correctly
  - Q2 (delay) stays disabled until Q1 (zone) is answered
  - Q3 (confidence) stays disabled until Q2 is answered
  - Submitting a video POSTs to the Sheet — check the `Ratings` tab appends 4–6 rows
  - Closing + reopening the file resumes at the next video

- [ ] **Confirm blinding**: grep the HTML for topic leakage (`grep -E "A0|B1|CTML" output/rater_R1.html | grep -v "internalId"` — only `internalId` entries should match).

- [ ] **Collect consent** via the Google Form (separate artifact per protocol §8) before sending the HTML.

- [ ] **Email each rater only their own file** (`rater_R1.html` → Rater 1, etc.). Include the zone reference card instructions and the estimated 3–4 h workload.

- [ ] **Keep `rater_R{n}_order.csv` files offline** — they contain the display_label → internal_id de-blinding map needed to analyse submissions.

### During data collection

- [ ] **Monitor the `RaterProgress` sheet tab** — live pivot of (rater × video) submissions.
- [ ] **After all 4 raters finish**: export `Ratings` as CSV → run κ / MAE / correlation analyses (out of scope for this tool).

---

## Rater-facing UX (what they see)

1. Open single HTML file in browser → progress bar + first incomplete video loads
2. YouTube player (left 60%) + rating panel (right 40%)
3. For each CPIP card:
   - **Jump to moment** button (seeks + plays)
   - Q1: Zone (4 radio options) — rated FIRST
   - Q2: Delay in seconds (unlocks after Q1)
   - Q3: Confidence (unlocks after Q2)
4. After all CPIPs rated: Overall score slider (0–100) + optional comments → Submit
5. On submit: row(s) append to Sheet, state saved, next video loads
6. Can close tab and resume later (localStorage keeps progress)

---

## Design decisions (per protocol §6)

| Decision | Rationale |
|---|---|
| Sequential Q1→Q2→Q3 unlocking | Zone judgement must not be biased by numerical delay estimate |
| Cognitive (not numerical) zone definitions | Prevents number-matching instead of perceptual judgement |
| Blinded `Video_R##` labels + no pipeline scores in UI | Independence of expert judgement |
| ≥4-video same-topic spacing | Reduces order-effect contamination for Set C versions |
| Single HTML file, no install | Zero friction, any browser with internet works |
| localStorage resumability | 3–4 h workload is self-paced over 10 days |
| Long-format sheet (1 row per CPIP) | Easier for κ / MAE analyses than wide format |

---

## Known limitations

- `fetch` with `mode: 'no-cors'` means the HTML cannot read the Apps Script response — it assumes success. Verify via the Sheet after each rater finishes.
- Chrome/Edge recommended; Safari's localStorage behaves slightly differently for `file://` URLs — if raters use Safari, host the HTML on a static URL instead.
- If a rater clears their browser storage mid-session, they lose draft answers (but submitted videos are safe on the Sheet).
