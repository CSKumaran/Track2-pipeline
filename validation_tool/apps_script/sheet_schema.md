# Google Sheet Setup — Expert Evaluation Backend

Backs the rebuilt rater HTML. Schema follows
`CompanionPaper_ValidationTool/tool_design_spec.md` §10 (locked 2026-05-31).
**Replaces** the Sep-2025 zone / delay_s / confidence schema.

## One-time setup (~10 minutes)

1. **Create a new Google Sheet** (or open the existing one). If you are migrating
   from the Sep-2025 schema, rename the existing `Ratings` tab to `Ratings_legacy`
   first so the new schema can be added cleanly alongside.
2. **Extensions → Apps Script** → paste `Code.gs` from this folder, overwriting
   anything already there. Save (name the project e.g. `validation_backend_v2`).
3. In the Apps Script editor, select `setupSheets` from the dropdown and **Run**.
   - On first run, grant authorization to your own account.
   - You should see an alert listing three tabs (`Ratings`, `RaterProgress`,
     `CurationLog`).
4. Back in Apps Script: **Deploy → New deployment**
   - Type: **Web app**
   - Description: `expert eval v2`
   - Execute as: **Me**
   - Who has access: **Anyone**  ← required, raters post without auth
   - Click **Deploy**, grant authorization
5. Copy the **Web app URL** (e.g. `https://script.google.com/macros/s/AKfy…/exec`).
6. Paste it into `validation_tool/config/apps_script_url.txt`.
7. **Test the URL**: open it in a browser — you should see
   `{"ok":true,"service":"expert-evaluation-v2",…}`.

## `Ratings` sheet — 30 columns, one row per CPIP (long format)

| # | Column | Type | Notes |
|---|---|---|---|
| 1 | `timestamp_utc` | ISO string | Server timestamp when the row was appended |
| 2 | `rater_id` | string | `R1`..`R5` |
| 3 | `rater_institution` | string | From demographics gate |
| 4 | `rater_years_exp` | int | Demographics |
| 5 | `rater_ctml_fam` | int | 1–5 |
| 6 | `rater_qualification` | string | Bachelors / Masters / MPhil / PhD / Postdoc / Other |
| 7 | `rater_prior_tool_eval` | string | yes / no |
| 8 | `video_internal_id` | string | True id (`A0`, `F_v3`, …) — NOT shown to rater |
| 9 | `video_display_label` | string | Blinded label (`Video_R07`) |
| 10 | `video_set` | enum | `manipulated` / `fresh` |
| 11 | `video_topic` | string | Base topic label |
| 12 | `video_delay_s` | float | 0 / 1.5 / 5 for manipulated; blank for fresh |
| 13 | `video_order_index` | int | Global position (1..N) in this rater's full sequence across all sessions |
| 14 | `session_index` | int | **NEW** — 1 / 2 / 3 (session this video belongs to in this rater's design) |
| 15 | `position_in_session` | int | **NEW** — 1..N within the session |
| 16 | `cpip_id` | string | Stable pipeline CPIP id |
| 17 | `cpip_index` | int | 1..N within the video |
| 18 | `cpip_t_narr_s` | float | Narration anchor time |
| 19 | `cpip_window_start_s` | float | Viewing-window start (§8) |
| 20 | `cpip_window_end_s` | float | Viewing-window end (§8) |
| 21 | `concept_label` | string | Short label shown to rater |
| 22 | `support_rating` | int | 1–5 — visual-narration support (Q1, single click) |
| 23 | `cpip_dwell_ms` | int | Auto: time card was "active" |
| 24 | `cpip_rewinds` | int | Auto: re-watches of the moment (first watch not counted) |
| 25 | `overall_score` | int | Q2, 0–100 (repeated per CPIP row) |
| 26 | `confidence` | string | Q3, `very_low` / `low` / `medium` / `high` / `very_high` |
| 27 | `principles_violated` | string | Q4, semicolon-joined CTML principles |
| 28 | `moment_anchor` | string | Q5, optional ≤200-char free text |
| 29 | `video_dwell_ms` | int | Total ms on this video |
| 30 | `video_rewinds` | int | Sum of `cpip_rewinds` across the video |

### Session-design schema notes (2026-06-02)

Columns 14 and 15 capture the 3-session block structure introduced on
2026-06-02. Each rater's 23 videos are split into 3 sessions of 7–8
videos each, with topic × delay counterbalanced per session (2-2-2 delay
balance) and across sessions (each topic at each delay exactly once).
For analysis:

- Filter by `session_index` to stratify any per-rater analysis by session
  position (tests for fatigue / practice effects).
- For per-rater stringency check (§11 step 1), aggregate across all
  sessions — the design guarantees each rater sees all 3 delay levels
  for every topic, just in different sittings.

If you are migrating from the pre-session 28-column schema: rename the
existing `Ratings` tab to `Ratings_legacy` before re-running
`setupSheets()`. The new tab will be created with the 30-column header.

## `CurationLog` sheet — per spec §9

Authoritative record of which pipeline-marked CPIPs were kept vs. removed,
plus the 20% audit sub-sample. Filled in by the author (and auditor); the
rater HTML never writes here.

| # | Column | Type | Notes |
|---|---|---|---|
| 1 | `logged_at_utc` | ISO string | When the row was added |
| 2 | `video_internal_id` | string | matches `Ratings.video_internal_id` |
| 3 | `cpip_id` | string | Stable pipeline CPIP id |
| 4 | `pipeline_t_narr_s` | float | Narration anchor as the pipeline produced it |
| 5 | `pipeline_concept_label` | string | Pipeline's concept label |
| 6 | `curation_status` | enum | `kept` / `removed_rule1` / `removed_rule2` / `removed_rule3` / `removed_rule4` |
| 7 | `rule_justification` | string | Free text — why this rule applies |
| 8 | `curator` | string | Initials of author who made the call |
| 9 | `is_audit_sample` | bool | TRUE for the 20% random sub-sample |
| 10 | `audit_decision` | string | `agree` / `disagree` (blank if not audited) |
| 11 | `audit_notes` | string | Optional auditor note |

Use this tab to compute Cohen's κ between author and auditor on the
keep/remove decision (target κ ≥ 0.80 per pre-flight checklist §12).

## `RaterProgress` sheet — live pivot

Auto-computed via QUERY. Shows one row per (rater_id, video_internal_id,
video_display_label, video_order_index) with earliest submission time.
Use this to monitor progress in real time across all raters.

## Analysis export

After data collection, export `Ratings` as CSV and follow spec §11 in order:

1. Per-rater stringency check (mean rating on delay=0 vs delay=5 blocks)
2. Inter-rater reliability: Fleiss' κ on `support_rating`; ICC(2,k) on `overall_score`
3. Pipeline-vs-panel agreement, Fold 1: Cohen's weighted κ
4. SROCC, Fold 1: Spearman ρ on per-video ranks (pre-registered ρ ≥ 0.65)
5. Pipeline-vs-panel agreement + SROCC, Fold 2 (separate AI/ML and non-AI/ML)
6. Confidence-stratified replication of 3 + 4 (`confidence` in {`medium`, `high`, `very_high`})
7. Curation-robustness check (add audit-removed CPIPs back at pipeline scores)
8. Inter-rater principle-checklist concordance (descriptive only)
