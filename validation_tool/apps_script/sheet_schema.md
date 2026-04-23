# Google Sheet Setup — Expert Validation Backend

## One-time setup (≈10 minutes)

1. **Create a new Google Sheet**, name it e.g. `ExpertValidation_Ratings`
2. Open **Extensions → Apps Script**
3. Delete the default `Code.gs` contents, paste contents of `Code.gs` from this folder, click the save icon (name the project e.g. `validation_backend`)
4. In the Apps Script editor, select the `setupSheets` function from the dropdown and click **Run**
   - First run will ask for authorization — grant it (only you, as owner)
   - An alert should pop up: "Sheets set up successfully"
   - Return to the Sheet: you should see two new tabs: `Ratings` (with headers) and `RaterProgress` (with a live QUERY)
5. Back in Apps Script: **Deploy → New deployment**
   - **Type**: Web app
   - **Description**: `validation v1`
   - **Execute as**: Me (your account)
   - **Who has access**: Anyone  ← required, because the rater HTML posts without auth
   - Click **Deploy**, grant authorization
6. Copy the **Web app URL** (looks like `https://script.google.com/macros/s/AKfy…/exec`)
7. Paste it into `validation_tool/config/apps_script_url.txt` (replacing the placeholder), save the file
8. **Test**: open the URL in a browser — you should see `{"ok":true,"service":"expert-validation",…}` (health check)

## Schema — `Ratings` sheet (long format, one row per CPIP)

| Column | Type | Description |
|---|---|---|
| timestamp_utc | ISO string | Server timestamp when row was appended |
| rater_id | string | `R1`..`R4` |
| video_internal_id | string | True id (`A0`, `B3`, `E2`, …) — NOT visible to rater |
| video_display_label | string | Neutral label shown to rater (`Video_R07`) |
| video_order_index | int | Position (1–20) in this rater's sequence |
| cpip_index | int | CPIP position within the video (1–6) |
| cpip_id | string | Stable CPIP id (`A0_c3`) |
| cpip_timestamp_s | float | Seconds into the video where the CPIP occurs |
| concept_label | string | Short label shown to rater |
| zone | string | `Optimal` / `Suboptimal` / `Disruptive` / `Unacceptable` |
| delay_s | float | Perceived delay in seconds (negative = visual leads) |
| confidence | string | `High` / `Medium` / `Low` |
| overall_score | int | 0–100, repeated across all CPIP rows for the video |
| comments | string | Optional, repeated across all CPIP rows |
| client_duration_ms | int | Wall-clock ms spent on this video (informational) |

## Schema — `RaterProgress` sheet (live pivot)

Auto-computed from `Ratings` via QUERY. Shows one row per (rater, video) with earliest submission time. Use this to monitor progress in real time:

- Expected 20 rows per rater (× 4 raters = 80 rows total when complete)
- Sort by `rater_id` then `video_order_index` to see each rater's trajectory

## Analysis export

After data collection, export `Ratings` as CSV and run:

- **Fleiss' κ** across R1–R4 on `zone` column (per cpip_id)
- **Cohen's κ** pipeline zones vs. modal expert zone (per cpip_id)
- **MAE** on `delay_s` vs. ground truth (Set C only)
- **Correlation** pipeline score vs. mean `overall_score` (per video)
