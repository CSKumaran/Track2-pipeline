/**
 * Expert Evaluation — Multimedia Learning (validation tool backend)
 *
 * Implements tool_design_spec.md §10 schema (locked 2026-05-31).
 *  - Ratings tab: one row per CPIP. Columns include support_rating (1-5),
 *    per-CPIP dwell + rewinds, demographics, per-video Q2-Q5.
 *  - RaterProgress tab: live pivot of (rater, video) submissions.
 *  - CurationLog tab: per-CPIP keep/remove decisions and 20% audit (§9).
 *
 * Deployment:
 *   1. Open the target Google Sheet -> Extensions -> Apps Script
 *   2. Paste this file as Code.gs (overwrite any previous version)
 *   3. Run setupSheets() ONCE -- creates / migrates the three tabs
 *   4. Deploy -> New deployment -> Type: Web app
 *      - Execute as: Me
 *      - Who has access: Anyone (required: rater HTML posts without auth)
 *   5. Copy the Web App URL -> paste into validation_tool/config/apps_script_url.txt
 */

const RATINGS_SHEET    = 'Ratings';
const PROGRESS_SHEET   = 'RaterProgress';
const CURATION_SHEET   = 'CurationLog';

const RATINGS_HEADERS = [
  'timestamp_utc',
  'rater_id',
  'rater_institution',
  'rater_years_exp',
  'rater_ctml_fam',
  'rater_qualification',
  'rater_prior_tool_eval',
  'video_internal_id',
  'video_display_label',
  'video_set',
  'video_topic',
  'video_delay_s',
  'video_order_index',
  'session_index',
  'position_in_session',
  'cpip_id',
  'cpip_index',
  'cpip_t_narr_s',
  'cpip_window_start_s',
  'cpip_window_end_s',
  'concept_label',
  'support_rating',
  'cpip_dwell_ms',
  'cpip_rewinds',
  'overall_score',
  'confidence',
  'principles_violated',
  'moment_anchor',
  'video_dwell_ms',
  'video_rewinds',
];

const CURATION_HEADERS = [
  'logged_at_utc',
  'video_internal_id',
  'cpip_id',
  'pipeline_t_narr_s',
  'pipeline_concept_label',
  'curation_status',   // kept / removed_rule1 / removed_rule2 / removed_rule3 / removed_rule4
  'rule_justification',
  'curator',
  'is_audit_sample',   // TRUE / FALSE
  'audit_decision',    // agree / disagree / (blank if not audited)
  'audit_notes',
];

function doPost(e) {
  try {
    const data = JSON.parse(e.postData.contents);
    const ss = SpreadsheetApp.getActive();
    const sheet = getOrCreateSheet(ss, RATINGS_SHEET, RATINGS_HEADERS);

    const ts = new Date().toISOString();
    const dem = data.demographics || {};
    const principles = (data.principlesViolated || []).join(';');
    const cpips = data.cpips || [];

    const rows = cpips.map(c => [
      ts,
      data.raterId || '',
      dem.institution || '',
      dem.yearsExp != null ? dem.yearsExp : '',
      dem.ctmlFamiliarity != null ? dem.ctmlFamiliarity : '',
      dem.qualification || '',
      dem.priorToolEval || '',
      data.videoInternalId || '',
      data.videoDisplayLabel || '',
      data.videoSet || '',
      data.videoTopic || '',
      data.videoDelayS != null ? data.videoDelayS : '',
      data.videoOrderIndex != null ? data.videoOrderIndex : '',
      data.sessionIndex != null ? data.sessionIndex : '',
      data.positionInSession != null ? data.positionInSession : '',
      c.cpipId || '',
      c.cpipIndex != null ? c.cpipIndex : '',
      c.tNarrS != null ? c.tNarrS : '',
      c.windowStartS != null ? c.windowStartS : '',
      c.windowEndS != null ? c.windowEndS : '',
      c.conceptLabel || '',
      c.supportRating != null ? c.supportRating : '',
      c.cpipDwellMs != null ? c.cpipDwellMs : '',
      c.cpipRewinds != null ? c.cpipRewinds : '',
      data.overallScore != null ? data.overallScore : '',
      data.confidence || '',
      principles,
      data.momentAnchor || '',
      data.videoDwellMs != null ? data.videoDwellMs : '',
      data.videoRewinds != null ? data.videoRewinds : '',
    ]);

    if (rows.length) {
      sheet.getRange(sheet.getLastRow() + 1, 1, rows.length, RATINGS_HEADERS.length)
           .setValues(rows);
    }

    return ContentService
      .createTextOutput(JSON.stringify({ ok: true, rowsAppended: rows.length }))
      .setMimeType(ContentService.MimeType.JSON);
  } catch (err) {
    return ContentService
      .createTextOutput(JSON.stringify({ ok: false, error: String(err) }))
      .setMimeType(ContentService.MimeType.JSON);
  }
}

function doGet(e) {
  return ContentService
    .createTextOutput(JSON.stringify({
      ok: true,
      service: 'expert-evaluation-v2',
      time: new Date().toISOString()
    }))
    .setMimeType(ContentService.MimeType.JSON);
}

function getOrCreateSheet(ss, name, headers) {
  let sh = ss.getSheetByName(name);
  if (!sh) sh = ss.insertSheet(name);
  if (sh.getLastRow() === 0) {
    sh.appendRow(headers);
    sh.getRange(1, 1, 1, headers.length).setFontWeight('bold').setBackground('#e5e7eb');
    sh.setFrozenRows(1);
  }
  return sh;
}

/**
 * Run ONCE after pasting / re-pasting this script. Idempotent — safe to re-run.
 * Note: if you are migrating from the Sep-2025 schema, archive the old
 * 'Ratings' tab manually (rename to e.g. 'Ratings_legacy') BEFORE running.
 * This function will not touch an existing Ratings tab's contents; it only
 * adds headers when the tab is empty.
 */
function setupSheets() {
  const ss = SpreadsheetApp.getActive();

  getOrCreateSheet(ss, RATINGS_SHEET, RATINGS_HEADERS);
  getOrCreateSheet(ss, CURATION_SHEET, CURATION_HEADERS);

  let progress = ss.getSheetByName(PROGRESS_SHEET);
  if (!progress) progress = ss.insertSheet(PROGRESS_SHEET);
  if (progress.getLastRow() === 0) {
    progress.getRange('A1').setValue(
      'Auto-populated progress view. Edit the QUERY below if you change Ratings columns.'
    );
    progress.getRange('A3').setFormula(
      "=QUERY(Ratings!A:AD, \"SELECT B, H, I, M, N, MIN(A) WHERE A IS NOT NULL " +
      "GROUP BY B, H, I, M, N ORDER BY B, N, M LABEL MIN(A) 'first_submitted'\", 1)"
    );
  }

  SpreadsheetApp.getUi().alert(
    'Sheets are ready:\n' +
    '  - ' + RATINGS_SHEET + ' (30 columns, one row per CPIP)\n' +
    '  - ' + PROGRESS_SHEET + ' (live pivot, grouped by session)\n' +
    '  - ' + CURATION_SHEET + ' (per-CPIP keep/remove + 20% audit)\n\n' +
    'Next: Deploy -> New deployment -> Type: Web app.'
  );
}
