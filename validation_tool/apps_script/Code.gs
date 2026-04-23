/**
 * Expert Validation — Temporal Contiguity Pipeline
 * Google Apps Script backend: receives rating submissions from rater HTML
 * and appends one row per CPIP to the "Ratings" sheet (long format).
 *
 * Deployment:
 *   1. Open the target Google Sheet -> Extensions -> Apps Script
 *   2. Paste this file as Code.gs
 *   3. Run setupSheets() ONCE to create Ratings + RaterProgress tabs with headers
 *   4. Deploy -> New deployment -> Type: Web app
 *      - Execute as: Me
 *      - Who has access: Anyone
 *   5. Copy the Web App URL -> paste into validation_tool/config/apps_script_url.txt
 */

const RATINGS_SHEET = 'Ratings';
const PROGRESS_SHEET = 'RaterProgress';

const RATINGS_HEADERS = [
  'timestamp_utc', 'rater_id', 'video_internal_id', 'video_display_label',
  'video_order_index', 'cpip_index', 'cpip_id', 'cpip_timestamp_s',
  'concept_label', 'zone', 'delay_s', 'confidence',
  'overall_score', 'comments', 'client_duration_ms'
];

function doPost(e) {
  try {
    const data = JSON.parse(e.postData.contents);
    const ss = SpreadsheetApp.getActive();
    const sheet = ss.getSheetByName(RATINGS_SHEET) || ss.insertSheet(RATINGS_SHEET);
    if (sheet.getLastRow() === 0) sheet.appendRow(RATINGS_HEADERS);

    const ts = new Date().toISOString();
    const rows = (data.cpips || []).map((c, idx) => [
      ts, data.raterId, data.videoInternalId, data.videoDisplayLabel,
      data.videoOrderIndex, idx + 1, c.cpipId, c.timestamp, c.conceptLabel,
      c.zone, c.delaySeconds, c.confidence,
      data.overallScore, data.comments || '', data.clientDurationMs || ''
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
  // Simple health check
  return ContentService
    .createTextOutput(JSON.stringify({ ok: true, service: 'expert-validation', time: new Date().toISOString() }))
    .setMimeType(ContentService.MimeType.JSON);
}

/**
 * Run this ONCE after pasting the script to create the sheets with headers.
 */
function setupSheets() {
  const ss = SpreadsheetApp.getActive();

  let ratings = ss.getSheetByName(RATINGS_SHEET);
  if (!ratings) ratings = ss.insertSheet(RATINGS_SHEET);
  if (ratings.getLastRow() === 0) {
    ratings.appendRow(RATINGS_HEADERS);
    ratings.getRange(1, 1, 1, RATINGS_HEADERS.length).setFontWeight('bold').setBackground('#e5e7eb');
    ratings.setFrozenRows(1);
  }

  let progress = ss.getSheetByName(PROGRESS_SHEET);
  if (!progress) progress = ss.insertSheet(PROGRESS_SHEET);
  if (progress.getLastRow() === 0) {
    // Pivot formula: unique (rater_id, video) combos with submission timestamp
    progress.getRange('A1').setValue(
      'Auto-populated progress view. Formula below:'
    );
    progress.getRange('A3').setFormula(
      '=QUERY(Ratings!A:O, "SELECT B, C, D, E, MIN(A) WHERE A IS NOT NULL GROUP BY B, C, D, E ORDER BY B, E LABEL MIN(A) \'first_submitted\'", 1)'
    );
  }

  SpreadsheetApp.getUi().alert('Sheets set up successfully. You can now deploy as Web App.');
}
