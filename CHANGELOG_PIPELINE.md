# Pipeline Changes Log
> Track all modifications to verify no regressions against earlier improvements.

---

## Change 001 — Scene-to-Scene OCR Comparison (Previous Session)
**File:** `pipeline/stages/scene_detection.py`
**What:** Changed `new_ocr_words` from frame-to-frame comparison to scene-to-scene comparison.
**Why:** Only 1 word per scene was captured instead of all new words.
**Key code:** `prev_scene_words` set tracks previous SCENE's OCR, not previous frame's.
**Result:** CTML_03_01 mean Δt dropped from 0.74s to 0.20s.

## Change 002 — Word Order Preservation (Previous Session)
**File:** `pipeline/stages/scene_detection.py`
**What:** Preserved original OCR reading order for new words instead of alphabetical sorting.
**Why:** "active knowledge long prior processing term" → "Active Processing Prior Knowledge term Long"
**Key code:** Walk raw OCR tokens in order, keep only those in `new_word_set`.

## Change 003 — NaN Guard in OCR Text (Previous Session)
**File:** `pipeline/stages/scene_detection.py`
**What:** Added `isinstance(text, str)` guard in `_normalize_ocr_words()` and float guard on `current_ocr`.
**Why:** NaN values from cached CSV caused `AttributeError: 'float' object has no attribute 'lower'`.

## Change 004 — Separate OCR/VLM/New Words Columns (Previous Session)
**File:** `pipeline/main.py`, `pipeline/utils/viz_reports.py`
**What:** Split single concept column into 3 separate dashboard columns: New Words, OCR Text, VLM Description.
**Why:** Duplicate OCR text was showing in merged column; no way to see what triggered each track.

## Change 005 — Track A Matched Words Display (Previous Session)
**File:** `pipeline/stages/alignment.py`
**What:** When Track A wins, "Matched Words" column shows Track A matched words instead of Track B/C word window.
**Why:** "Best word match" was showing irrelevant semantic word window even when exact match was used.

## Change 006 — OCR on t_vis Keyframe (Previous Session)
**File:** `pipeline/stages/vlm_concepts.py`
**What:** OCR runs on scene keyframe (extracted at t_vis) so OCR text matches dashboard thumbnail.
**Why:** Frame/OCR timestamp mismatch — OCR was running on mid-scene frame but thumbnail showed t_vis frame.

## Change 007 — CLIP Integration: clip_utils.py (Previous Session)
**File:** `pipeline/utils/clip_utils.py` (NEW)
**What:** Created CLIP ViT-B/16 utilities: image/text embeddings, scene classification, alignment.
**Functions:** `get_clip_image_embedding()`, `get_clip_text_embeddings()`, `classify_scene_type()`, `compute_clip_alignment()`

## Change 008 — CLIP Config Fields (Previous Session)
**File:** `pipeline/config.py`
**What:** Added CLIP configuration: `CLIP_ENABLED`, `CLIP_MODEL_NAME`, `CLIP_MIN_SIM`, `CLIP_ALPHA_LOW`, `CLIP_ALPHA_HIGH`.
**Also:** Added `--no-clip` handling in `config_from_args()`.

## Change 009 — 3-Track Priority Cascade in alignment.py (Previous Session)
**File:** `pipeline/stages/alignment.py`
**What:** Renamed Track B → Track C (semantic), added new Track B (CLIP vision), implemented 3-tier cascade decision logic.
**Columns added:** `scene_type`, `scene_type_conf`, `clip_sim`, `clip_word_window`, `trackB_clip_delta_t`, `trackC_delta_t`.

---

## Change 010 — Track A Temporal Window + Stop-Word Filter (Current Session)
**File:** `pipeline/stages/alignment.py` — `_run_track_a_exact_match()` function
**File:** `pipeline/config.py` — new `TRACK_A_TEMPORAL_WINDOW` field
**What:** Two fixes to Track A exact word matching:
  - (A) Temporal window: only match OCR words to transcript words within ±30s of t_vis
  - (B) Stop-word filter: skip common English function words + require word length ≥ 3
**Why:** Ground truth analysis of A0.mp4 revealed common-word pollution: Scene 1 matched 29 words (incl. "of","the","in") → median Δt=63.3s vs true Δt≈10-14s. Scene 6 had Δt=22.0s vs true Δt≈3.2s.
**Also fixed:** NaN handling bug — `str(NaN)="nan"` was truthy, blocking fallback from empty `new_ocr_words` to `ocr_text`. Added `pd.notna()` guard.
**Regression check:** Does NOT conflict with Change 005 (matched words display). The matched_words list will simply be shorter/more accurate. No changes to scene detection (Changes 001-003 preserved). Scenes 8-13 verified unchanged.
**Verified results (A0.mp4):**
  - Mean |Δt|: 16.57s → **6.63s** (−60%)
  - SD: 21.71s → **7.59s** (−65%)
  - Matched: 16/17 → 16/17 (same)
  - Scene 2: 71.6s → **0.5s**, Scene 4: 29.4s → **−2.6s**
  - Scenes 8-13: NO regression (all unchanged within 0.1s)
