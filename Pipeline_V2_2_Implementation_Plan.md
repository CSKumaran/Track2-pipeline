# Pipeline V2.2 GPU — Implementation Plan

## Context

V2.1 is conceptually strong but has 6 categories of methodological issues identified via expert review:
- **P1 (CRITICAL):** delta_t uses t_start but matching centers on t_keyframe → inflates |delta_t| by 5-40s
- **P2 (CRITICAL):** Monster scenes on GPU (13 scenes, 51-56s max) due to threshold/fusion issues
- **P4+P6:** Alignment σ=15s vs scoring τ=2.5s disconnect → alignment accepts matches scoring zeros out
- **P5+P7+P8:** SigLIP classification near-random; keyword grounding threshold 0.15 too lenient; OCR only at keyframe
- **P9:** No temporal ordering constraint → pedagogically implausible out-of-order matches
- **Importance:** Heuristic-only weights (0.3-2.0 range) inject noise into final score

**Goal:** Create `pipeline_v2_2_gpu` with all fixes + detailed diagnostic outputs at every stage.

## Source: `D:\PhD\Track2_codeClaude_19Mar26\pipeline_v2_1_gpu\` → copy to `pipeline_v2_2_gpu\`

---

## Implementation Order

Phase 0 (scaffold) → Phase 7 (diagnostics infra) → Phase 1 (FIX 1, biggest impact) → Phase 2 (FIX 2, scenes) → Phase 3 (FIX 3, scoring) → Phase 4 (FIX 4, channels) → Phase 5 (FIX 5, monotonicity) → Phase 6 (FIX 6, importance) → Phase 8 (config/CLI) → Phase 9 (dashboard)

---

## Phase 0: Scaffolding

1. Copy `pipeline_v2_1_gpu/` → `pipeline_v2_2_gpu/`
2. Update version in `__init__.py`, logger name in `main.py`, output root default to `outputs_v2_2`
3. Create `utils/diagnostics.py` — `DiagnosticsWriter` class with `write_json()` and `write_csv()`

---

## Phase 1: FIX 1 — Core Timing (P1 CRITICAL)

**File: `stages/alignment.py`**
- Line 52: `t_vis = float(scene["t_start"])` → `t_vis = float(scene["t_keyframe"]) if cfg.USE_KEYFRAME_AS_TVIS else float(scene["t_start"])`
- Lines 70-74, 81-86, 92-97: Add `result["t_start"] = float(scene["t_start"])` alongside existing `result["t_vis"]`
- Line 286 (`_no_match_result`): Add `"t_start"` field

**File: `config.py`**
- Add: `USE_KEYFRAME_AS_TVIS: bool = True`

**Impact:** Single most important fix. All delta_t values immediately become accurate.

---

## Phase 2: FIX 2 — Scene Detection (P2 CRITICAL)

**File: `config.py`** — Add:
- `MAX_SCENE_DURATION: float = 30.0`
- `SCENE_THRESHOLD_K_FALLBACK: float = 1.0` (was 1.5)
- `EXPECTED_SCENES_PER_MINUTE: float = 4.0`

**File: `stages/scene_detection.py`** — 4 changes:

1. **Density-aware threshold guard** (lines 89-96): Replace `len(boundaries) < 3 and duration > 60` with check against `EXPECTED_SCENES_PER_MINUTE` AND `MAX_SCENE_DURATION`. Progressively reduce threshold (0.75×, 0.5×, 0.25×) until max scene duration ≤ limit.

2. **Otsu fallback** (line 218): Use `SCENE_THRESHOLD_K_FALLBACK = 1.0` instead of 1.5.

3. **Force-split long scenes**: New function `_force_split_long_scenes()` called after `_build_scenes()` but before `_smart_merge()`. For scenes > MAX_SCENE_DURATION: find highest DINOv2 distance peak within scene, split there, select new keyframes via centroid, run OCR on new keyframes.

4. **OCR at scene start** (for FIX 4): When `cfg.OCR_SAMPLE_SCENE_START=True`, run OCR on scene start frame AND keyframe, union word sets.

**Diagnostic output:** `diagnostics/stage2_scene_detection.json` — 3 signal arrays, combined signal, threshold, pre/post NMS counts, merge log, split log.

---

## Phase 3: FIX 3 — Search Windows & Scoring (P4, P6)

**File: `config.py`** — Change defaults + add:
- `TEMPORAL_SIGMA: float = 5.0` (was 15.0 — now 2×τ)
- `TRACK_A_TEMPORAL_WINDOW: float = 10.0` (was 30.0)
- `SCORING_MODE: str = "both"` — `"gaussian" | "piecewise" | "both"`
- `PIECEWISE_FLOOR_FACTOR: float = 0.5` — for V2.0-style S_final formula

**File: `stages/scoring.py`** — 3 changes:

1. **New function `piecewise_score(delta_t)`**: V2.0 formula — |d|≤1→100, |d|∈(1,3]→100-15(d-1), |d|∈(3,5]→70-35(d-3), >5→0

2. **Dual scoring mode**: Compute both `S_gaussian` and `S_piecewise` columns. `S_temporal` defaults to gaussian. When mode="both", both columns present.

3. **Floor formula**: Add `S_floored = S_temporal × (floor + (1-floor) × α)`. Add `S_unweighted` column.

4. **Aggregates in results.json**: Report stats for both scoring modes + weighted vs unweighted.

**No changes needed in alignment.py** — `TEMPORAL_SIGMA` and `TRACK_A_TEMPORAL_WINDOW` are already parameterized through cfg.

---

## Phase 4: FIX 4 — Channel Hardening (P5, P7, P8)

**File: `config.py`** — Add:
- `SIGLIP_CLASSIFY_ENABLED: bool = False` (disable zero-shot by default)
- `SIGLIP_CLASSIFY_MIN_CONF: float = 0.65` (raised from 0.55)
- `SIGLIP_KEYWORD_MIN_SIM: float = 0.30` (was hardcoded 0.15)
- `OCR_SAMPLE_SCENE_START: bool = True`
- `OCR_SPELLCHECK_ENABLED: bool = True`
- `SIGLIP_WEIGHT_WITH_OCR: float = 0.5`

**File: `stages/vlm_concepts.py`** (lines 55-74):
- When `SIGLIP_CLASSIFY_ENABLED=False`: simple heuristic — has OCR text → content, else → non-content
- When enabled: raise threshold to 0.65, fallback to heuristic below
- Diagnostic: capture all 13 SigLIP label scores per scene

**File: `utils/ocr_utils.py`**:
- Add `spellcheck_ocr_words(words, cfg)` using symspellpy or bundled wordlist
- Gate behind `OCR_SPELLCHECK_ENABLED`

**File: `stages/keyword_analysis.py`** (line 343):
- `if best_sim > 0.15:` → `if best_sim > cfg.SIGLIP_KEYWORD_MIN_SIM:`

**File: `stages/alignment.py`** (`_track_b`):
- When scene has OCR words: `best_sim *= cfg.SIGLIP_WEIGHT_WITH_OCR` (down-weight SigLIP when OCR exists)

---

## Phase 5: FIX 5 — Monotonic Ordering (P9)

**File: `config.py`** — Add:
- `MONOTONIC_CHECK_ENABLED: bool = True`
- `MONOTONIC_SLACK_MIN: float = 10.0`
- `MONOTONIC_RERUN_VIOLATORS: bool = False`

**File: `stages/alignment.py`** — New function `_check_monotonicity(results, scenes_df, cfg)`:
- Walk sorted scenes: flag if `t_narr[i+1] < t_narr[i] - slack`
- Add `monotonic_violation: bool` column
- Optional: re-run violating scenes with halved sigma
- Diagnostic: `diagnostics/stage4b_monotonic_violations.json`

---

## Phase 6: FIX 6 — Importance Weights

**File: `config.py`** — Add:
- `IMPORTANCE_HEURISTIC_WEIGHTS: dict = {1: 0.8, 2: 0.9, 3: 1.0, 4: 1.1, 5: 1.2}`

**File: `stages/scoring.py`**:
- Check `pedagogical_importance.csv` backend column
- If `"heuristic"` → use compressed weights `IMPORTANCE_HEURISTIC_WEIGHTS`
- If LLM-backed → use full range `IMPORTANCE_WEIGHTS`
- Report both `mean_S_unweighted` and `mean_S_weighted` in results.json

---

## Phase 7: Diagnostics Infrastructure

**New file: `utils/diagnostics.py`**
- `DiagnosticsWriter(output_dir)` — creates `diagnostics/` subfolder
- Methods: `write_json(filename, data)`, `write_csv(filename, df)`

**File: `config.py`**: Add `DIAGNOSTICS_ENABLED: bool = True`

**File: `main.py`**: Instantiate writer, pass `diag=diag` to each stage

**Diagnostic files per stage:**
| Stage | File | Contents |
|-------|------|----------|
| 1 | `stage1_asr_summary.json` | n_words, n_flagged, pct_unreliable, flag_reasons |
| 2 | `stage2_scene_detection.json` | 3 signals, combined, threshold, NMS counts, merge/split log |
| 3 | `stage3_vlm_concepts.json` | Per-scene: all SigLIP scores (13), OCR text, classification method |
| 4b | `stage4b_alignment.json` | Per-scene per-track: attempted tracks, similarities, success/fail reason |
| 5 | `stage5_keywords.json` | Per-keyword: cascade trace (steps tried, result at each) |
| 6 | `stage6_importance.json` | Raw heuristic scores, percentile thresholds, per-segment raw vs binned |
| 7 | `stage7_scoring.json` | Both Gaussian + piecewise, zone distributions, weighted vs unweighted |

---

## Phase 8: Config & CLI Updates

**File: `main.py`** — New CLI arguments:
```
--scoring-mode {gaussian,piecewise,both}    (default: both)
--temporal-sigma FLOAT                       (default: 5.0)
--max-scene-duration FLOAT                   (default: 30.0)
--no-diagnostics
--no-monotonic-check
--use-v21-timing                             (revert to t_start for delta_t)
```

---

## Phase 9: Dashboard Updates

**File: `utils/viz_reports.py`**:
- Show both Gaussian and piecewise scores when mode="both"
- Monotonic violation flags as warning markers
- Diagnostics summary panel

---

## Critical Files (ordered by change complexity)

| File | Changes | Phases |
|------|---------|--------|
| `config.py` | ~20 new/changed params | All |
| `stages/alignment.py` | t_vis fix, monotonicity, SigLIP down-weight, diagnostics | 1, 4, 5 |
| `stages/scene_detection.py` | Threshold, force-split, OCR at start, diagnostics | 2, 4 |
| `stages/scoring.py` | Dual scoring, floor formula, weight compression, diagnostics | 3, 6 |
| `stages/vlm_concepts.py` | Heuristic classification, diagnostics | 4 |
| `stages/keyword_analysis.py` | SigLIP threshold configurable | 4 |
| `utils/ocr_utils.py` | Spell-check post-correction | 4 |
| `utils/diagnostics.py` | NEW file | 7 |
| `main.py` | CLI args, diagnostics wiring | 0, 8 |
| `utils/viz_reports.py` | Dashboard updates | 9 |

---

## Verification Plan

1. **Unit check after Phase 1:** Run on A0, confirm delta_t values are much smaller (close to 0 for Track A matches)
2. **After Phase 2:** Verify scene count ≥ 16, max duration ≤ 30s, check diagnostics for signal arrays
3. **After Phase 3:** Compare Gaussian vs piecewise scores — piecewise should show score=70 at d=3s (not 48.7)
4. **After all phases:** Run on all 7 videos, compare V2.1 vs V2.2 results.json side by side
5. **Cross-version sanity:** V2.2 scores should be closer to V2.0 (~65) than V2.1 (~6-50)
6. **Diagnostics review:** Open each diagnostic JSON, verify all intermediate values are captured

---

## V2.1 Algorithm Reference (preserved below for comparison)

## Stage 2: Scene Detection (Multi-Signal Fusion)

**Purpose:** Detect visual scene boundaries by fusing 3 complementary signals.

**Input:** Video file

### Step 2.1: Frame Extraction
- Sample at `SAMPLE_INTERVAL = 0.5s` (2 fps)
- ffmpeg with quality `-q:v 2`

### Step 2.2: Signal A — PySceneDetect (weight 0.35)
- `AdaptiveDetector(window_width=5)`
- Returns per-frame binary signal: 1.0 at boundaries, 0.0 elsewhere

### Step 2.3: Signal B — DINOv2 Embedding Distance (weight 0.45)
- Model: `facebook/dinov2-base` (768-dim CLS token)
- Batch size: 16
- Consecutive cosine distance: `1 - cos_sim(frame[i], frame[i+1])`
- Pad first frame with 0, normalize to [0, 1]

### Step 2.4: Signal C — OCR Jaccard Distance (weight 0.20)
- Run OCR every ~1.5s (every 3rd frame at 0.5s interval)
- Engine: EasyOCR or PaddleOCR (GPU-accelerated)
- Extract words per frame, compute Jaccard distance between consecutive samples:
  ```
  J(A, B) = 1 - |A ∩ B| / |A ∪ B|
  ```
- Only flag if distance > `OCR_JACCARD_THRESHOLD = 0.5`
- Spread signal across skipped frames, normalize to [0, 1]

### Step 2.5: Signal Fusion
- If OCR signal has non-zero values:
  ```
  combined = 0.35 * A + 0.45 * B + 0.20 * C
  ```
- If OCR all-zero (no text detected):
  ```
  combined = 0.4375 * A + 0.5625 * B    (reweighted to sum to 1)
  ```

### Step 2.6: Adaptive Threshold (Otsu-like)
- Try 50 candidate thresholds from min to max of combined signal
- For each: compute between-class variance `w0 * w1 * (μ0 - μ1)²`
- Pick threshold with maximum variance
- Fallback: `mean + 1.5 * std`
- Guard: if < 3 boundaries and video > 60s, lower threshold by `0.5 * std`

### Step 2.7: Boundary Detection + NMS
- Find frames where combined signal ≥ threshold
- Group adjacent frames (distance ≤ 2)
- Keep peak score in each group

### Step 2.8: Scene Construction
- Scenes: intervals between consecutive boundaries
- **Keyframe selection:** DINOv2 centroid of inner 80% of frames (avoids first/last 10%)
  - Compute mean embedding, find closest real frame
- Run OCR on keyframe to get scene's text content
- Record: t_start, t_end, duration, t_keyframe, keyframe_path, ocr_words

### Step 2.9: Smart Merge
- If scene duration < `2.0s` AND DINOv2 similarity with predecessor > `0.85`:
  - Merge into predecessor (extend t_end)
- Re-number scene IDs

**Output:**
| File | Columns |
|------|---------|
| `scenes.csv` | scene_id, t_start, t_end, duration, t_keyframe, keyframe_path, ocr_words, n_ocr_words |
| `ocr_per_frame.csv` | frame_time, words, n_words, mean_confidence |
| `dinov2_distances.csv` | frame_time, distance |

---

## Stage 3: Visual Concept Extraction

**Purpose:** Label each scene with its visual content (OCR text + frame type classification).

**Input:** scenes.csv + keyframe images

### Step 3.1: OCR on Keyframe
- If `OCR_ENABLED=True`: run OCR engine on keyframe
- Concatenate detected words → `ocr_text`

### Step 3.2: VLM Description (optional)
- Mode: `"skip"` (current), `"ollama"`, or `"gemini"`
- Prompt: "Describe the educational content visible in this frame..."
- If skip: `vlm_text = ""`

### Step 3.3: SigLIP Frame Type Classification
- Model: `ViT-B-16-SigLIP` (pretrained: webli, 512-dim embeddings)
- **Content labels:** "content slide", "diagram", "code editor", "whiteboard", "animation frame", "demonstration"
- **Non-content labels:** "title slide", "logo screen", "blank screen", "loading screen", "transition effect", "section divider", "talking head with no visual aids"
- Similarity: `sigmoid(image_emb @ text_emb.T)` (per-label independent probability)
- Best label: argmax over all labels
- `is_content = True` if best label is in content set
- **Low-confidence override:** if confidence < `0.55` → default to content
- **OCR override:** if any OCR text detected → force `is_content = True`

### Step 3.4: Build concept_text
- Combine: `ocr_text + " " + vlm_text` (trimmed)

**Output:**
| File | Columns |
|------|---------|
| `scene_concepts.csv` | scene_id, ocr_text, vlm_text, concept_text, is_content, frame_type, frame_type_confidence, vlm_backend |

---

## Stage 4a: Text Unit Embeddings

**Purpose:** Pre-compute embeddings for all transcript segments (reused by alignment).

**Input:** transcript_segments_improved.csv

### Algorithm:
1. Compute segment midpoints: `t_mid = (start_time + end_time) / 2`
2. Embed all segment texts with **BGE-large-en-v1.5** (1024-dim)
   - Prefix: `"Represent this sentence: "` + text
   - L2 normalize
   - Batch size: 4

**Output:**
| File | Content |
|------|---------|
| `segment_meta.csv` | segment_id, text, start_time, end_time, t_mid |
| `segment_embeddings.npy` | (N_segments, 1024) float32 array |

---

## Stage 4b: Narration Alignment (3-Track Cascade)

**Purpose:** For each scene, find the best-matching narration segment. This is the core of temporal contiguity measurement.

**Input:** scenes.csv + scene_concepts.csv + transcript_words.csv + segment_meta.csv + segment_embeddings.npy

### Cascade Logic:
```
For each scene:
  if non-content → skip (no_match)
  Try Track A → if matched, done
  Try Track B → if matched, done
  Try Track C → if matched, done
  Else → no_match
```

### Track A: Exact OCR Word Matching (highest confidence)

1. Extract OCR words from scene (length ≥ 3, not stop words)
2. If no OCR words → skip to Track B
3. **Search window:** transcript words within ±30s of **t_keyframe**
4. For each OCR word: find exact normalized match in transcript
   - Normalize: `re.sub(r'[^a-z0-9]', '', word.lower())`
   - Record offset: `t_word - t_keyframe`
5. If no matches → skip to Track B
6. **Context validation:** embed concept_text, check cosine similarity with closest segment ≥ `0.25`
7. **t_narr** = t_keyframe + median(all offsets)
8. **alpha = 1.0** (maximum confidence for exact word match)

### Track B: SigLIP Vision-to-Text (medium confidence)

1. Embed keyframe with SigLIP vision encoder → (1, 512)
2. Embed all transcript segments with SigLIP text encoder → (N, 512)
3. Compute sigmoid similarities (not softmax — per-pair independent)
4. Apply **temporal Gaussian decay** centered on t_keyframe:
   ```
   weight(t) = exp(-0.5 * ((t_mid - t_keyframe) / σ)²)     σ = 15.0s
   ```
   - At ±15s: weight = 0.607
   - At ±30s: weight = 0.135
   - At ±45s: weight = 0.011
5. Weighted similarity = raw_sim × temporal_weight
6. Best segment = argmax(weighted similarity)
7. If raw similarity < `0.20` → skip to Track C
8. **t_narr** = t_mid of best segment
9. **Alpha:**
   ```
   α = 0.5 + 0.5 × (sim - 0.20) / (0.60 - 0.20)
   α = clamp(α, 0.5, 1.0)
   ```

### Track C: Semantic Text Similarity (lowest confidence)

1. Embed concept_text with BGE-large → (1, 1024)
2. Cosine similarity with pre-computed segment embeddings
3. Apply same temporal Gaussian decay (σ = 15.0s) centered on t_keyframe
4. Best segment = argmax(weighted similarity)
5. If raw similarity < `0.20` → no match
6. **t_narr** = t_mid of best segment
7. **Alpha:**
   ```
   α = (sim - 0.30) / (0.80 - 0.30)
   α = clamp(α, 0.0, 1.0)
   ```

### delta_t Computation (the wrapper):
After any track returns a result:
```python
t_vis = scene["t_start"]          # scene start time
delta_t = t_narr - t_vis          # OVERWRITES track's internal delta_t
```
**⚠️ KNOWN BUG:** All tracks match relative to t_keyframe, but delta_t is recalculated relative to t_start. For long scenes, t_start << t_keyframe, inflating delta_t massively.

**Output:**
| File | Columns |
|------|---------|
| `alignment_events.csv` | scene_id, t_vis, t_keyframe, concept_text, match_type, match_track, t_narr, delta_t, n_word_matches, siglip_sim, semantic_sim, alpha, scene_type, scene_type_conf |

---

## Stage 5: Keyword Extraction & Visual Grounding

**Purpose:** Extract keywords from narration, then find where each keyword appears visually in the video. Provides fine-grained temporal contiguity at keyword level.

**Input:** transcript_segments_improved.csv, transcript_words.csv, ocr_per_frame.csv, scenes.csv

### Step 5a: Keyword Extraction

**spaCy pipeline:**
1. Load `en_core_web_sm`
2. Extract noun chunks (lemmatized, stripped of determiners)
3. Extract named entities (types: WORK_OF_ART, LAW, LANGUAGE, NORP, ORG, PRODUCT)
4. Extract standalone nouns (POS=NOUN) not already in chunks
5. Filter: length ≥ 3, not in stop words

**KeyBERT enrichment** (if `KEYWORD_USE_KEYBERT=True`):
6. Extract top-5 keyphrases using BGE embeddings
7. N-gram range: (1, 3)
8. MMR diversity: 0.5

**Merge & deduplicate:**
9. If single word is substring of multi-word phrase → keep only phrase

### Step 5b: Keyword Timestamp Assignment
- Search transcript words within [segment_start - 1s, segment_end + 1s]
- Find first word matching keyword's first token
- Return word's start_time as `t_narr`
- Fallback: segment midpoint

### Step 5c: Groundability Classification

| Condition | Groundability |
|-----------|--------------|
| In VISUAL_NOUNS (diagram, chart, code, formula...) | HIGH |
| In ABSTRACT_WORDS (understanding, concept, method...) | at most MEDIUM |
| Concreteness score ≥ 4.0 | HIGH |
| Concreteness score ≥ 2.5 | MEDIUM |
| Concreteness score < 2.5 | LOW |
| Multi-word phrase | at least MEDIUM |

- **LOW groundability → skip grounding** (marked `is_visual=False`)

### Step 5d: 4-Step Grounding Cascade

**Step 1: OCR Fuzzy Search** (confidence: HIGH)
- Window: ±60s around t_narr
- Fuzzy match: Levenshtein ratio > `0.80`
- Multi-word: ALL component words must match
- Pick closest frame to t_narr
- If matched → done, skip steps 2-4

**Step 2: GroundingDINO** (confidence: MEDIUM, disabled by default)
- Window: ±30s around t_narr
- Object detection on nearby keyframes
- Query augmentation: "keyword label or keyword diagram"
- Box threshold: 0.25, text threshold: 0.25
- If matched → persistence check → done

**Step 3: SigLIP Contextual** (confidence: LOW)
- Window: ±30s around t_narr
- Embed images of nearby keyframes
- Text query: "A visual showing {keyword}"
- Sigmoid similarity with temporal Gaussian decay (σ = 15s)
- Threshold: 0.15
- If matched → persistence check → done

**Step 4: VLM Existence Check** (confidence: LOW, only for HIGH groundability)
- Only if `VLM_MODE != "skip"`
- Query: "Does this frame visually depict '{keyword}'? Answer YES or NO."
- Flag for review

### Step 5e: Segment-Level Aggregation
- Confidence weights: HIGH=1.0, MEDIUM=0.7, LOW=0.4, VERY_LOW=0.2
- Weighted median of delta_t values per segment
- delta_t for keywords: `t_vis(grounded frame) - t_narr(keyword)`

**Output:**
| File | Key Columns |
|------|-------------|
| `keyword_alignment.csv` | keyword_id, keyword_text, segment_id, t_narr, t_vis, delta_t, method, confidence, groundability, is_visual |
| `segment_keyword_scores.csv` | segment_id, n_keywords, n_grounded, delta_t_weighted_median |

---

## Stage 6: Pedagogical Importance Rating

**Purpose:** Rate each transcript segment's educational importance (1-5) to weight the final score.

**Input:** transcript_segments_improved.csv

### 3-Tier Backend (auto-fallback):

**Tier 1: Gemini API** (if API key available)
- Send all segments to Gemini in one prompt
- Request JSON: `[{segment_id, importance: 1-5, reason}]`
- Optional double-run: run twice, flag if disagreement > 1 level
- Backend: `"gemini"`

**Tier 2: Local LLM via Ollama** (if Ollama running)
- Same prompt format as Gemini
- Model: `llava:7b`
- Backend: `"local_llm"`

**Tier 3: Heuristic** (always available)
- Score = `0.4 × keyword_density + 0.3 × speech_rate + 0.3 × word_count_norm`
- Map to 1-5 via percentile bins (20th, 40th, 60th, 80th)
- Backend: `"heuristic"`

### Importance Scale:
```
1 = Low:          intro/outro, greetings, filler
2 = Below Avg:    recap, tangential examples
3 = Average:      supporting explanation, context
4 = Above Avg:    key concept intro, important examples
5 = Critical:     core derivation, formula, definition
```

**Output:**
| File | Columns |
|------|---------|
| `pedagogical_importance.csv` | segment_id, importance, reason, backend, is_reliable |

---

## Stage 7: Scoring & Aggregation

**Purpose:** Compute temporal contiguity scores from delta_t values.

**Input:** alignment_events.csv, keyword_alignment.csv, pedagogical_importance.csv

### Temporal Scoring Formula:
```
S_temporal = 100 × exp(-0.5 × (|delta_t| / τ)²)     τ = 2.5s
```

| delta_t | S_temporal | Zone |
|---------|-----------|------|
| 0.0s | 100.0 | Optimal |
| 0.5s | 98.0 | Optimal |
| 1.0s | 92.3 | Optimal (≤1s) |
| 2.0s | 72.6 | Suboptimal |
| 2.5s | 60.7 | Suboptimal |
| 3.0s | 48.7 | Disruptive (≤5s) |
| 4.0s | 27.8 | Disruptive |
| 5.0s | 13.5 | Unacceptable (>5s) |
| 7.0s | 1.6 | Unacceptable |
| 10.0s | 0.0003 | Unacceptable |

### Zone Classification:
```
|delta_t| ≤ 1.0s  → Optimal
|delta_t| ≤ 3.0s  → Suboptimal
|delta_t| ≤ 5.0s  → Disruptive
|delta_t| > 5.0s  → Unacceptable
```

### Importance-Weighted Score:
```
S_weighted = S_temporal × importance_weight
```
Where importance_weight = {1: 0.3, 2: 0.6, 3: 1.0, 4: 1.5, 5: 2.0}

### Priority (which scenes to fix first):
```
priority = (100 - S_temporal) × importance_weight × alpha
```

### Overall Score:
```
overall_score = Σ(S_temporal × alpha) / Σ(alpha)     (alpha-weighted mean)
```

### Overall Grade:
```
≥ 80  → Excellent
≥ 60  → Good
≥ 40  → Acceptable
≥ 20  → Poor
< 20  → Unacceptable
```

**Output:**
| File | Content |
|------|---------|
| `scores_per_scene.csv` | Per-scene: S_temporal, zone, all alignment columns |
| `scores_weighted.csv` | Per-scene: + importance_weight, S_weighted, priority |
| `results.json` | Aggregates: mean/median scores, zone %, grade, keyword stats |

---

## Dashboard (report_dashboard.html)

- Summary cards: overall score, grade, zone percentages
- Per-scene table with: keyframe thumbnail, t_vis, t_narr, delta_t, S_temporal, zone (color-coded), match track
- Zone color coding: Green (Optimal), Yellow (Suboptimal), Orange (Disruptive), Red (Unacceptable)

---

## Known Issues (from analysis)

| # | Issue | Severity | Location |
|---|-------|----------|----------|
| P1 | delta_t uses t_start but tracks match at t_keyframe | CRITICAL | alignment.py:68-70 |
| P2 | Scene detection creates monster scenes (51-56s) on GPU | CRITICAL | scene_detection.py |
| P3 | OCR captured at keyframe only, not scene start | HIGH | scene_detection.py:303 |
| P4 | Gaussian scoring harsher than v2.0 piecewise at d=2-4s | HIGH | scoring.py:14-19 |
| P5 | SigLIP classification all ~0.51 confidence (useless) | MEDIUM | vlm_concepts.py |
| P6 | TEMPORAL_SIGMA=15s >> SCORE_TAU=2.5s disconnect | MEDIUM | config.py |
| P7 | SigLIP contextual grounding threshold 0.15 too lenient | MEDIUM | keyword_analysis.py |
| P8 | OCR text quality (Iocal → local, garbled text) | LOW-MED | ocr_utils.py |
| P9 | No temporal ordering constraint across scenes | LOW | alignment.py |
