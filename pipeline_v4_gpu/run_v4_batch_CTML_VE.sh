#!/bin/bash
#SBATCH --job-name=v4_VE_CTML
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=v4_VE_CTML_%j.log
#SBATCH --error=v4_VE_CTML_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

# ═══════════════════════════════════════════════════════════════════════════════
# V4: Video_evaluation (01-04) + CTML_03 (01-04) — Full Pipeline
#
# 4 conditions (same for both sets):
#   01 = Simultaneous     (visual + narration together)     — Expected BEST
#   02 = Animation first  (visual appears before narration) — Expected 2nd-3rd
#   03 = Narration first  (narration plays before visual)   — Expected WORST
#   04 = Segmented        (visual and narration in chunks)  — Expected 2nd-3rd
#
# V3 problem: Animation first (02) scores 100.0 because visual_on_screen=True
# with delta_t<0 gives S=100 regardless of dwell time. V4 baseline first,
# then dwell decay scoring to fix this.
#
# PREREQUISITE: vLLM server for Tier 2 (optional — runs Tier 3 if unavailable)
# ═══════════════════════════════════════════════════════════════════════════════

# === Environment Setup ===
export PATH="/usr/local/cuda-12.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"

CONDA_ENV=/iitjhome/senthil1/.conda/envs/tc_pipeline
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH=/iitjhome/senthil1
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0

cd /iitjhome/senthil1

if [ -f ~/.gemini_env ]; then
    source ~/.gemini_env
fi

find pipeline_v4_gpu/ -name "__pycache__" -exec rm -rf {} + 2>/dev/null

echo "═══════════════════════════════════════════════════════════════"
echo "  V4: Video_evaluation + CTML_03 (8 videos, 4 conditions)"
echo "  $(date)"
echo "  Node: $(hostname)"
echo "═══════════════════════════════════════════════════════════════"

echo ""
echo "=== GPU Check ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# === Check vLLM server (optional for Tier 2) ===
VLLM_OK=false
RESPONSE=$(curl -s http://localhost:8000/v1/models 2>/dev/null)
if echo "$RESPONSE" | python -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null; then
    echo "vLLM server is ready — will run Tier 2 after pipeline"
    VLLM_OK=true
else
    echo "vLLM server not available — Tier 2 will be skipped"
fi

# === Video List ===
VIDEOS=("Video_evaluation_01" "Video_evaluation_02" "Video_evaluation_03" "Video_evaluation_04"
        "CTML_03_01" "CTML_03_02" "CTML_03_03" "CTML_03_04")
STAGES=(1 2 5 4 6 7)

N_SUCCESS=0
N_FAIL=0

for VID in "${VIDEOS[@]}"; do
    echo ""
    echo "######################################################################"
    echo "# Processing: $VID (V4 full pipeline)"
    echo "######################################################################"
    echo "Start: $(date)"

    # Check video exists
    if [ ! -f "videos/$VID.mp4" ]; then
        echo "ERROR: videos/$VID.mp4 not found — skipping"
        N_FAIL=$((N_FAIL + 1))
        continue
    fi

    # Create output directory
    mkdir -p "outputs_v4/$VID"

    # Reuse V3 ASR if available (skip Stage 1)
    if [ -f "outputs_v3/$VID/transcript_segments_improved.csv" ]; then
        echo "=== $VID — Copying V3 ASR ==="
        cp -n "outputs_v3/$VID/transcript_segments_improved.csv" "outputs_v4/$VID/"
        cp -n "outputs_v3/$VID/transcript_words.csv" "outputs_v4/$VID/" 2>/dev/null
        cp -n "outputs_v3/$VID/transcript_raw.json" "outputs_v4/$VID/" 2>/dev/null
        STAGES_RUN=(2 5 4 6 7)
    elif [ -f "outputs_v2_2/$VID/transcript_segments_improved.csv" ]; then
        echo "=== $VID — Copying V2.2 ASR ==="
        cp -n "outputs_v2_2/$VID/transcript_segments_improved.csv" "outputs_v4/$VID/"
        cp -n "outputs_v2_2/$VID/transcript_words.csv" "outputs_v4/$VID/" 2>/dev/null
        cp -n "outputs_v2_2/$VID/transcript_raw.json" "outputs_v4/$VID/" 2>/dev/null
        STAGES_RUN=(2 5 4 6 7)
    else
        echo "=== $VID — No existing ASR, running full pipeline ==="
        STAGES_RUN=(1 2 5 4 6 7)
    fi

    # Clear V4 output files for fresh run (keep ASR)
    rm -f "outputs_v4/$VID/scenes.csv"
    rm -f "outputs_v4/$VID/ocr_per_frame.csv"
    rm -f "outputs_v4/$VID/dinov2_distances.csv"
    rm -f "outputs_v4/$VID/keyword_alignment.csv"
    rm -f "outputs_v4/$VID/segment_alignment.csv"
    rm -f "outputs_v4/$VID/keywords.csv"
    rm -f "outputs_v4/$VID/keyword_scores.csv"
    rm -f "outputs_v4/$VID/segment_scores.csv"
    rm -f "outputs_v4/$VID/results.json"
    rm -f "outputs_v4/$VID/pedagogical_importance.csv"
    rm -f "outputs_v4/$VID/report_dashboard.html"
    rm -rf "outputs_v4/$VID/diagnostics"

    GEMINI_ARGS=""
    if [ -n "$GEMINI_API_KEY" ]; then
        GEMINI_ARGS="--gemini-api-key $GEMINI_API_KEY"
    fi

    FAILED=false
    for STG in "${STAGES_RUN[@]}"; do
        echo ""
        echo "=== $VID — Stage $STG ==="
        echo "Time: $(date)"
        python -m pipeline_v4_gpu.main \
            --video "videos/$VID.mp4" \
            --output-root outputs_v4 \
            --stage $STG \
            $GEMINI_ARGS

        if [ $? -ne 0 ]; then
            echo "ERROR: Stage $STG failed for $VID"
            FAILED=true
            break
        fi
    done

    if [ "$FAILED" = true ]; then
        N_FAIL=$((N_FAIL + 1))
        continue
    fi

    # === Tier 2 (vLLM) if server available ===
    if [ "$VLLM_OK" = true ]; then
        echo ""
        echo "=== $VID — Tier 2 (vLLM) ==="
        TIER_DIR="outputs_v4/$VID/tier_comparison"
        mkdir -p "$TIER_DIR"

        # Save Tier 1 (Gemini)
        cp "$OUTDIR/pedagogical_importance.csv" "$TIER_DIR/importance_tier1_gemini.csv" 2>/dev/null
        cp "outputs_v4/$VID/pedagogical_importance.csv" "$TIER_DIR/importance_tier1_gemini.csv"
        cp "outputs_v4/$VID/results.json" "$TIER_DIR/results_tier1_gemini.json" 2>/dev/null
        cp "outputs_v4/$VID/keyword_scores.csv" "$TIER_DIR/keyword_scores_tier1_gemini.csv" 2>/dev/null

        # Save Tier 3 (Heuristic)
        rm -f "outputs_v4/$VID/pedagogical_importance.csv"
        python -m pipeline_v4_gpu.main \
            --video "videos/$VID.mp4" \
            --output-root outputs_v4 \
            --stage 6 \
            --importance-backend heuristic
        cp "outputs_v4/$VID/pedagogical_importance.csv" "$TIER_DIR/importance_tier3_heuristic.csv"
        rm -f "outputs_v4/$VID/keyword_scores.csv" "outputs_v4/$VID/segment_scores.csv" "outputs_v4/$VID/results.json"
        python -m pipeline_v4_gpu.main \
            --video "videos/$VID.mp4" \
            --output-root outputs_v4 \
            --stage 7
        cp "outputs_v4/$VID/results.json" "$TIER_DIR/results_tier3_heuristic.json" 2>/dev/null
        cp "outputs_v4/$VID/keyword_scores.csv" "$TIER_DIR/keyword_scores_tier3_heuristic.csv" 2>/dev/null

        # Run Tier 2 (vLLM)
        rm -f "outputs_v4/$VID/pedagogical_importance.csv"
        python -m pipeline_v4_gpu.main \
            --video "videos/$VID.mp4" \
            --output-root outputs_v4 \
            --stage 6 \
            --importance-backend local_llm \
            --local-vlm-backend vllm

        cp "outputs_v4/$VID/pedagogical_importance.csv" "$TIER_DIR/importance_tier2_vllm.csv"
        rm -f "outputs_v4/$VID/keyword_scores.csv" "outputs_v4/$VID/segment_scores.csv" "outputs_v4/$VID/results.json"
        python -m pipeline_v4_gpu.main \
            --video "videos/$VID.mp4" \
            --output-root outputs_v4 \
            --stage 7
        cp "outputs_v4/$VID/results.json" "$TIER_DIR/results_tier2_vllm.json" 2>/dev/null
        cp "outputs_v4/$VID/keyword_scores.csv" "$TIER_DIR/keyword_scores_tier2_vllm.csv" 2>/dev/null

        # Restore Gemini (Tier 1) as primary
        cp "$TIER_DIR/importance_tier1_gemini.csv" "outputs_v4/$VID/pedagogical_importance.csv"
        rm -f "outputs_v4/$VID/keyword_scores.csv" "outputs_v4/$VID/segment_scores.csv" "outputs_v4/$VID/results.json"
        python -m pipeline_v4_gpu.main \
            --video "videos/$VID.mp4" \
            --output-root outputs_v4 \
            --stage 7
    fi

    N_SUCCESS=$((N_SUCCESS + 1))
    echo "Completed $VID at $(date)"
done

# ═══════════════════════════════════════════════════════════════════════════════
# Condition Comparison: Video_evaluation set
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "######################################################################"
echo "# CONDITION COMPARISON: Video_evaluation (01-04)"
echo "######################################################################"

python -c "
import json, os
import pandas as pd
import numpy as np

conditions = {
    '01': 'Simultaneous',
    '02': 'Animation first',
    '03': 'Narration first',
    '04': 'Segmented'
}

for prefix, label in [('Video_evaluation', 'Video_evaluation'), ('CTML_03', 'CTML_03')]:
    print(f'')
    print(f'=== {label} ===')
    print(f'{\"Video\":<24} {\"Condition\":<18} {\"Score\":>8} {\"Grade\":<12} {\"Coverage\":>10} {\"Matched\":>10} {\"Opt%\":>8} {\"CaseA\":>8}')
    print('─' * 102)

    scores = {}
    for cond, cond_name in conditions.items():
        vid = f'{prefix}_{cond}'
        try:
            r = json.load(open(f'outputs_v4/{vid}/results.json'))
            kl = r['keyword_level']
            cov = r.get('coverage', {}).get('coverage_rate', 0) * 100

            # Case distribution
            kw = pd.read_csv(f'outputs_v4/{vid}/keyword_scores.csv')
            n_a = len(kw[kw['case'] == 'A']) if 'case' in kw.columns else 0

            scores[cond] = r['overall_score']
            print(f'{vid:<24} {cond_name:<18} {r[\"overall_score\"]:>8.2f} {r[\"overall_grade\"]:<12} {cov:>9.1f}% {kl[\"n_matched\"]}/{kl[\"n_total\"]:<7} {kl[\"pct_Optimal\"]:>7.1f}% {n_a:>8}')
        except Exception as e:
            print(f'{vid:<24} Error: {e}')

    if len(scores) == 4:
        print()
        print(f'  Expected ranking: 01 (Simul) > 04 (Seg) >= 02 (Anim) > 03 (Narr)')
        ranking = sorted(scores.items(), key=lambda x: -x[1])
        print(f'  Actual ranking:   {\" > \".join(f\"{c} ({s:.1f})\" for c, s in ranking)}')
        spread = max(scores.values()) - min(scores.values())
        print(f'  Spread: {spread:.2f} points')

        # Check key relationships
        if scores.get('01', 0) > scores.get('02', 0):
            print(f'  01 > 02 (Simul > Anim first): YES')
        else:
            print(f'  01 > 02 (Simul > Anim first): NO ← PROBLEM')
        if scores.get('01', 0) > scores.get('03', 0):
            print(f'  01 > 03 (Simul > Narr first): YES')
        else:
            print(f'  01 > 03 (Simul > Narr first): NO ← PROBLEM')

# === delta_t distribution analysis ===
print()
print('═══ Delta_t Distribution by Condition ═══')
for prefix in ['Video_evaluation', 'CTML_03']:
    print(f'')
    print(f'--- {prefix} ---')
    for cond, cond_name in conditions.items():
        vid = f'{prefix}_{cond}'
        try:
            kw = pd.read_csv(f'outputs_v4/{vid}/keyword_scores.csv')
            dt = kw['delta_t']
            vis = kw['visual_on_screen'].sum() if 'visual_on_screen' in kw.columns else 0
            n_neg = (dt < 0).sum()
            n_pos = (dt > 0).sum()
            n_zero = (dt == 0).sum()
            print(f'  {cond} ({cond_name:<16}): mean_dt={dt.mean():>+7.1f}s  neg={n_neg}  pos={n_pos}  vis_on={vis}  S_mean={kw[\"score\"].mean():>5.1f}')
        except Exception as e:
            print(f'  {cond} ({cond_name:<16}): Error: {e}')

# === V3 vs V4 comparison ===
print()
print('═══ V3 vs V4 Comparison ═══')
print(f'{\"Video\":<24} {\"V3 Score\":>10} {\"V4 Score\":>10} {\"Delta\":>8}')
print('─' * 55)
for prefix in ['Video_evaluation', 'CTML_03']:
    for cond in ['01','02','03','04']:
        vid = f'{prefix}_{cond}'
        try:
            # Try multiple V3 output locations
            r3 = None
            for v3dir in ['outputs_v3', 'outputs_v2_2']:
                p = f'{v3dir}/{vid}/results.json'
                if os.path.exists(p):
                    r3 = json.load(open(p))
                    break
            r4 = json.load(open(f'outputs_v4/{vid}/results.json'))
            s3 = r3['overall_score'] if r3 else None
            s4 = r4['overall_score']
            if s3 is not None:
                print(f'{vid:<24} {s3:>10.2f} {s4:>10.2f} {s4-s3:>+8.2f}')
            else:
                print(f'{vid:<24} {\"N/A\":>10} {s4:>10.2f}')
        except Exception as e:
            print(f'{vid:<24} Error: {e}')
" 2>&1

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  VE + CTML Batch Complete: $N_SUCCESS success, $N_FAIL failed"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════════════"
