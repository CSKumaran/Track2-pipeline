#!/bin/bash
#SBATCH --job-name=v4_A_all
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=v4_A_series_%j.log
#SBATCH --error=v4_A_series_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

# ═══════════════════════════════════════════════════════════════════════════════
# V4: A Series — A0 (baseline), A1 (1s), A3 (3s), A5 (5s delay)
# Run AFTER run_v4_test_A0_A5.sh validates the pipeline.
#
# Expected: A0 > A1 > A3 > A5 (monotonically decreasing scores)
# Target:   Coverage 50%+, Spread 5+ points
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
echo "  V4: A Series (A0, A1, A3, A5)"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════════════"

echo ""
echo "=== GPU Check ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# A0 and A5 may already be done from test run — only run A1, A3
# Change to all 4 if you want fresh runs
VIDEOS=("A1" "A3")
STAGES=(2 5 4 6 7)

for VID in "${VIDEOS[@]}"; do
    echo ""
    echo "######################################################################"
    echo "# Processing: $VID (V4 pipeline)"
    echo "######################################################################"
    echo "Start: $(date)"

    # Reuse V3 ASR
    mkdir -p "outputs_v4/$VID"
    if [ -f "outputs_v3/$VID/transcript_segments_improved.csv" ]; then
        echo "=== $VID — Copying V3 ASR ==="
        cp -n "outputs_v3/$VID/transcript_segments_improved.csv" "outputs_v4/$VID/"
        cp -n "outputs_v3/$VID/transcript_words.csv" "outputs_v4/$VID/" 2>/dev/null
        cp -n "outputs_v3/$VID/transcript_raw.json" "outputs_v4/$VID/" 2>/dev/null
    else
        echo "=== $VID — Stage 1 (ASR) ==="
        python -m pipeline_v4_gpu.main \
            --video "videos/$VID.mp4" \
            --output-root outputs_v4 \
            --stage 1
    fi

    # Clear stages 2+ for fresh V4 run
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

    for STG in "${STAGES[@]}"; do
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
            break
        fi
    done

    echo "Completed $VID at $(date)"
done

# ═══════════════════════════════════════════════════════════════════════════════
# A Series Comparison & Monotonic Test
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "######################################################################"
echo "# V4 MONOTONIC TEST: A0 vs A1 vs A3 vs A5"
echo "######################################################################"

python -c "
import pandas as pd, json

videos = ['A0', 'A1', 'A3', 'A5']
delays = [0, 1, 3, 5]

print('=== V4 A SERIES ===')
print(f'{\"Video\":<8} {\"Delay\":<8} {\"Score\":<10} {\"Grade\":<12} {\"Matched\":<10} {\"Coverage\":<10} {\"TrackE\":<8} {\"Opt%\":<8}')
print('-' * 80)

scores = []
for vid, delay in zip(videos, delays):
    try:
        r = json.load(open(f'outputs_v4/{vid}/results.json'))
        s = r['overall_score']
        scores.append(s)
        kl = r['keyword_level']
        cov = r.get('coverage', {})

        kw = pd.read_csv(f'outputs_v4/{vid}/keyword_alignment.csv')
        n_e = len(kw[kw['match_case'] == 'E']) if 'match_case' in kw.columns else 0

        print(f'{vid:<8} {delay:<8} {s:<10} {r[\"overall_grade\"]:<12} {kl[\"n_matched\"]}/{kl[\"n_total\"]:<7} {cov.get(\"coverage_rate\",0)*100:<9.1f}% {n_e:<8} {kl[\"pct_Optimal\"]:<7.1f}%')
    except Exception as e:
        print(f'{vid:<8} Error: {e}')
        scores.append(None)

print()
valid = [s for s in scores if s is not None]
is_monotonic = all(valid[i] >= valid[i+1] for i in range(len(valid)-1))
print(f'Scores: {\" > \".join(f\"{s:.2f}\" for s in valid)}')
print(f'Monotonically decreasing: {\"YES\" if is_monotonic else \"NO\"}')
if len(valid) >= 2:
    print(f'Total spread (A0-A5): {valid[0] - valid[-1]:.2f} points')

# V3 comparison
print()
print('=== V3 vs V4 COMPARISON ===')
print(f'{\"Video\":<8} {\"V3 Score\":<12} {\"V4 Score\":<12} {\"V3 Cov\":<10} {\"V4 Cov\":<10}')
print('-' * 55)
for vid in videos:
    try:
        r3 = json.load(open(f'outputs_v3/{vid}/results.json'))
        r4 = json.load(open(f'outputs_v4/{vid}/results.json'))
        c3 = r3.get('coverage', {}).get('coverage_rate', 0) * 100
        c4 = r4.get('coverage', {}).get('coverage_rate', 0) * 100
        print(f'{vid:<8} {r3[\"overall_score\"]:<12} {r4[\"overall_score\"]:<12} {c3:<9.1f}% {c4:<9.1f}%')
    except:
        pass

# Spread comparison
try:
    r3_a0 = json.load(open('outputs_v3/A0/results.json'))
    r3_a5 = json.load(open('outputs_v3/A5/results.json'))
    r4_a0 = json.load(open('outputs_v4/A0/results.json'))
    r4_a5 = json.load(open('outputs_v4/A5/results.json'))
    print()
    print(f'V3 spread: {r3_a0[\"overall_score\"] - r3_a5[\"overall_score\"]:.2f}')
    print(f'V4 spread: {r4_a0[\"overall_score\"] - r4_a5[\"overall_score\"]:.2f}')
except:
    pass
" 2>&1

echo ""
echo "=== DONE ==="
echo "End: $(date)"
