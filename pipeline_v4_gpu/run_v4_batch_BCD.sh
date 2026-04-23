#!/bin/bash
#SBATCH --job-name=v4_BCD
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=v4_BCD_%j.log
#SBATCH --error=v4_BCD_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

# ═══════════════════════════════════════════════════════════════════════════════
# V4: B, C, D Series — Full pipeline run
#
# B series: high coverage baseline (V3: 72.8%)
# C series: low coverage (V3: 27%) — should improve significantly
# D series: lowest coverage (V3: 11.9%) — acid test for Track E
#
# Run AFTER A series validates the pipeline.
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
echo "  V4: B, C, D Series"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════════════"

echo ""
echo "=== GPU Check ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

VIDEOS=("B0" "B1" "B3" "B5" "C0" "C1" "C3" "C5" "D0" "D1" "D3" "D5")
STAGES=(2 5 4 6 7)

for VID in "${VIDEOS[@]}"; do
    echo ""
    echo "######################################################################"
    echo "# Processing: $VID (V4 — full pipeline)"
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

    # Fresh V4 run
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
# Summary
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "######################################################################"
echo "# V4 BCD SUMMARY"
echo "######################################################################"

python -c "
import pandas as pd, json

series = {
    'B': ['B0', 'B1', 'B3', 'B5'],
    'C': ['C0', 'C1', 'C3', 'C5'],
    'D': ['D0', 'D1', 'D3', 'D5'],
}

for name, videos in series.items():
    print(f'')
    print(f'=== {name} Series ===')
    print(f'{\"Video\":<8} {\"V3 Score\":<12} {\"V4 Score\":<12} {\"V3 Cov\":<10} {\"V4 Cov\":<10} {\"TrackE\":<8}')
    print('-' * 62)

    scores = []
    for vid in videos:
        try:
            r4 = json.load(open(f'outputs_v4/{vid}/results.json'))
            s4 = r4['overall_score']
            c4 = r4.get('coverage', {}).get('coverage_rate', 0) * 100
            scores.append(s4)

            kw = pd.read_csv(f'outputs_v4/{vid}/keyword_alignment.csv')
            n_e = len(kw[kw['match_case'] == 'E'])

            try:
                r3 = json.load(open(f'outputs_v3/{vid}/results.json'))
                s3 = r3['overall_score']
                c3 = r3.get('coverage', {}).get('coverage_rate', 0) * 100
            except:
                s3, c3 = '?', '?'

            print(f'{vid:<8} {str(s3):<12} {s4:<12} {str(c3):<10} {c4:<9.1f}% {n_e:<8}')
        except Exception as e:
            print(f'{vid:<8} Error: {e}')

    valid = [s for s in scores if s is not None]
    if len(valid) >= 2:
        is_mono = all(valid[i] >= valid[i+1] for i in range(len(valid)-1))
        print(f'  Monotonic: {\"YES\" if is_mono else \"NO\"}')
        print(f'  Spread ({videos[0]}-{videos[-1]}): {valid[0] - valid[-1]:.2f}')
" 2>&1

echo ""
echo "=== DONE ==="
echo "End: $(date)"
