#!/bin/bash
#SBATCH --job-name=v3_test
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=v3_test_A0_A5_%j.log
#SBATCH --error=v3_test_A0_A5_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

# === Environment Setup ===
export PATH="/usr/local/cuda-12.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"

CONDA_ENV=/iitjhome/senthil1/.conda/envs/tc_pipeline
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH=/iitjhome/senthil1
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0

cd /iitjhome/senthil1

find pipeline_v3_gpu/ -name "__pycache__" -exec rm -rf {} + 2>/dev/null

echo "=== GPU Check ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "######################################################################"
echo "# V3 TEST: A0 (baseline) vs A5 (5s delay)"
echo "# Expected: A5 should score LOWER than A0"
echo "# Expected: 4-6 keywords in A5 should show +5s delta_t"
echo "######################################################################"

VIDEOS=("A0" "A5")
STAGES=(2 5 4 6 7)

for VID in "${VIDEOS[@]}"; do
    echo ""
    echo "######################################################################"
    echo "# Processing: $VID (V3 — full pipeline fresh)"
    echo "######################################################################"
    echo "Start: $(date)"

    # Clear old outputs but keep Stage 1 (ASR is identical across versions)
    rm -f "outputs_v3/$VID/scenes.csv"
    rm -f "outputs_v3/$VID/ocr_per_frame.csv"
    rm -f "outputs_v3/$VID/dinov2_distances.csv"
    rm -f "outputs_v3/$VID/keyword_alignment.csv"
    rm -f "outputs_v3/$VID/segment_alignment.csv"
    rm -f "outputs_v3/$VID/keywords.csv"
    rm -f "outputs_v3/$VID/keyword_scores.csv"
    rm -f "outputs_v3/$VID/segment_scores.csv"
    rm -f "outputs_v3/$VID/results.json"
    rm -f "outputs_v3/$VID/pedagogical_importance.csv"
    rm -f "outputs_v3/$VID/report_dashboard.html"
    rm -rf "outputs_v3/$VID/diagnostics"
    rm -rf "outputs_v3/$VID/frames"

    # Run Stage 1 only if transcript doesn't exist
    if [ ! -f "outputs_v3/$VID/transcript_segments_improved.csv" ]; then
        echo "=== $VID — Stage 1 (ASR) ==="
        python -m pipeline_v3_gpu.main \
            --video "videos/$VID.mp4" \
            --output-root outputs_v3 \
            --stage 1
    else
        echo "=== $VID — Stage 1 SKIPPED (transcript exists) ==="
    fi

    for STG in "${STAGES[@]}"; do
        echo ""
        echo "=== $VID — Stage $STG ==="
        echo "Time: $(date)"
        python -m pipeline_v3_gpu.main \
            --video "videos/$VID.mp4" \
            --output-root outputs_v3 \
            --stage $STG

        if [ $? -ne 0 ]; then
            echo "ERROR: Stage $STG failed for $VID"
            break
        fi
    done

    echo ""
    echo "=== $VID — Dashboard? ==="
    ls -la "outputs_v3/$VID/report_dashboard.html" 2>/dev/null && echo "YES" || echo "NO"
    echo "Completed $VID at $(date)"
done

# Generate comparison
echo ""
echo "=== Generating A0 vs A5 Comparison ==="
python -m pipeline_v3_gpu.utils.viz_compare outputs_v3 A0 A5

echo ""
echo "######################################################################"
echo "# V3 RESULTS COMPARISON: A0 vs A5"
echo "######################################################################"

for VID in A0 A5; do
    echo ""
    echo "=== $VID — Results JSON ==="
    cat "outputs_v3/$VID/results.json"
done

echo ""
echo "######################################################################"
echo "# DELTA_T COMPARISON (keywords matched in both)"
echo "######################################################################"

python -c "
import pandas as pd, json

a0_kw = pd.read_csv('outputs_v3/A0/keyword_scores.csv')
a5_kw = pd.read_csv('outputs_v3/A5/keyword_scores.csv')

a0_matched = a0_kw[a0_kw['match_case'] != 'F'][['keyword_text', 'delta_t', 'S_temporal', 'match_case', 't_vis']].copy()
a5_matched = a5_kw[a5_kw['match_case'] != 'F'][['keyword_text', 'delta_t', 'S_temporal', 'match_case', 't_vis']].copy()

a0_matched.columns = ['keyword', 'dt_A0', 'S_A0', 'case_A0', 'tvis_A0']
a5_matched.columns = ['keyword', 'dt_A5', 'S_A5', 'case_A5', 'tvis_A5']

merged = pd.merge(a0_matched, a5_matched, on='keyword', how='outer', indicator=True)

print('=== Keywords matched in BOTH A0 and A5 ===')
both = merged[merged['_merge'] == 'both'].copy()
both['dt_diff'] = both['dt_A5'] - both['dt_A0']
both_sorted = both.sort_values('dt_diff', ascending=False)
print(both_sorted[['keyword', 'dt_A0', 'dt_A5', 'dt_diff', 'S_A0', 'S_A5', 'case_A0', 'case_A5']].to_string(index=False))

print()
print(f'=== SUMMARY ===')
print(f'A0 matched: {len(a0_matched)}, A5 matched: {len(a5_matched)}, In both: {len(both)}')
print(f'A0 mean delta_t: {a0_matched[\"dt_A0\"].mean():.2f}s')
print(f'A5 mean delta_t: {a5_matched[\"dt_A5\"].mean():.2f}s')
print(f'Mean delta_t DIFFERENCE (A5-A0): {both[\"dt_diff\"].mean():.2f}s')
print(f'Keywords where A5 is >1s worse: {(both[\"dt_diff\"] > 1).sum()}')
print(f'Keywords where A5 is >3s worse: {(both[\"dt_diff\"] > 3).sum()}')
print(f'Keywords where A5 is >5s worse: {(both[\"dt_diff\"] > 5).sum()}')

r0 = json.load(open('outputs_v3/A0/results.json'))
r5 = json.load(open('outputs_v3/A5/results.json'))
print(f'')
print(f'A0 score: {r0[\"overall_score\"]} ({r0[\"overall_grade\"]})')
print(f'A5 score: {r5[\"overall_score\"]} ({r5[\"overall_grade\"]})')
print(f'Score difference: {r0[\"overall_score\"] - r5[\"overall_score\"]:.2f}')
" 2>&1

echo ""
echo "=== DONE ==="
echo "End: $(date)"
