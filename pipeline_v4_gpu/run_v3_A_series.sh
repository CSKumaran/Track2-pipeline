#!/bin/bash
#SBATCH --job-name=v3_A_all
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=v3_A_series_%j.log
#SBATCH --error=v3_A_series_%j.err
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

echo ""
echo "######################################################################"
echo "# V3: A Series — A0 (baseline), A1 (1s), A3 (3s), A5 (5s delay)"
echo "# Expected: A0 > A1 > A3 > A5 (monotonically decreasing scores)"
echo "######################################################################"

# A0 and A5 already done in previous test — only run A1 and A3
VIDEOS=("A1" "A3")
STAGES=(2 5 4 6 7)

for VID in "${VIDEOS[@]}"; do
    echo ""
    echo "######################################################################"
    echo "# Processing: $VID (V3 pipeline)"
    echo "######################################################################"
    echo "Start: $(date)"

    # Clear stages 2+ but keep Stage 1
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

# Generate comparison for all 4
echo ""
echo "=== Generating A Series Comparison ==="
python -m pipeline_v3_gpu.utils.viz_compare outputs_v3 A0 A1 A3 A5

echo ""
echo "######################################################################"
echo "# MONOTONIC TEST: A0 vs A1 vs A3 vs A5"
echo "######################################################################"

python -c "
import pandas as pd, json

videos = ['A0', 'A1', 'A3', 'A5']
delays = [0, 1, 3, 5]

print('=== SCORE COMPARISON ===')
print(f'{\"Video\":<8} {\"Delay\":<8} {\"Score\":<10} {\"Grade\":<12} {\"Matched\":<10} {\"Coverage\":<10} {\"Opt%\":<8} {\"Violations\":<10}')
print('-' * 80)

scores = []
for vid, delay in zip(videos, delays):
    try:
        r = json.load(open(f'outputs_v3/{vid}/results.json'))
        s = r['overall_score']
        scores.append(s)
        kl = r['keyword_level']
        cov = r.get('coverage', {})
        pos = r.get('positive_delta_t_only', {})
        print(f'{vid:<8} {delay:<8} {s:<10} {r[\"overall_grade\"]:<12} {kl[\"n_matched\"]}/{kl[\"n_total\"]:<7} {cov.get(\"coverage_rate\",0)*100:<9.1f}% {kl[\"pct_Optimal\"]:<7.1f}% {pos.get(\"n\",0)}')
    except Exception as e:
        print(f'{vid:<8} Error: {e}')
        scores.append(None)

print()
print('=== MONOTONIC CHECK ===')
valid = [s for s in scores if s is not None]
is_monotonic = all(valid[i] >= valid[i+1] for i in range(len(valid)-1))
print(f'Scores: {\" > \".join(str(s) for s in valid)}')
print(f'Monotonically decreasing: {\"YES\" if is_monotonic else \"NO\"}')
if len(valid) >= 2:
    print(f'Total drop (A0→A5): {valid[0] - valid[-1]:.2f} points')

# Per-keyword delta_t comparison across all 4 videos
print()
print('=== TOP KEYWORDS WITH LARGEST TIMING SHIFT (A0→A5) ===')
try:
    a0 = pd.read_csv('outputs_v3/A0/keyword_scores.csv')
    a1 = pd.read_csv('outputs_v3/A1/keyword_scores.csv')
    a3 = pd.read_csv('outputs_v3/A3/keyword_scores.csv')
    a5 = pd.read_csv('outputs_v3/A5/keyword_scores.csv')

    for df, name in [(a0,'A0'),(a1,'A1'),(a3,'A3'),(a5,'A5')]:
        df.rename(columns={'delta_t': f'dt_{name}', 'S_temporal': f'S_{name}', 'match_case': f'case_{name}'}, inplace=True)

    m0 = a0[a0[f'case_A0'] != 'F'][['keyword_text', f'dt_A0', f'S_A0']].copy()
    m1 = a1[a1[f'case_A1'] != 'F'][['keyword_text', f'dt_A1', f'S_A1']].copy()
    m3 = a3[a3[f'case_A3'] != 'F'][['keyword_text', f'dt_A3', f'S_A3']].copy()
    m5 = a5[a5[f'case_A5'] != 'F'][['keyword_text', f'dt_A5', f'S_A5']].copy()

    merged = m0.merge(m1, on='keyword_text', how='outer')
    merged = merged.merge(m3, on='keyword_text', how='outer')
    merged = merged.merge(m5, on='keyword_text', how='outer')

    merged['shift_A0_A5'] = merged['dt_A5'] - merged['dt_A0']
    top = merged.dropna(subset=['shift_A0_A5']).sort_values('shift_A0_A5', ascending=False).head(10)

    print(f'{\"keyword\":<40} {\"dt_A0\":>7} {\"dt_A1\":>7} {\"dt_A3\":>7} {\"dt_A5\":>7} {\"shift\":>7}')
    print('-' * 75)
    for _, row in top.iterrows():
        kw = str(row['keyword_text'])[:38]
        print(f'{kw:<40} {row[\"dt_A0\"]:>7.2f} {row.get(\"dt_A1\",float(\"nan\")):>7.2f} {row.get(\"dt_A3\",float(\"nan\")):>7.2f} {row[\"dt_A5\"]:>7.2f} {row[\"shift_A0_A5\"]:>+7.2f}')
except Exception as e:
    print(f'Error: {e}')
" 2>&1

echo ""
echo "=== DONE ==="
echo "End: $(date)"
