#!/bin/bash
#SBATCH --job-name=v3_BCD
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=v3_BCD_%j.log
#SBATCH --error=v3_BCD_%j.err
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

VIDEOS=("B0" "B1" "B3" "B5" "C0" "C1" "C3" "C5" "D0" "D1" "D3" "D5")
STAGES=(1 2 5 4 6 7)

for VID in "${VIDEOS[@]}"; do
    echo ""
    echo "######################################################################"
    echo "# Processing: $VID (V3 — full pipeline)"
    echo "######################################################################"
    echo "Start: $(date)"

    rm -rf "outputs_v3/$VID"

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

# Comparisons
echo ""
echo "=== Generating Comparisons ==="
python -m pipeline_v3_gpu.utils.viz_compare outputs_v3 B0 B1 B3 B5
python -m pipeline_v3_gpu.utils.viz_compare outputs_v3 C0 C1 C3 C5
python -m pipeline_v3_gpu.utils.viz_compare outputs_v3 D0 D1 D3 D5

echo ""
echo "######################################################################"
echo "# V3 BATCH BCD DONE"
echo "######################################################################"
echo "End: $(date)"

echo ""
echo "=== Summary ==="
for VID in B0 B1 B3 B5 C0 C1 C3 C5 D0 D1 D3 D5; do
    echo "--- $VID ---"
    python -c "
import json
try:
    r = json.load(open('outputs_v3/$VID/results.json'))
    kl = r['keyword_level']
    cov = r.get('coverage', {})
    pos = r.get('positive_delta_t_only', {})
    print(f'  Score={r[\"overall_score\"]} Grade={r[\"overall_grade\"]} Match={kl[\"n_matched\"]}/{kl[\"n_total\"]} Cov={cov.get(\"coverage_rate\",0)*100:.1f}% Opt={kl[\"pct_Optimal\"]:.0f}% Violations={pos.get(\"n\",0)}')
except Exception as e:
    print(f'  Error: {e}')
" 2>/dev/null
done
