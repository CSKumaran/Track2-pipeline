#!/bin/bash
#SBATCH --job-name=v22_D_rerun
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=v22_D_rerun_%j.log
#SBATCH --error=v22_D_rerun_%j.err
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

find pipeline_v2_2_gpu/ -name "__pycache__" -exec rm -rf {} + 2>/dev/null

echo "=== GPU Check ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

echo "=== Video file sizes (should be ~17-18MB each) ==="
ls -la videos/D*.mp4

VIDEOS=("D0" "D1" "D3" "D5")
STAGES=(1 2 5 4 6 7)

for VID in "${VIDEOS[@]}"; do
    echo ""
    echo "######################################################################"
    echo "# Processing: $VID (FULL PIPELINE — RERUN WITH CORRECTED FILES)"
    echo "######################################################################"
    echo "Start: $(date)"

    rm -rf "outputs_v2_2/$VID"

    for STG in "${STAGES[@]}"; do
        echo ""
        echo "=== $VID — Stage $STG ==="
        echo "Time: $(date)"
        python -m pipeline_v2_2_gpu.main \
            --video "videos/$VID.mp4" \
            --output-root outputs_v2_2 \
            --stage $STG

        if [ $? -ne 0 ]; then
            echo "ERROR: Stage $STG failed for $VID"
            break
        fi
    done

    echo ""
    echo "=== $VID — Results ==="
    cat "outputs_v2_2/$VID/results.json" 2>/dev/null || echo "No results.json"
    echo ""
    echo "=== $VID — Dashboard? ==="
    ls -la "outputs_v2_2/$VID/report_dashboard.html" 2>/dev/null && echo "YES" || echo "NO"
    echo "Completed $VID at $(date)"
done

# Regenerate comparison
echo ""
echo "=== Generating D Series Comparison ==="
python -m pipeline_v2_2_gpu.utils.viz_compare outputs_v2_2 D0 D1 D3 D5

echo ""
echo "######################################################################"
echo "# D SERIES RERUN DONE"
echo "######################################################################"
echo "End: $(date)"

echo ""
echo "=== Summary ==="
for VID in D0 D1 D3 D5; do
    echo "--- $VID ---"
    python -c "
import json
try:
    r = json.load(open('outputs_v2_2/$VID/results.json'))
    kl = r['keyword_level']
    cov = r.get('coverage', {})
    pos = r.get('positive_delta_t_only', {})
    print(f'  Score={r[\"overall_score\"]} Grade={r[\"overall_grade\"]} Match={kl[\"n_matched\"]}/{kl[\"n_total\"]} Cov={cov.get(\"coverage_rate\",0)*100:.1f}% Opt={kl[\"pct_Optimal\"]:.0f}% Violations={pos.get(\"n\",0)}')
except Exception as e:
    print(f'  Error: {e}')
" 2>/dev/null
done
