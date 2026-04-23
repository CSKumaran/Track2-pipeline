#!/bin/bash
#SBATCH --job-name=v22_all
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=v22_all_videos_%j.log
#SBATCH --error=v22_all_videos_%j.err
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

# Clear stale cache
find pipeline_v2_2_gpu/ -name "__pycache__" -exec rm -rf {} + 2>/dev/null

echo "=== GPU Check ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

# ===================================================================
# Run ALL stages (1-7 + dashboard) FRESH for A1, A3, A5
# A0 is already done — skip it.
# ===================================================================

VIDEOS=("A1" "A3" "A5")
STAGES=(1 2 5 4 6 7)

for VID in "${VIDEOS[@]}"; do
    echo ""
    echo "######################################################################"
    echo "# Processing: $VID (FULL PIPELINE — ALL STAGES FRESH)"
    echo "######################################################################"
    echo "Start: $(date)"

    # Delete ALL previous outputs for this video (fresh run)
    echo "Clearing old outputs for $VID..."
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
            echo "ERROR: Stage $STG failed for $VID — skipping remaining stages"
            break
        fi
    done

    echo ""
    echo "=== $VID — Output files ==="
    ls -la "outputs_v2_2/$VID/" 2>/dev/null
    echo ""
    echo "=== $VID — Results JSON ==="
    cat "outputs_v2_2/$VID/results.json" 2>/dev/null || echo "No results.json"
    echo ""
    echo "=== $VID — Dashboard? ==="
    ls -la "outputs_v2_2/$VID/report_dashboard.html" 2>/dev/null && echo "YES" || echo "NO"
    echo ""
    echo "Completed $VID at $(date)"
done

# ===================================================================
# Generate Comparative Dashboard (A0, A1, A3, A5)
# ===================================================================
echo ""
echo "######################################################################"
echo "# Generating Comparative Dashboard"
echo "######################################################################"
echo "Time: $(date)"
python -m pipeline_v2_2_gpu.utils.viz_compare outputs_v2_2 A0 A1 A3 A5

echo ""
echo "######################################################################"
echo "# ALL DONE"
echo "######################################################################"
echo "End: $(date)"

echo ""
echo "=== Summary: All Videos ==="
for VID in A0 A1 A3 A5; do
    echo ""
    echo "--- $VID ---"
    python -c "
import json
try:
    r = json.load(open('outputs_v2_2/$VID/results.json'))
    print(f'  Overall Score: {r.get(\"overall_score\", \"N/A\")}')
    print(f'  Grade: {r.get(\"overall_grade\", \"N/A\")}')
    kl = r['keyword_level']
    print(f'  Keywords: {kl[\"n_matched\"]}/{kl[\"n_total\"]} matched ({kl[\"match_rate\"]*100:.1f}%)')
    cov = r.get('coverage', {})
    print(f'  Coverage: {cov.get(\"coverage_rate\",0)*100:.1f}%')
    print(f'  Coverage-adjusted: {cov.get(\"tc_score_coverage_adjusted\", \"N/A\")}')
    print(f'  Zone: Opt={kl[\"pct_Optimal\"]:.1f}% Sub={kl[\"pct_Suboptimal\"]:.1f}% Dis={kl[\"pct_Disruptive\"]:.1f}% Una={kl[\"pct_Unacceptable\"]:.1f}%')
    cd = kl['case_distribution']
    print(f'  Cases: A={cd.get(\"A\",0)} D={cd.get(\"D\",0)} B={cd.get(\"B\",0)} F={kl[\"n_unmatched\"]}')
    pos = r.get('positive_delta_t_only', {})
    print(f'  True violations (pos delta_t): n={pos.get(\"n\",0)}, mean={pos.get(\"mean_delta_t\",\"N/A\")}s')
except Exception as e:
    print(f'  Error: {e}')
" 2>/dev/null
done
