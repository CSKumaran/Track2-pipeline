#!/bin/bash
#SBATCH --job-name=v22_stg67
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=v22_stage67_%j.log
#SBATCH --error=v22_stage67_%j.err
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

# Delete Stage 6+7 cache (force re-run)
rm -f outputs_v2_2/A0/pedagogical_importance.csv
rm -f outputs_v2_2/A0/keyword_scores.csv
rm -f outputs_v2_2/A0/segment_scores.csv
rm -f outputs_v2_2/A0/results.json
rm -f outputs_v2_2/A0/diagnostics/stage6_importance.json
rm -f outputs_v2_2/A0/diagnostics/stage7_scoring.json

echo "=== GPU Check ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=== Pipeline v2.2 — Stage 6 (Pedagogical Importance) ==="
echo "Time: $(date)"
python -m pipeline_v2_2_gpu.main \
    --video videos/A0.mp4 \
    --output-root outputs_v2_2 \
    --stage 6

echo ""
echo "=== Pipeline v2.2 — Stage 7 (Scoring & Aggregation) ==="
echo "Time: $(date)"
python -m pipeline_v2_2_gpu.main \
    --video videos/A0.mp4 \
    --output-root outputs_v2_2 \
    --stage 7

echo ""
echo "=== Done ==="
echo "End: $(date)"
echo ""
echo "=== Output files ==="
ls -la outputs_v2_2/A0/
echo ""
echo "=== Pedagogical Importance (first 20 lines) ==="
head -20 outputs_v2_2/A0/pedagogical_importance.csv
echo ""
echo "=== Keyword Scores (first 20 lines) ==="
head -20 outputs_v2_2/A0/keyword_scores.csv
echo ""
echo "=== Segment Scores ==="
cat outputs_v2_2/A0/segment_scores.csv
echo ""
echo "=== Results JSON ==="
cat outputs_v2_2/A0/results.json
echo ""
echo "=== Stage 6 Diagnostics ==="
cat outputs_v2_2/A0/diagnostics/stage6_importance.json
echo ""
echo "=== Stage 7 Diagnostics ==="
cat outputs_v2_2/A0/diagnostics/stage7_scoring.json
