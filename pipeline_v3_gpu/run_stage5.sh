#!/bin/bash
#SBATCH --job-name=v22_stg5
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=v22_stage5_%j.log
#SBATCH --error=v22_stage5_%j.err
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

echo ""
echo "=== Dependency Check ==="
python -c "import spacy; nlp = spacy.load('en_core_web_sm'); print('spaCy en_core_web_sm: available')" 2>/dev/null || {
    echo "Installing spaCy model..."
    python -m spacy download en_core_web_sm
}
python -c "from keybert import KeyBERT; print('KeyBERT: available')" 2>/dev/null || echo "WARNING: KeyBERT not available (will use spaCy only)"

echo ""
echo "=== Pipeline v2.2 — Stage 5 ONLY (single video: A0) ==="
echo "Time: $(date)"
echo ""

# Run Stage 5 only on A0
python -m pipeline_v2_2_gpu.main \
    --video videos/A0.mp4 \
    --output-root outputs_v2_2 \
    --stage 5

echo ""
echo "=== Stage 5 Done ==="
echo "End: $(date)"
echo ""
echo "=== Output files ==="
ls -la outputs_v2_2/A0/
echo ""
echo "=== Keywords CSV (first 30 lines) ==="
head -30 outputs_v2_2/A0/keywords.csv
echo ""
echo "=== Diagnostics ==="
cat outputs_v2_2/A0/diagnostics/stage5_keywords.json
