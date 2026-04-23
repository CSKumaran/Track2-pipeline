#!/bin/bash
#SBATCH --job-name=v22_stg1_all
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=v22_stage1_all_%j.log
#SBATCH --error=v22_stage1_all_%j.err
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

echo "=== GPU Check ==="
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=== Pipeline v2.2 — Stage 1 ALL videos ==="
echo "Time: $(date)"
echo ""

# All 7 videos (A0 already done but will use cache)
VIDEOS=(
    videos/A0.mp4
    videos/A1.mp4
    videos/A3.mp4
    videos/A5.mp4
    videos/CTML_03_01.mp4
    videos/Video_evaluation_01.mp4
    videos/test_video.mp4
)

for VIDEO in "${VIDEOS[@]}"; do
    if [ -f "$VIDEO" ]; then
        NAME=$(basename "$VIDEO" .mp4)
        echo ""
        echo "=========================================="
        echo "Processing: $VIDEO"
        echo "Start: $(date)"
        echo "=========================================="

        python -m pipeline_v2_2_gpu.main \
            --video "$VIDEO" \
            --output-root outputs_v2_2 \
            --stage 1 \
            --whisper-model large-v3 \
            --asr-backend whisperx \
            --ocr-engine paddleocr

        echo ""
        echo "=== $NAME: Output files ==="
        ls -la "outputs_v2_2/$NAME/"
        echo ""
        echo "=== $NAME: Diagnostics ==="
        cat "outputs_v2_2/$NAME/diagnostics/stage1_asr.json"
        echo ""
        echo "Finished: $VIDEO at $(date)"
    else
        echo "SKIP: $VIDEO not found"
    fi
done

echo ""
echo "=== All Stage 1 Done ==="
echo "End: $(date)"
