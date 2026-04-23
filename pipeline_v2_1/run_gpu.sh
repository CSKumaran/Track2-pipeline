#!/bin/bash
#SBATCH --job-name=tc_pipeline
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=pipeline_%j.log
#SBATCH --error=pipeline_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

# === Environment Setup ===
export PATH="/usr/local/cuda-12.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"

eval "$(conda shell.bash hook)"
conda activate tc_pipeline

cd /iitjhome/senthil1

echo "=== GPU Check ==="
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=== Starting Pipeline v2.1 ==="
echo "Time: $(date)"
echo ""

# Process all 7 videos sequentially
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
        echo ""
        echo "=========================================="
        echo "Processing: $VIDEO"
        echo "Start: $(date)"
        echo "=========================================="

        python -m pipeline_v2_1.main \
            --video "$VIDEO" \
            --output-root outputs_v2_1_gpu \
            --whisper-model medium \
            --ocr-engine paddleocr \
            --vlm-mode skip \
            --importance-backend heuristic \
            --score-tau 2.5

        echo "Finished: $VIDEO at $(date)"
    else
        echo "SKIP: $VIDEO not found"
    fi
done

echo ""
echo "=== All Done ==="
echo "End: $(date)"
