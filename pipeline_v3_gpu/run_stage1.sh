#!/bin/bash
#SBATCH --job-name=v22_stg1
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=v22_stage1_%j.log
#SBATCH --error=v22_stage1_%j.err
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
echo "=== Checking OCR engine availability ==="
python -c "from paddleocr import PaddleOCR; print('PaddleOCR: available')" 2>/dev/null && OCR_ENGINE="paddleocr" || {
    echo "PaddleOCR not available, using EasyOCR"
    OCR_ENGINE="easyocr"
}
echo "OCR engine: $OCR_ENGINE"

echo ""
echo "=== Pipeline v2.2 — Stage 1 ONLY (single video: A0) ==="
echo "Time: $(date)"
echo ""

# Run Stage 1 only on A0
python -m pipeline_v2_2_gpu.main \
    --video videos/A0.mp4 \
    --output-root outputs_v2_2 \
    --stage 1 \
    --whisper-model large-v3 \
    --asr-backend whisperx \
    --ocr-engine $OCR_ENGINE

echo ""
echo "=== Stage 1 Done ==="
echo "End: $(date)"
echo ""
echo "=== Output files ==="
ls -la outputs_v2_2/A0/
echo ""
echo "=== Diagnostics ==="
cat outputs_v2_2/A0/diagnostics/stage1_asr.json
