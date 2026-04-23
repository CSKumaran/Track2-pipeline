#!/bin/bash
#SBATCH --job-name=v22_stg2
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=v22_stage2_%j.log
#SBATCH --error=v22_stage2_%j.err
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
nvidia-smi
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

echo ""
echo "=== Dependency Check ==="
python -c "from scenedetect import open_video; print('PySceneDetect: available')" 2>/dev/null || echo "WARNING: PySceneDetect not available"
python -c "from transformers import AutoModel; print('Transformers (DINOv2): available')" 2>/dev/null || echo "WARNING: Transformers not available"

# OCR engine check: Surya primary, EasyOCR fallback
OCR_ENGINE="surya"
python -c "from surya.recognition import RecognitionPredictor; print('Surya OCR: available')" 2>/dev/null || {
    echo "Surya not available, trying EasyOCR"
    python -c "import easyocr; print('EasyOCR: available')" 2>/dev/null && OCR_ENGINE="easyocr" || {
        echo "WARNING: No OCR engine available — will run without OCR"
        OCR_ENGINE="surya"
    }
}

python -c "from sklearn.metrics.pairwise import cosine_similarity; print('scikit-learn: available')" 2>/dev/null || echo "WARNING: scikit-learn not available"
echo "OCR engine: $OCR_ENGINE"

echo ""
echo "=== Pipeline v2.2 — Stage 2 ONLY (single video: A0) ==="
echo "Time: $(date)"
echo ""

# Run Stage 2 only on A0
python -m pipeline_v2_2_gpu.main \
    --video videos/A0.mp4 \
    --output-root outputs_v2_2 \
    --stage 2 \
    --ocr-engine $OCR_ENGINE \
    --max-scene-duration 30.0 \
    --sample-interval 0.5

echo ""
echo "=== Stage 2 Done ==="
echo "End: $(date)"
echo ""
echo "=== Output files ==="
ls -la outputs_v2_2/A0/
echo ""
echo "=== Scenes CSV (first 30 lines) ==="
head -30 outputs_v2_2/A0/scenes.csv
echo ""
echo "=== OCR sample (first 20 lines) ==="
head -20 outputs_v2_2/A0/ocr_per_frame.csv
echo ""
echo "=== Diagnostics ==="
cat outputs_v2_2/A0/diagnostics/stage2_scene_detection.json
