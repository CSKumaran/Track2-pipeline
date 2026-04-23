#!/bin/bash
#SBATCH --job-name=v3_gemini
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=v3_gemini_%j.log
#SBATCH --error=v3_gemini_%j.err
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

# === Load Gemini API key (secured) ===
source ~/.gemini_env

# Clear stale cache
find pipeline_v3_gpu/ -name "__pycache__" -exec rm -rf {} + 2>/dev/null

# Test batch: 4 videos (paid tier — no quota issues)
VIDEOS=(A0 A1 A3 A5)

echo "=== V3 Stage 6 (Gemini) + Stage 7 + Dashboard ==="
echo "Start: $(date)"
echo "Videos: ${#VIDEOS[@]}"
echo ""

for vid in "${VIDEOS[@]}"; do
    echo "--- Processing $vid ---"

    # Delete Stage 6+7 cache to force re-run
    rm -f outputs_v3/$vid/pedagogical_importance.csv
    rm -f outputs_v3/$vid/keyword_scores.csv
    rm -f outputs_v3/$vid/segment_scores.csv
    rm -f outputs_v3/$vid/results.json
    rm -f outputs_v3/$vid/diagnostics/stage6_importance.json
    rm -f outputs_v3/$vid/diagnostics/stage7_scoring.json
    rm -f outputs_v3/$vid/dashboard.html

    # Check if video output exists (stages 1-5 must be cached)
    if [ ! -d "outputs_v3/$vid" ]; then
        echo "  SKIP: outputs_v3/$vid not found (run full pipeline first)"
        continue
    fi

    # Paid tier: RPM=2000, no need for long delays
    sleep 2

    # Run Stage 6 with Gemini
    echo "  Stage 6 (Gemini): $(date)"
    python -m pipeline_v3_gpu.main \
        --video "videos/$vid.mp4" \
        --output-root outputs_v3 \
        --gemini-api-key "$GEMINI_API_KEY" \
        --stage 6

    # Run Stage 7
    echo "  Stage 7 (Scoring): $(date)"
    python -m pipeline_v3_gpu.main \
        --video "videos/$vid.mp4" \
        --output-root outputs_v3 \
        --stage 7

    # Check backend used
    if [ -f "outputs_v3/$vid/pedagogical_importance.csv" ]; then
        backend=$(head -2 "outputs_v3/$vid/pedagogical_importance.csv" | tail -1 | grep -o 'gemini\|heuristic\|local_llm')
        echo "  Backend: $backend"
    fi

    echo "  Done: $(date)"
    echo ""
done

echo "=== All videos processed ==="
echo "End: $(date)"

# Summary: check which videos used Gemini vs heuristic
echo ""
echo "=== Backend Summary ==="
for vid in "${VIDEOS[@]}"; do
    if [ -f "outputs_v3/$vid/pedagogical_importance.csv" ]; then
        backend=$(head -2 "outputs_v3/$vid/pedagogical_importance.csv" | tail -1 | grep -o 'gemini\|heuristic\|local_llm')
        echo "  $vid: $backend"
    else
        echo "  $vid: NO OUTPUT"
    fi
done
