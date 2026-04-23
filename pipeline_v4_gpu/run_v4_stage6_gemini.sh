#!/bin/bash
#SBATCH --job-name=v4_gemini
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=v4_gemini_%j.log
#SBATCH --error=v4_gemini_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

# ═══════════════════════════════════════════════════════════════════════════════
# V4: Re-run Stage 6 (Gemini multimodal) + Stage 7 + Dashboard
#
# Use this to upgrade existing V4 outputs from heuristic → Gemini importance
# without re-running Stages 1-5.
#
# V4 multimodal mode: sends keyframe images + transcript to Gemini
# for visual-verbal integration rating.
# ═══════════════════════════════════════════════════════════════════════════════

# === Environment Setup ===
export PATH="/usr/local/cuda-12.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"

CONDA_ENV=/iitjhome/senthil1/.conda/envs/tc_pipeline
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH=/iitjhome/senthil1
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0

cd /iitjhome/senthil1

# === Load Gemini API key (required) ===
if [ -f ~/.gemini_env ]; then
    source ~/.gemini_env
else
    echo "ERROR: ~/.gemini_env not found. Cannot run Gemini importance."
    exit 1
fi

if [ -z "$GEMINI_API_KEY" ]; then
    echo "ERROR: GEMINI_API_KEY is empty"
    exit 1
fi

find pipeline_v4_gpu/ -name "__pycache__" -exec rm -rf {} + 2>/dev/null

echo "═══════════════════════════════════════════════════════════════"
echo "  V4: Stage 6 Gemini Multimodal + Stage 7"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════════════"

VIDEOS=("A0" "A1" "A3" "A5" "B0" "B1" "B3" "B5" "C0" "C1" "C3" "C5" "D0" "D1" "D3" "D5")

for VID in "${VIDEOS[@]}"; do
    echo ""
    echo "--- Processing $VID ---"

    # Check stages 1-5 exist
    if [ ! -d "outputs_v4/$VID" ] || [ ! -f "outputs_v4/$VID/keyword_alignment.csv" ]; then
        echo "  SKIP: outputs_v4/$VID incomplete (run full pipeline first)"
        continue
    fi

    # Delete Stage 6+7 cache to force re-run
    rm -f "outputs_v4/$VID/pedagogical_importance.csv"
    rm -f "outputs_v4/$VID/keyword_scores.csv"
    rm -f "outputs_v4/$VID/segment_scores.csv"
    rm -f "outputs_v4/$VID/results.json"
    rm -f "outputs_v4/$VID/diagnostics/stage6_importance.json"
    rm -f "outputs_v4/$VID/diagnostics/stage7_scoring.json"
    rm -f "outputs_v4/$VID/report_dashboard.html"

    sleep 2  # Respect Gemini rate limits between videos

    # Stage 6 with Gemini multimodal
    echo "  Stage 6 (Gemini multimodal): $(date)"
    python -m pipeline_v4_gpu.main \
        --video "videos/$VID.mp4" \
        --output-root outputs_v4 \
        --gemini-api-key "$GEMINI_API_KEY" \
        --stage 6

    # Stage 7
    echo "  Stage 7 (Scoring): $(date)"
    python -m pipeline_v4_gpu.main \
        --video "videos/$VID.mp4" \
        --output-root outputs_v4 \
        --stage 7

    # Check backend used
    if [ -f "outputs_v4/$VID/pedagogical_importance.csv" ]; then
        backend=$(head -2 "outputs_v4/$VID/pedagogical_importance.csv" | tail -1 | grep -o 'gemini_multimodal\|gemini\|heuristic\|local_vlm')
        echo "  Backend: $backend"
    fi

    echo "  Done: $(date)"
done

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Backend Summary"
echo "═══════════════════════════════════════════════════════════════"

for VID in "${VIDEOS[@]}"; do
    if [ -f "outputs_v4/$VID/pedagogical_importance.csv" ]; then
        backend=$(head -2 "outputs_v4/$VID/pedagogical_importance.csv" | tail -1 | grep -o 'gemini_multimodal\|gemini\|heuristic\|local_vlm')
        echo "  $VID: $backend"
    else
        echo "  $VID: NO OUTPUT"
    fi
done

echo ""
echo "=== DONE ==="
echo "End: $(date)"
