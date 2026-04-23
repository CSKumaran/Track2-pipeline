#!/bin/bash
#SBATCH --job-name=vllm_gemma
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=vllm_gemma_%j.log
#SBATCH --error=vllm_gemma_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G

# ═══════════════════════════════════════════════════════════════════════════════
# vLLM Server for Tier 2b (Gemma 4) on port 8001
#
# Gemma 4 model variants on HuggingFace:
#   - google/gemma-4-E2B-it    → 2.3B effective, ~7GB bf16 (fits MIG slice)
#   - google/gemma-4-E4B-it    → 4.5B effective, ~10GB bf16 (fits A30 easily)
#   - google/gemma-4-26B-A4B-it → MoE, 4B active params, ~18GB bf16 (fits A30)
#   - google/gemma-4-31B-it    → 30.7B dense, ~24GB bf16 (tight on A30)
#
# Default: 26B-A4B-it (best smartness/speed tradeoff — MoE with only 4B active)
# Falls back to E4B-it if VRAM < 20GB
#
# IMPORTANT: Gemma 4 requires vLLM >= 0.12 (released ~Feb 2026).
#            If you see "model not supported", upgrade vLLM:
#            pip install -U vllm
#
# Usage:
#   sbatch --nodelist=cn21 pipeline_v4_gpu/run_v4_vllm_gemma.sh   # full A30
#   # Wait for "Started server" in log, then:
#   sbatch --nodelist=cn21 pipeline_v4_gpu/run_v4_tier2_gemma_batch.sh
# ═══════════════════════════════════════════════════════════════════════════════

export PATH="/usr/local/cuda-12.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"

CONDA_ENV=/iitjhome/senthil1/.conda/envs/tc_pipeline
export PATH="$CONDA_ENV/bin:$PATH"

cd /iitjhome/senthil1

echo "═══════════════════════════════════════════════════════════════"
echo "  vLLM Server — Gemma 3-12B-IT"
echo "  $(date)"
echo "  Node: $(hostname)"
echo "═══════════════════════════════════════════════════════════════"

# === GPU Info ===
echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# === Install / upgrade vLLM nightly (Gemma 4 needs vLLM nightly) ===
echo ""
echo "=== Checking vLLM installation ==="
VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null)
echo "Current vLLM version: ${VLLM_VERSION:-not installed}"

# Test if vLLM supports Gemma 4 architecture
VLLM_GEMMA4_OK=$(python -c "
try:
    from vllm.model_executor.models.registry import ModelRegistry
    archs = ModelRegistry.get_supported_archs()
    print('yes' if any('Gemma4' in a for a in archs) else 'no')
except Exception:
    print('no')
" 2>/dev/null)

if [ "$VLLM_GEMMA4_OK" != "yes" ]; then
    echo "vLLM ${VLLM_VERSION} does not support Gemma 4 — installing nightly..."
    pip install --pre -U vllm \
        --extra-index-url https://wheels.vllm.ai/nightly \
        --quiet 2>&1 | tail -5
    VLLM_VERSION=$(python -c "import vllm; print(vllm.__version__)" 2>/dev/null)
    echo "Upgraded vLLM nightly: $VLLM_VERSION"
else
    echo "vLLM already supports Gemma 4"
fi

# Check transformers — Gemma 4 requires a recent version (4.50+) that recognizes
# the 'gemma4' architecture. If too old, install from source.
echo ""
echo "=== Checking transformers for Gemma 4 support ==="
TFM_VERSION=$(python -c "import transformers; print(transformers.__version__)" 2>/dev/null)
echo "transformers version: $TFM_VERSION"

# Test if transformers recognizes gemma4
GEMMA4_OK=$(python -c "
from transformers import AutoConfig
try:
    AutoConfig.for_model('gemma4')
    print('yes')
except Exception:
    print('no')
" 2>/dev/null)

if [ "$GEMMA4_OK" != "yes" ]; then
    echo "transformers $TFM_VERSION does not support gemma4 — upgrading from source..."
    pip install -U --quiet "git+https://github.com/huggingface/transformers.git" 2>&1 | tail -5
    python -c "import transformers; print(f'Upgraded transformers: {transformers.__version__}')"
    # Re-test
    GEMMA4_OK=$(python -c "
from transformers import AutoConfig
try:
    AutoConfig.for_model('gemma4')
    print('yes')
except Exception as e:
    print(f'no: {e}')
" 2>/dev/null)
    echo "Gemma 4 support after upgrade: $GEMMA4_OK"
else
    echo "transformers already supports gemma4"
fi

# === Model selection based on VRAM ===
VRAM_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo ""
echo "=== Available VRAM: ${VRAM_MB} MB ==="

if [ "$VRAM_MB" -ge 20000 ]; then
    # Full A30 (24GB) — use 26B-A4B MoE (smart, fast, ~18GB bf16)
    MODEL="google/gemma-4-26B-A4B-it"
    echo "Using Gemma 4 26B-A4B-IT (MoE: 25B total, 4B active, ~18GB)"
elif [ "$VRAM_MB" -ge 12000 ]; then
    MODEL="google/gemma-4-E4B-it"
    echo "Using Gemma 4 E4B-IT (4.5B effective, ~10GB)"
else
    # MIG slice (~6GB) — use E2B
    MODEL="google/gemma-4-E2B-it"
    echo "Using Gemma 4 E2B-IT (2.3B effective, ~7GB) — smallest variant"
fi

PORT=8001

# === Start vLLM server ===
echo ""
echo "=== Starting vLLM server ==="
echo "Model: $MODEL"
echo "Endpoint: http://localhost:$PORT/v1"
echo "Node: $(hostname)"
echo ""
echo "To test:"
echo "  curl http://localhost:$PORT/v1/models"
echo ""
echo "To run pipeline with Gemma Tier 2:"
echo "  python -m pipeline_v4_gpu.main --video videos/A0.mp4 --output-root outputs_v4 \\"
echo "    --stage 6 --importance-backend local_llm --local-vlm-backend vllm \\"
echo "    --local-vlm-endpoint http://localhost:$PORT/v1 --local-vlm-model $MODEL"
echo ""
echo "Server starting at $(date)..."
echo "═══════════════════════════════════════════════════════════════"

python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port $PORT \
    --max-model-len 4096 \
    --trust-remote-code \
    --dtype bfloat16 \
    --gpu-memory-utilization 0.90

echo ""
echo "vLLM Gemma server stopped at $(date)"
