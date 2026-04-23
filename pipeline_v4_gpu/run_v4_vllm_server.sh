#!/bin/bash
#SBATCH --job-name=vllm_srv
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=vllm_server_%j.log
#SBATCH --error=vllm_server_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G

# ═══════════════════════════════════════════════════════════════════════════════
# vLLM Server for Tier 2 (Qwen2.5-VL-7B)
#
# Starts a vLLM OpenAI-compatible server on the GPU node.
# The pipeline connects to this at http://localhost:8000/v1
#
# NOTE: This job must be running on the SAME node as the pipeline job.
#       Use --dependency or run both in the same interactive session.
#
# Model options:
#   - Qwen2.5-VL-7B-Instruct  → fits on A30 (24GB), good quality
#   - Qwen2.5-VL-72B-Instruct-AWQ → needs 2x A30 or 1x A100 (40GB+)
#
# For A30 (24GB), we use the 7B model.
# ═══════════════════════════════════════════════════════════════════════════════

export PATH="/usr/local/cuda-12.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"

CONDA_ENV=/iitjhome/senthil1/.conda/envs/tc_pipeline
export PATH="$CONDA_ENV/bin:$PATH"

cd /iitjhome/senthil1

echo "═══════════════════════════════════════════════════════════════"
echo "  vLLM Server Setup"
echo "  $(date)"
echo "  Node: $(hostname)"
echo "═══════════════════════════════════════════════════════════════"

# === Step 1: Check GPU ===
echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# === Step 2: Install vLLM if needed ===
echo ""
echo "=== Checking vLLM installation ==="
if python -c "import vllm; print(f'vLLM version: {vllm.__version__}')" 2>/dev/null; then
    echo "vLLM already installed"
else
    echo "Installing vLLM (this may take 5-10 minutes)..."
    pip install vllm --quiet 2>&1 | tail -3
    python -c "import vllm; print(f'vLLM version: {vllm.__version__}')"
fi

# === Step 3: Model selection ===
# Check available VRAM and choose model
VRAM_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo ""
echo "=== Available VRAM: ${VRAM_MB} MB ==="

if [ "$VRAM_MB" -ge 40000 ]; then
    MODEL="Qwen/Qwen2.5-72B-Instruct-AWQ"
    echo "Using 72B-AWQ (sufficient VRAM)"
else
    MODEL="Qwen/Qwen2.5-7B-Instruct"
    echo "Using 7B text-only (A30 24GB — no vision encoder overhead)"
fi

# === Step 4: Start vLLM server ===
echo ""
echo "=== Starting vLLM server ==="
echo "Model: $MODEL"
echo "Endpoint: http://localhost:8000/v1"
echo "Node: $(hostname)"
echo ""
echo "To test from another terminal on the same node:"
echo "  curl http://localhost:8000/v1/models"
echo ""
echo "To run pipeline with Tier 2:"
echo "  python -m pipeline_v4_gpu.main --video videos/A0.mp4 --output-root outputs_v4 \\"
echo "    --stage 6 --importance-backend local_llm --local-vlm-backend vllm"
echo ""
echo "Server starting at $(date)..."
echo "═══════════════════════════════════════════════════════════════"

# Start the server (blocks until killed or job ends)
python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 4096 \
    --trust-remote-code \
    --dtype auto \
    --gpu-memory-utilization 0.75

echo ""
echo "vLLM server stopped at $(date)"
