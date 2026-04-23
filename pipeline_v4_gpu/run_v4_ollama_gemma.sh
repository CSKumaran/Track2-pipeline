#!/bin/bash
#SBATCH --job-name=ollama_gemma
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=ollama_gemma_%j.log
#SBATCH --error=ollama_gemma_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=16G

# ═══════════════════════════════════════════════════════════════════════════════
# Ollama Server for Tier 2b (Gemma 4) on port 11435
#
# WHY OLLAMA INSTEAD OF vLLM?
#   vLLM 0.11.2 (and current nightly) does not include a Gemma 4 model loader,
#   even after upgrading transformers to 5.6.0.dev0. Ollama supports Gemma 4
#   natively via 4-bit GGUF quantization — same models that run on laptops.
#
# Gemma 4 variants on Ollama (https://ollama.com/library/gemma4):
#   - gemma4:e2b   → 2.3B effective, ~3GB Q4 (fits MIG slice)
#   - gemma4:e4b   → 4.5B effective, ~5GB Q4 (fits MIG slice)
#   - gemma4:26b   → MoE 25B total / 4B active, ~18GB Q4 (fits A30 24GB)
#   - gemma4:31b   → 30.7B dense, ~23GB Q4 (tight on A30)
#
# Default: gemma4:26b (best smartness/speed tradeoff via MoE)
#
# Ollama serves an OpenAI-compatible API at http://localhost:11435/v1
# The pipeline's _tier2_vllm() reads cfg.LOCAL_VLM_ENDPOINT and works unchanged.
#
# Usage:
#   sbatch --nodelist=cn23 pipeline_v4_gpu/run_v4_ollama_gemma.sh
#   # Wait for "Ollama server READY" in log, then on the SAME node:
#   sbatch --nodelist=cn23 pipeline_v4_gpu/run_v4_tier2_gemma_batch.sh
# ═══════════════════════════════════════════════════════════════════════════════

export PATH="/usr/local/cuda-12.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"

CONDA_ENV=/iitjhome/senthil1/.conda/envs/tc_pipeline
export PATH="$CONDA_ENV/bin:$PATH"

cd /iitjhome/senthil1

echo "═══════════════════════════════════════════════════════════════"
echo "  Ollama Server — Gemma 4"
echo "  $(date)"
echo "  Node: $(hostname)"
echo "═══════════════════════════════════════════════════════════════"

# === GPU Info ===
echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# === Step 1: Install Ollama (standalone, no root) if missing ===
OLLAMA_DIR="$HOME/.local/ollama"
OLLAMA_BIN="$OLLAMA_DIR/bin/ollama"
export OLLAMA_MODELS="$HOME/.ollama/models"
mkdir -p "$OLLAMA_MODELS"

echo ""
echo "=== Checking Ollama installation ==="
if [ ! -x "$OLLAMA_BIN" ]; then
    echo "Ollama not found — please install on the LOGIN node first:"
    echo "  cd \$HOME/.local/ollama"
    echo "  curl -fL -o ollama.tar.zst \\"
    echo "    https://github.com/ollama/ollama/releases/download/v0.20.5/ollama-linux-amd64.tar.zst"
    echo "  tar -I zstd -xf ollama.tar.zst   # or: zstd -d ollama.tar.zst -o ollama.tar && tar -xf ollama.tar"
    echo "  ./bin/ollama --version"
    echo ""
    echo "(Compute nodes have limited internet — download must happen on the login node.)"
    exit 1
else
    echo "Ollama already installed at: $OLLAMA_BIN"
fi

"$OLLAMA_BIN" --version 2>&1 || true

# === Step 2: Pick port and start Ollama serve in background ===
export OLLAMA_HOST="0.0.0.0:11435"
export OLLAMA_KEEP_ALIVE="24h"
# Tell Ollama to use our scratch dir for model blobs (avoid quota issues)
export OLLAMA_MODELS="$HOME/.ollama/models"

echo ""
echo "=== Starting Ollama server on $OLLAMA_HOST ==="
"$OLLAMA_BIN" serve > ollama_server_${SLURM_JOB_ID}.log 2>&1 &
OLLAMA_PID=$!
echo "Ollama PID: $OLLAMA_PID"

# Wait for server to be ready
echo "Waiting for Ollama to come up..."
for i in $(seq 1 30); do
    if curl -s http://localhost:11435/api/tags >/dev/null 2>&1; then
        echo "Ollama responding after ${i}s"
        break
    fi
    sleep 1
done

if ! curl -s http://localhost:11435/api/tags >/dev/null 2>&1; then
    echo "ERROR: Ollama failed to start. Last log lines:"
    tail -50 ollama_server_${SLURM_JOB_ID}.log
    exit 1
fi

# === Step 3: Pick model based on VRAM ===
VRAM_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1 | tr -d ' ')
echo ""
echo "=== Available VRAM: ${VRAM_MB} MB ==="

if [ "$VRAM_MB" -ge 20000 ]; then
    MODEL="gemma4:26b"
    echo "Using gemma4:26b (MoE 25B total / 4B active, ~18GB Q4)"
elif [ "$VRAM_MB" -ge 6000 ]; then
    MODEL="gemma4:e4b"
    echo "Using gemma4:e4b (4.5B effective, ~5GB Q4)"
else
    MODEL="gemma4:e2b"
    echo "Using gemma4:e2b (2.3B effective, ~3GB Q4) — smallest variant"
fi

# === Step 4: Pull the model (if not cached) ===
echo ""
echo "=== Pulling $MODEL ==="
"$OLLAMA_BIN" pull "$MODEL"
if [ $? -ne 0 ]; then
    echo "ERROR: ollama pull failed for $MODEL"
    kill $OLLAMA_PID 2>/dev/null
    exit 1
fi

# === Step 5: Verify model is loaded and OpenAI endpoint works ===
echo ""
echo "=== Testing OpenAI-compatible endpoint ==="
curl -s http://localhost:11435/v1/models | head -100
echo ""
TEST_OUT=$(curl -s http://localhost:11435/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"Reply with the single word OK.\"}],\"max_tokens\":10}")
echo "Test response: $TEST_OUT"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Ollama server READY"
echo "  Model:    $MODEL"
echo "  Endpoint: http://localhost:11435/v1"
echo "  Node:     $(hostname)"
echo "═══════════════════════════════════════════════════════════════"
echo ""
echo "To run pipeline with Gemma Tier 2 (from this node):"
echo "  python -m pipeline_v4_gpu.main --video videos/A0.mp4 --output-root outputs_v4 \\"
echo "    --stage 6 --importance-backend local_llm --local-vlm-backend vllm \\"
echo "    --local-vlm-endpoint http://localhost:11435/v1 --local-vlm-model $MODEL"
echo ""
echo "Or submit the batch comparison (same node):"
echo "  sbatch --nodelist=$(hostname) pipeline_v4_gpu/run_v4_tier2_gemma_batch.sh"
echo ""

# === Step 6: Block until job time expires (or killed) ===
wait $OLLAMA_PID

echo ""
echo "Ollama server stopped at $(date)"
