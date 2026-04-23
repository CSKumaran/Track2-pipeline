#!/bin/bash
#SBATCH --job-name=v4_tier2
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=v4_tier2_test_%j.log
#SBATCH --error=v4_tier2_test_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

# ═══════════════════════════════════════════════════════════════════════════════
# V4 Tier 2 Test: Run Stage 6 with vLLM backend on A0
#
# PREREQUISITE: vLLM server must already be running on this node.
#               Start it first: sbatch run_v4_vllm_server.sh
#               Then submit this job to the SAME node:
#               sbatch --nodelist=cn24 run_v4_tier2_test.sh
#
# Alternative: Run both in one interactive session (see below).
# ═══════════════════════════════════════════════════════════════════════════════

export PATH="/usr/local/cuda-12.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"

CONDA_ENV=/iitjhome/senthil1/.conda/envs/tc_pipeline
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH=/iitjhome/senthil1
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0

cd /iitjhome/senthil1

if [ -f ~/.gemini_env ]; then
    source ~/.gemini_env
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  V4 Tier 2 Test (vLLM/Qwen2.5-VL)"
echo "  $(date)"
echo "  Node: $(hostname)"
echo "═══════════════════════════════════════════════════════════════"

# === Step 1: Check vLLM server is running ===
echo ""
echo "=== Checking vLLM server ==="
MAX_RETRIES=5
RETRY=0
VLLM_OK=false

while [ $RETRY -lt $MAX_RETRIES ]; do
    RESPONSE=$(curl -s http://localhost:8000/v1/models 2>/dev/null)
    if echo "$RESPONSE" | python -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null; then
        echo "  vLLM server is ready!"
        VLLM_OK=true
        break
    fi
    RETRY=$((RETRY + 1))
    echo "  Attempt $RETRY/$MAX_RETRIES — server not ready, waiting 30s..."
    sleep 30
done

if [ "$VLLM_OK" = false ]; then
    echo "  ERROR: vLLM server not reachable at http://localhost:8000/v1"
    echo "  Make sure run_v4_vllm_server.sh is running on this node ($(hostname))"
    echo ""
    echo "  Quick start (interactive):"
    echo "    srun --gres=gpu:1 --partition=fat --time=04:00:00 --pty bash"
    echo "    # In session: start vLLM in background, then run pipeline"
    echo "    python -m vllm.entrypoints.openai.api_server --model Qwen/Qwen2.5-VL-7B-Instruct --port 8000 &"
    echo "    sleep 60  # wait for model load"
    echo "    python -m pipeline_v4_gpu.main --video videos/A0.mp4 --output-root outputs_v4 --stage 6 --importance-backend local_llm --local-vlm-backend vllm"
    exit 1
fi

# === Step 2: Run Stage 6 with vLLM backend (A0 only) ===
VID="A0"
OUTDIR="outputs_v4/$VID"
TIER_DIR="$OUTDIR/tier_comparison"
mkdir -p "$TIER_DIR"

echo ""
echo "=== Running Stage 6 with vLLM (Tier 2) for $VID ==="

# Remove cached importance to force re-run
rm -f "$OUTDIR/pedagogical_importance.csv"

python -m pipeline_v4_gpu.main \
    --video "videos/$VID.mp4" \
    --output-root outputs_v4 \
    --stage 6 \
    --importance-backend local_llm \
    --local-vlm-backend vllm

if [ $? -ne 0 ]; then
    echo "ERROR: vLLM Stage 6 failed"
    # Restore Gemini importance
    cp "$TIER_DIR/importance_tier1_gemini.csv" "$OUTDIR/pedagogical_importance.csv" 2>/dev/null
    exit 1
fi

# Save Tier 2 results
cp "$OUTDIR/pedagogical_importance.csv" "$TIER_DIR/importance_tier2_vllm.csv"
echo "  Saved Tier 2 (vLLM) importance"

# Re-run Stage 7 with Tier 2 importance
echo ""
echo "=== Running Stage 7 with vLLM importance ==="
rm -f "$OUTDIR/keyword_scores.csv" "$OUTDIR/segment_scores.csv" "$OUTDIR/results.json"

python -m pipeline_v4_gpu.main \
    --video "videos/$VID.mp4" \
    --output-root outputs_v4 \
    --stage 7

cp "$OUTDIR/results.json" "$TIER_DIR/results_tier2_vllm.json" 2>/dev/null
cp "$OUTDIR/keyword_scores.csv" "$TIER_DIR/keyword_scores_tier2_vllm.csv" 2>/dev/null

# Restore Gemini as primary
echo ""
echo "=== Restoring Tier 1 (Gemini) as primary ==="
cp "$TIER_DIR/importance_tier1_gemini.csv" "$OUTDIR/pedagogical_importance.csv"
rm -f "$OUTDIR/keyword_scores.csv" "$OUTDIR/segment_scores.csv" "$OUTDIR/results.json"

python -m pipeline_v4_gpu.main \
    --video "videos/$VID.mp4" \
    --output-root outputs_v4 \
    --stage 7

# === Step 3: Compare all 3 tiers ===
echo ""
echo "######################################################################"
echo "# 3-TIER COMPARISON: $VID"
echo "######################################################################"

python -c "
import pandas as pd, json, os, numpy as np
from scipy.stats import spearmanr

vid = 'A0'
tc = f'outputs_v4/{vid}/tier_comparison'

tiers = {}
for tier_name, fname in [('T1_Gemini', 'importance_tier1_gemini.csv'),
                          ('T2_vLLM', 'importance_tier2_vllm.csv'),
                          ('T3_Heuristic', 'importance_tier3_heuristic.csv')]:
    path = f'{tc}/{fname}'
    if os.path.exists(path):
        tiers[tier_name] = pd.read_csv(path)

print(f'Available tiers: {list(tiers.keys())}')
print()

# --- Distribution ---
print('═══ Importance Distribution ═══')
print(f'{\"Tier\":<16} {\"1\":>5} {\"2\":>5} {\"3\":>5} {\"4\":>5} {\"5\":>5}  {\"Mean\":>6} {\"Std\":>5}')
print('─' * 60)
for name, df in tiers.items():
    dist = df['importance'].value_counts().reindex([1,2,3,4,5], fill_value=0)
    print(f'{name:<16} {dist[1]:>5} {dist[2]:>5} {dist[3]:>5} {dist[4]:>5} {dist[5]:>5}  {df[\"importance\"].mean():>6.2f} {df[\"importance\"].std():>5.2f}')

# --- Scores ---
print()
print('═══ Overall Scores ═══')
for tier_name, rfile in [('T1_Gemini', 'results_tier1_gemini.json'),
                          ('T2_vLLM', 'results_tier2_vllm.json'),
                          ('T3_Heuristic', 'results_tier3_heuristic.json')]:
    path = f'{tc}/{rfile}'
    if os.path.exists(path):
        r = json.load(open(path))
        print(f'  {tier_name:<16} Score: {r[\"overall_score\"]:>7.2f}  Grade: {r[\"overall_grade\"]}')

# --- Pairwise agreement ---
print()
print('═══ Pairwise Agreement (Spearman ρ) ═══')
tier_names = list(tiers.keys())
for i in range(len(tier_names)):
    for j in range(i+1, len(tier_names)):
        n1, n2 = tier_names[i], tier_names[j]
        merged = pd.merge(tiers[n1][['segment_id','importance']],
                          tiers[n2][['segment_id','importance']],
                          on='segment_id', suffixes=('_a','_b'))
        if len(merged) >= 3:
            rho, p = spearmanr(merged['importance_a'], merged['importance_b'])
            exact = (merged['importance_a'] == merged['importance_b']).mean() * 100
            near = (abs(merged['importance_a'] - merged['importance_b']) <= 1).mean() * 100
            print(f'  {n1} vs {n2}: ρ={rho:.3f} (p={p:.4f}), exact={exact:.1f}%, ±1={near:.1f}%')

print()
print('═══ INTERPRETATION ═══')
print('  T1 vs T2: Gemini vs vLLM — both are LLMs, expect moderate-high agreement')
print('  T1 vs T3: Gemini vs Heuristic — LLM vs features, expect lower agreement')
print('  T2 vs T3: vLLM vs Heuristic — same as above')
print('  If T1≈T2 >> T3: LLMs agree, heuristic is a rough approximation')
print('  If all similar: the task is well-defined, any backend works')
" 2>&1

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Tier 2 Test Complete"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════════════"
