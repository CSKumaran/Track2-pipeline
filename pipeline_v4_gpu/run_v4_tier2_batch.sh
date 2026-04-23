#!/bin/bash
#SBATCH --job-name=v4_t2_all
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=v4_tier2_batch_%j.log
#SBATCH --error=v4_tier2_batch_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

# ═══════════════════════════════════════════════════════════════════════════════
# V4: Tier 2 (vLLM/Qwen2.5-7B) Batch — All Videos
#
# PREREQUISITE: vLLM server must be running on the SAME node.
#               sbatch --nodelist=cn24 pipeline_v4_gpu/run_v4_vllm_server.sh
#
# For each video with existing V4 outputs:
#   1. Save existing Gemini importance as Tier 1 (if not already saved)
#   2. Run Stage 6 with vLLM backend (Tier 2)
#   3. Save Tier 2 importance + re-run Stage 7
#   4. Restore Gemini (Tier 1) as primary
#
# Results saved to outputs_v4/<VID>/tier_comparison/
# ═══════════════════════════════════════════════════════════════════════════════

# === Environment Setup ===
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

find pipeline_v4_gpu/ -name "__pycache__" -exec rm -rf {} + 2>/dev/null

echo "═══════════════════════════════════════════════════════════════"
echo "  V4 Tier 2 Batch: vLLM (Qwen2.5-7B) — All Videos"
echo "  $(date)"
echo "  Node: $(hostname)"
echo "═══════════════════════════════════════════════════════════════"

# === Step 1: Check vLLM server ===
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
    echo "  Start it first: sbatch --nodelist=$(hostname) pipeline_v4_gpu/run_v4_vllm_server.sh"
    exit 1
fi

# === Step 2: Find videos with V4 outputs ===
VIDEOS=()
for VID in A0 A1 A3 A5 B0 B1 B3 B5 C0 C1 C3 C5 D0 D1 D3 D5; do
    if [ -f "outputs_v4/$VID/results.json" ] && [ -f "outputs_v4/$VID/pedagogical_importance.csv" ]; then
        VIDEOS+=("$VID")
    fi
done

echo ""
echo "Videos with V4 outputs: ${VIDEOS[*]}"
echo "Count: ${#VIDEOS[@]}"

if [ ${#VIDEOS[@]} -eq 0 ]; then
    echo "ERROR: No V4 outputs found. Run the full pipeline first."
    exit 1
fi

# === Step 3: Run Tier 2 for each video ===
N_SUCCESS=0
N_FAIL=0

for VID in "${VIDEOS[@]}"; do
    echo ""
    echo "--- $VID: Tier 2 (vLLM) ---"

    OUTDIR="outputs_v4/$VID"
    TIER_DIR="$OUTDIR/tier_comparison"
    mkdir -p "$TIER_DIR"

    # Save Tier 1 (Gemini) if not already saved
    if [ ! -f "$TIER_DIR/importance_tier1_gemini.csv" ]; then
        cp "$OUTDIR/pedagogical_importance.csv" "$TIER_DIR/importance_tier1_gemini.csv"
        cp "$OUTDIR/results.json" "$TIER_DIR/results_tier1_gemini.json" 2>/dev/null
        cp "$OUTDIR/keyword_scores.csv" "$TIER_DIR/keyword_scores_tier1_gemini.csv" 2>/dev/null
        echo "  Saved Tier 1 (Gemini) importance"
    fi

    # Save Tier 3 (Heuristic) if not already saved
    if [ ! -f "$TIER_DIR/importance_tier3_heuristic.csv" ]; then
        rm -f "$OUTDIR/pedagogical_importance.csv"
        python -m pipeline_v4_gpu.main \
            --video "videos/$VID.mp4" \
            --output-root outputs_v4 \
            --stage 6 \
            --importance-backend heuristic
        cp "$OUTDIR/pedagogical_importance.csv" "$TIER_DIR/importance_tier3_heuristic.csv"

        # Stage 7 with heuristic
        rm -f "$OUTDIR/keyword_scores.csv" "$OUTDIR/segment_scores.csv" "$OUTDIR/results.json"
        python -m pipeline_v4_gpu.main \
            --video "videos/$VID.mp4" \
            --output-root outputs_v4 \
            --stage 7
        cp "$OUTDIR/results.json" "$TIER_DIR/results_tier3_heuristic.json" 2>/dev/null
        cp "$OUTDIR/keyword_scores.csv" "$TIER_DIR/keyword_scores_tier3_heuristic.csv" 2>/dev/null
        echo "  Saved Tier 3 (Heuristic) importance"
    fi

    # Run Tier 2 (vLLM)
    rm -f "$OUTDIR/pedagogical_importance.csv"
    python -m pipeline_v4_gpu.main \
        --video "videos/$VID.mp4" \
        --output-root outputs_v4 \
        --stage 6 \
        --importance-backend local_llm \
        --local-vlm-backend vllm

    if [ $? -ne 0 ]; then
        echo "  ERROR: vLLM Stage 6 failed for $VID"
        cp "$TIER_DIR/importance_tier1_gemini.csv" "$OUTDIR/pedagogical_importance.csv"
        N_FAIL=$((N_FAIL + 1))
        continue
    fi

    # Check if vLLM was actually used (not heuristic fallback)
    BACKEND=$(head -2 "$OUTDIR/pedagogical_importance.csv" | tail -1 | grep -o 'local_llm_vllm\|heuristic\|gemini')
    if [ "$BACKEND" = "heuristic" ]; then
        echo "  WARNING: Fell through to heuristic (vLLM may have failed)"
    fi

    # Save Tier 2 results
    cp "$OUTDIR/pedagogical_importance.csv" "$TIER_DIR/importance_tier2_vllm.csv"
    echo "  Saved Tier 2 (vLLM) importance (backend: $BACKEND)"

    # Re-run Stage 7 with Tier 2 importance
    rm -f "$OUTDIR/keyword_scores.csv" "$OUTDIR/segment_scores.csv" "$OUTDIR/results.json"
    python -m pipeline_v4_gpu.main \
        --video "videos/$VID.mp4" \
        --output-root outputs_v4 \
        --stage 7

    cp "$OUTDIR/results.json" "$TIER_DIR/results_tier2_vllm.json" 2>/dev/null
    cp "$OUTDIR/keyword_scores.csv" "$TIER_DIR/keyword_scores_tier2_vllm.csv" 2>/dev/null

    # Restore Gemini (Tier 1) as primary
    cp "$TIER_DIR/importance_tier1_gemini.csv" "$OUTDIR/pedagogical_importance.csv"
    rm -f "$OUTDIR/keyword_scores.csv" "$OUTDIR/segment_scores.csv" "$OUTDIR/results.json"
    python -m pipeline_v4_gpu.main \
        --video "videos/$VID.mp4" \
        --output-root outputs_v4 \
        --stage 7

    N_SUCCESS=$((N_SUCCESS + 1))
    echo "  $VID done"
done

# === Step 4: 3-Tier Comparison ===
echo ""
echo "######################################################################"
echo "# 3-TIER COMPARISON: ALL VIDEOS"
echo "######################################################################"

python -c "
import pandas as pd, json, os, numpy as np
from scipy.stats import spearmanr

videos = []
for v in ['A0','A1','A3','A5','B0','B1','B3','B5','C0','C1','C3','C5','D0','D1','D3','D5']:
    tc = f'outputs_v4/{v}/tier_comparison'
    if os.path.exists(f'{tc}/importance_tier1_gemini.csv') and \
       os.path.exists(f'{tc}/importance_tier2_vllm.csv') and \
       os.path.exists(f'{tc}/importance_tier3_heuristic.csv'):
        videos.append(v)

if not videos:
    print('No complete 3-tier comparison data found.')
    exit()

print(f'Videos with all 3 tiers: {len(videos)}')
print()

# ─── Score Table ───
print('═══ Overall Score by Tier ═══')
print(f'{\"Video\":<8} {\"T1 Gemini\":>10} {\"T2 vLLM\":>10} {\"T3 Heur\":>10} {\"T1-T3\":>8}')
print('─' * 50)

for vid in videos:
    tc = f'outputs_v4/{vid}/tier_comparison'
    try:
        r1 = json.load(open(f'{tc}/results_tier1_gemini.json'))
        r2 = json.load(open(f'{tc}/results_tier2_vllm.json'))
        r3 = json.load(open(f'{tc}/results_tier3_heuristic.json'))
        print(f'{vid:<8} {r1[\"overall_score\"]:>10.2f} {r2[\"overall_score\"]:>10.2f} {r3[\"overall_score\"]:>10.2f} {r1[\"overall_score\"]-r3[\"overall_score\"]:>+8.2f}')
    except Exception as e:
        print(f'{vid:<8} Error: {e}')

# ─── Distribution Table ───
print()
print('═══ Importance Distribution (Mean across videos) ═══')
for tier_name, fname in [('T1_Gemini', 'importance_tier1_gemini.csv'),
                          ('T2_vLLM', 'importance_tier2_vllm.csv'),
                          ('T3_Heuristic', 'importance_tier3_heuristic.csv')]:
    all_imp = []
    for vid in videos:
        path = f'outputs_v4/{vid}/tier_comparison/{fname}'
        if os.path.exists(path):
            df = pd.read_csv(path)
            all_imp.extend(df['importance'].tolist())
    if all_imp:
        s = pd.Series(all_imp)
        dist = s.value_counts().reindex([1,2,3,4,5], fill_value=0)
        pct = dist / len(all_imp) * 100
        print(f'  {tier_name:<16} 1:{pct[1]:>5.1f}%  2:{pct[2]:>5.1f}%  3:{pct[3]:>5.1f}%  4:{pct[4]:>5.1f}%  5:{pct[5]:>5.1f}%  mean={s.mean():.2f}  std={s.std():.2f}')

# ─── Pairwise Agreement ───
print()
print('═══ Pairwise Agreement (Mean Spearman rho) ═══')
pairs = [('T1_Gemini', 'T2_vLLM', 'importance_tier1_gemini.csv', 'importance_tier2_vllm.csv'),
         ('T1_Gemini', 'T3_Heuristic', 'importance_tier1_gemini.csv', 'importance_tier3_heuristic.csv'),
         ('T2_vLLM', 'T3_Heuristic', 'importance_tier2_vllm.csv', 'importance_tier3_heuristic.csv')]

for n1, n2, f1, f2 in pairs:
    rhos = []
    exacts = []
    nears = []
    for vid in videos:
        tc = f'outputs_v4/{vid}/tier_comparison'
        p1, p2 = f'{tc}/{f1}', f'{tc}/{f2}'
        if os.path.exists(p1) and os.path.exists(p2):
            d1 = pd.read_csv(p1)
            d2 = pd.read_csv(p2)
            m = pd.merge(d1[['segment_id','importance']], d2[['segment_id','importance']],
                         on='segment_id', suffixes=('_a','_b'))
            if len(m) >= 3:
                rho, _ = spearmanr(m['importance_a'], m['importance_b'])
                rhos.append(rho)
                exacts.append((m['importance_a'] == m['importance_b']).mean() * 100)
                nears.append((abs(m['importance_a'] - m['importance_b']) <= 1).mean() * 100)
    if rhos:
        print(f'  {n1} vs {n2}: rho={np.mean(rhos):.3f}  exact={np.mean(exacts):.1f}%  +/-1={np.mean(nears):.1f}%')
" 2>&1

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Tier 2 Batch Complete: $N_SUCCESS success, $N_FAIL failed"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════════════"
