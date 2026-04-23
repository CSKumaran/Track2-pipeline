#!/bin/bash
#SBATCH --job-name=v4_gemma_cmp
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=v4_gemma_batch_%j.log
#SBATCH --error=v4_gemma_batch_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

# ═══════════════════════════════════════════════════════════════════════════════
# V4: Tier 2b (Gemma 4) Batch — Importance Comparison
#
# PREREQUISITE: Gemma Ollama server on port 11435 (same node).
#               sbatch --nodelist=cn23 pipeline_v4_gpu/run_v4_ollama_gemma.sh
#
# (vLLM 0.11.2 doesn't support Gemma 4 yet — we use Ollama instead.
#  The pipeline's vllm backend talks plain OpenAI API, so it works unchanged
#  pointed at the Ollama OpenAI-compatible endpoint.)
#
# Reuses existing Tier 1 (Gemini), Tier 2 (Qwen), Tier 3 (Heuristic) data.
# Only runs Gemma Stage 6 + Stage 7, saves to tier_comparison/.
#
# Final output: 4-tier comparison table (Gemini vs Qwen vs Gemma vs Heuristic)
# ═══════════════════════════════════════════════════════════════════════════════

export PATH="/usr/local/cuda-12.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"

CONDA_ENV=/iitjhome/senthil1/.conda/envs/tc_pipeline
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH=/iitjhome/senthil1
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0

cd /iitjhome/senthil1

find pipeline_v4_gpu/ -name "__pycache__" -exec rm -rf {} + 2>/dev/null

GEMMA_PORT=11435
GEMMA_ENDPOINT="http://localhost:$GEMMA_PORT/v1"
# Auto-detect from running Ollama server (so batch script matches whichever
# Gemma 4 variant the server selected based on VRAM: 26b / e4b / e2b)
GEMMA_MODEL=$(curl -s "$GEMMA_ENDPOINT/models" 2>/dev/null | python -c "import sys,json; print(json.load(sys.stdin)['data'][0]['id'])" 2>/dev/null)
if [ -z "$GEMMA_MODEL" ]; then
    GEMMA_MODEL="gemma4:26b"  # default fallback
fi

echo "═══════════════════════════════════════════════════════════════"
echo "  V4 Tier 2b: Gemma 4 (Ollama) — Importance Comparison"
echo "  $(date)"
echo "  Node:     $(hostname)"
echo "  Endpoint: $GEMMA_ENDPOINT"
echo "  Model:    $GEMMA_MODEL"
echo "═══════════════════════════════════════════════════════════════"

# === Check Gemma Ollama server ===
echo ""
echo "=== Checking Gemma Ollama server on port $GEMMA_PORT ==="
MAX_RETRIES=10
RETRY=0
OLLAMA_OK=false

while [ $RETRY -lt $MAX_RETRIES ]; do
    RESPONSE=$(curl -s "$GEMMA_ENDPOINT/models" 2>/dev/null)
    if echo "$RESPONSE" | python -c "import sys,json; d=json.load(sys.stdin); print(d['data'][0]['id'])" 2>/dev/null; then
        echo "  Gemma Ollama server is ready!"
        OLLAMA_OK=true
        break
    fi
    RETRY=$((RETRY + 1))
    echo "  Attempt $RETRY/$MAX_RETRIES — server not ready, waiting 30s..."
    sleep 30
done

if [ "$OLLAMA_OK" = false ]; then
    echo "  ERROR: Gemma Ollama server not reachable at $GEMMA_ENDPOINT"
    echo "  Start it first: sbatch --nodelist=$(hostname) pipeline_v4_gpu/run_v4_ollama_gemma.sh"
    exit 1
fi

# === Find videos with existing tier_comparison data ===
# Run on evaluation videos first (condition differentiation matters most)
ALL_VIDEOS=(
    "Video_evaluation_01" "Video_evaluation_02" "Video_evaluation_03" "Video_evaluation_04"
    "CTML_03_01" "CTML_03_02" "CTML_03_03" "CTML_03_04"
    "A0" "A1" "A3" "A5"
    "B0" "B1" "B3" "B5"
)

VIDEOS=()
for VID in "${ALL_VIDEOS[@]}"; do
    if [ -f "outputs_v4/$VID/results.json" ] && [ -f "outputs_v4/$VID/pedagogical_importance.csv" ]; then
        VIDEOS+=("$VID")
    fi
done

echo ""
echo "Videos with V4 outputs: ${VIDEOS[*]}"
echo "Count: ${#VIDEOS[@]}"

if [ ${#VIDEOS[@]} -eq 0 ]; then
    echo "ERROR: No V4 outputs found."
    exit 1
fi

# === Run Gemma Tier 2b for each video ===
N_SUCCESS=0
N_FAIL=0

for VID in "${VIDEOS[@]}"; do
    echo ""
    echo "--- $VID: Tier 2b (Gemma) ---"

    OUTDIR="outputs_v4/$VID"
    TIER_DIR="$OUTDIR/tier_comparison"
    mkdir -p "$TIER_DIR"

    # Skip if Gemma already done
    if [ -f "$TIER_DIR/importance_tier2b_gemma.csv" ]; then
        echo "  Already has Gemma data — skipping"
        N_SUCCESS=$((N_SUCCESS + 1))
        continue
    fi

    # Save current importance (backup)
    cp "$OUTDIR/pedagogical_importance.csv" "$OUTDIR/pedagogical_importance.csv.bak"

    # Run Stage 6 with Gemma
    rm -f "$OUTDIR/pedagogical_importance.csv"
    python -m pipeline_v4_gpu.main \
        --video "videos/$VID.mp4" \
        --output-root outputs_v4 \
        --stage 6 \
        --importance-backend local_llm \
        --local-vlm-backend vllm \
        --local-vlm-endpoint "$GEMMA_ENDPOINT" \
        --local-vlm-model "$GEMMA_MODEL"

    if [ $? -ne 0 ]; then
        echo "  ERROR: Gemma Stage 6 failed for $VID"
        # Restore backup
        cp "$OUTDIR/pedagogical_importance.csv.bak" "$OUTDIR/pedagogical_importance.csv"
        N_FAIL=$((N_FAIL + 1))
        continue
    fi

    # Check backend
    BACKEND=$(head -2 "$OUTDIR/pedagogical_importance.csv" | tail -1 | grep -o 'local_llm_vllm\|heuristic\|gemini')
    if [ "$BACKEND" = "heuristic" ]; then
        echo "  WARNING: Fell through to heuristic (Gemma may have failed)"
    fi

    # Save Gemma results
    cp "$OUTDIR/pedagogical_importance.csv" "$TIER_DIR/importance_tier2b_gemma.csv"
    echo "  Saved Tier 2b (Gemma) importance (backend: $BACKEND)"

    # Re-run Stage 7 with Gemma importance
    rm -f "$OUTDIR/keyword_scores.csv" "$OUTDIR/segment_scores.csv" "$OUTDIR/results.json"
    python -m pipeline_v4_gpu.main \
        --video "videos/$VID.mp4" \
        --output-root outputs_v4 \
        --stage 7 \
        --dwell-threshold 15.0 \
        --dwell-tau 30.0

    cp "$OUTDIR/results.json" "$TIER_DIR/results_tier2b_gemma.json" 2>/dev/null
    cp "$OUTDIR/keyword_scores.csv" "$TIER_DIR/keyword_scores_tier2b_gemma.csv" 2>/dev/null

    # Restore original importance (Gemini/Tier1)
    cp "$OUTDIR/pedagogical_importance.csv.bak" "$OUTDIR/pedagogical_importance.csv"
    rm -f "$OUTDIR/pedagogical_importance.csv.bak"

    # Restore original Stage 7 results
    rm -f "$OUTDIR/keyword_scores.csv" "$OUTDIR/segment_scores.csv" "$OUTDIR/results.json"
    python -m pipeline_v4_gpu.main \
        --video "videos/$VID.mp4" \
        --output-root outputs_v4 \
        --stage 7 \
        --dwell-threshold 15.0 \
        --dwell-tau 30.0

    N_SUCCESS=$((N_SUCCESS + 1))
    echo "  $VID done"
done

# ═══════════════════════════════════════════════════════════════════════════════
# 4-TIER COMPARISON
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "######################################################################"
echo "# 4-TIER COMPARISON: Gemini vs Qwen vs Gemma vs Heuristic"
echo "######################################################################"

python -c "
import pandas as pd, json, os, numpy as np
from scipy.stats import spearmanr

tier_files = {
    'T1_Gemini':    'importance_tier1_gemini.csv',
    'T2_Qwen':      'importance_tier2_vllm.csv',
    'T2b_Gemma':    'importance_tier2b_gemma.csv',
    'T3_Heuristic': 'importance_tier3_heuristic.csv',
}
result_files = {
    'T1_Gemini':    'results_tier1_gemini.json',
    'T2_Qwen':      'results_tier2_vllm.json',
    'T2b_Gemma':    'results_tier2b_gemma.json',
    'T3_Heuristic': 'results_tier3_heuristic.json',
}

all_vids = ['Video_evaluation_01','Video_evaluation_02','Video_evaluation_03','Video_evaluation_04',
            'CTML_03_01','CTML_03_02','CTML_03_03','CTML_03_04',
            'A0','A1','A3','A5','B0','B1','B3','B5']

# Find videos with all 4 tiers
videos_4tier = []
videos_3tier = []
for v in all_vids:
    tc = f'outputs_v4/{v}/tier_comparison'
    has_all = all(os.path.exists(f'{tc}/{f}') for f in tier_files.values())
    has_3 = all(os.path.exists(f'{tc}/{tier_files[t]}') for t in ['T1_Gemini','T2_Qwen','T3_Heuristic'])
    if has_all:
        videos_4tier.append(v)
    elif has_3:
        videos_3tier.append(v)

print(f'Videos with all 4 tiers: {len(videos_4tier)}')
print(f'Videos with 3 tiers (no Gemma): {len(videos_3tier)}')
print()

# Use videos that have Gemma data
videos = videos_4tier if videos_4tier else []
if not videos:
    print('No complete 4-tier comparison data found.')
    exit()

# ─── Overall Score Table ───
print('=' * 75)
print('OVERALL SCORES BY TIER')
print('=' * 75)
print(f'{\"Video\":<24} {\"T1 Gemini\":>10} {\"T2 Qwen\":>10} {\"T2b Gemma\":>10} {\"T3 Heur\":>10}')
print('-' * 75)

for vid in videos:
    tc = f'outputs_v4/{vid}/tier_comparison'
    row = f'{vid:<24}'
    for tier, fname in result_files.items():
        path = f'{tc}/{fname}'
        try:
            r = json.load(open(path))
            row += f' {r[\"overall_score\"]:>10.2f}'
        except:
            row += f' {\"N/A\":>10}'
    print(row)

# ─── Importance Distribution ───
print()
print('=' * 75)
print('IMPORTANCE DISTRIBUTION (all videos pooled)')
print('=' * 75)
for tier_name, fname in tier_files.items():
    all_imp = []
    for vid in videos:
        path = f'outputs_v4/{vid}/tier_comparison/{fname}'
        if os.path.exists(path):
            df = pd.read_csv(path)
            if 'importance' in df.columns:
                all_imp.extend(df['importance'].dropna().astype(int).tolist())
    if all_imp:
        s = pd.Series(all_imp)
        dist = s.value_counts().reindex([1,2,3,4,5], fill_value=0)
        pct = dist / len(all_imp) * 100
        print(f'  {tier_name:<16} 1:{pct[1]:>5.1f}%  2:{pct[2]:>5.1f}%  3:{pct[3]:>5.1f}%  4:{pct[4]:>5.1f}%  5:{pct[5]:>5.1f}%  mean={s.mean():.2f}  std={s.std():.2f}  n={len(all_imp)}')

# ─── Pairwise Agreement ───
print()
print('=' * 75)
print('PAIRWISE AGREEMENT (Spearman rho, exact match %, +/-1 match %)')
print('=' * 75)

tier_names = list(tier_files.keys())
for i in range(len(tier_names)):
    for j in range(i+1, len(tier_names)):
        n1, n2 = tier_names[i], tier_names[j]
        f1, f2 = tier_files[n1], tier_files[n2]
        rhos, exacts, nears = [], [], []
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
            print(f'  {n1:>16} vs {n2:<16}  rho={np.mean(rhos):.3f}  exact={np.mean(exacts):.1f}%  +/-1={np.mean(nears):.1f}%')

# ─── Condition Differentiation ───
print()
print('=' * 75)
print('CONDITION DIFFERENTIATION')
print('=' * 75)

conditions = {'01': 'Simultaneous', '02': 'Anim first', '03': 'Narr first', '04': 'Segmented'}
for prefix, label in [('Video_evaluation', 'VIDEO EVALUATION'), ('CTML_03', 'CTML_03')]:
    print(f'')
    print(f'--- {label} ---')
    print(f'{\"Tier\":<16} {\"01(Simul)\":>10} {\"02(Anim)\":>10} {\"03(Narr)\":>10} {\"04(Seg)\":>10}   Ranking')
    for tier_name, rfname in result_files.items():
        scores = {}
        for cond in ['01','02','03','04']:
            vid = f'{prefix}_{cond}'
            path = f'outputs_v4/{vid}/tier_comparison/{rfname}'
            try:
                r = json.load(open(path))
                scores[cond] = r['overall_score']
            except:
                pass
        if len(scores) >= 2:
            row = f'{tier_name:<16}'
            for cond in ['01','02','03','04']:
                if cond in scores:
                    row += f' {scores[cond]:>10.2f}'
                else:
                    row += f' {\"N/A\":>10}'
            ranking = sorted(scores.items(), key=lambda x: -x[1])
            rank_str = ' > '.join(f'{c}({s:.0f})' for c, s in ranking)
            ok = scores.get('01', 0) > scores.get('02', 0)
            row += f'   {rank_str}  {\"OK\" if ok else \"PROBLEM\"}'
            print(row)

# ─── Per-Segment Disagreement (Qwen vs Gemma) ───
print()
print('=' * 75)
print('QWEN vs GEMMA: SEGMENTS WITH LARGEST DISAGREEMENT')
print('=' * 75)
print(f'{\"Video\":<24} {\"Seg\":>4} {\"Qwen\":>5} {\"Gemma\":>6} {\"Diff\":>5}  Transcript (first 60 chars)')
print('-' * 110)

disagreements = []
for vid in videos:
    tc = f'outputs_v4/{vid}/tier_comparison'
    p_q = f'{tc}/importance_tier2_vllm.csv'
    p_g = f'{tc}/importance_tier2b_gemma.csv'
    if os.path.exists(p_q) and os.path.exists(p_g):
        dq = pd.read_csv(p_q)
        dg = pd.read_csv(p_g)
        m = pd.merge(dq[['segment_id','importance']], dg[['segment_id','importance']],
                     on='segment_id', suffixes=('_qwen','_gemma'))
        for _, r in m.iterrows():
            diff = abs(r['importance_qwen'] - r['importance_gemma'])
            if diff >= 2:
                # Get transcript from segments.csv
                seg_path = f'outputs_v4/{vid}/segments.csv'
                text = ''
                if os.path.exists(seg_path):
                    segs = pd.read_csv(seg_path)
                    seg_row = segs[segs['segment_id'] == r['segment_id']]
                    if len(seg_row) > 0:
                        text = str(seg_row.iloc[0].get('text', ''))[:60]
                disagreements.append((vid, int(r['segment_id']), int(r['importance_qwen']),
                                     int(r['importance_gemma']), diff, text))

disagreements.sort(key=lambda x: -x[4])
for vid, sid, q, g, d, txt in disagreements[:20]:
    print(f'{vid:<24} {sid:>4} {q:>5} {g:>6} {d:>5}  {txt}')
if not disagreements:
    print('  No segments with disagreement >= 2')

# ─── A-Series Monotonicity Check ───
print()
print('=' * 75)
print('A-SERIES MONOTONICITY CHECK (per tier)')
print('=' * 75)
for tier_name, rfname in result_files.items():
    a_scores = []
    for vid in ['A0','A1','A3','A5']:
        path = f'outputs_v4/{vid}/tier_comparison/{rfname}'
        try:
            r = json.load(open(path))
            a_scores.append(r['overall_score'])
        except:
            a_scores.append(None)
    if all(s is not None for s in a_scores):
        mono = all(a_scores[i] >= a_scores[i+1] for i in range(3))
        print(f'  {tier_name:<16} A0={a_scores[0]:.1f}  A1={a_scores[1]:.1f}  A3={a_scores[2]:.1f}  A5={a_scores[3]:.1f}  Monotonic: {\"YES\" if mono else \"NO\"}')" 2>&1

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Gemma Batch Complete: $N_SUCCESS success, $N_FAIL failed"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════════════"
