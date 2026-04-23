#!/bin/bash
#SBATCH --job-name=v4_tiers
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=v4_tier_comparison_%j.log
#SBATCH --error=v4_tier_comparison_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

# ═══════════════════════════════════════════════════════════════════════════════
# V4 Stage 6 Tier Comparison: Gemini (Tier 1) vs Heuristic (Tier 3)
#
# For each video that already has V4 outputs (from previous runs):
#   1. Save existing Gemini importance ratings (Tier 1)
#   2. Force re-run Stage 6+7 with heuristic backend (Tier 3)
#   3. Compare importance distributions, scores, and inter-rater agreement
#
# Tier 2 (vLLM/Qwen2.5-VL) requires a running vLLM server — tested separately.
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
echo "  V4 Tier Comparison: Gemini (T1) vs Heuristic (T3)"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════════════"

# Which videos already have V4 outputs?
VIDEOS=()
for VID in A0 A5 A1 A3 B0 B1 B3 B5 C0 C1 C3 C5 D0 D1 D3 D5; do
    if [ -f "outputs_v4/$VID/results.json" ]; then
        VIDEOS+=("$VID")
    fi
done

echo "Videos with V4 outputs: ${VIDEOS[*]}"
echo "Count: ${#VIDEOS[@]}"

if [ ${#VIDEOS[@]} -eq 0 ]; then
    echo "ERROR: No V4 outputs found. Run the pipeline first."
    exit 1
fi

# ═══════════════════════════════════════════════════════════════════════════════
# For each video: save Tier 1, re-run with Tier 3, then restore Tier 1
# ═══════════════════════════════════════════════════════════════════════════════

for VID in "${VIDEOS[@]}"; do
    echo ""
    echo "######################################################################"
    echo "# $VID — Tier Comparison"
    echo "######################################################################"

    OUTDIR="outputs_v4/$VID"
    TIER_DIR="$OUTDIR/tier_comparison"
    mkdir -p "$TIER_DIR"

    # --- Step 1: Save existing Tier 1 (Gemini) results ---
    if [ -f "$OUTDIR/pedagogical_importance.csv" ]; then
        cp "$OUTDIR/pedagogical_importance.csv" "$TIER_DIR/importance_tier1_gemini.csv"
        echo "  Saved Tier 1 (Gemini) importance"
    else
        echo "  WARNING: No existing importance file for $VID — skipping"
        continue
    fi

    # Save existing scores too
    cp "$OUTDIR/results.json" "$TIER_DIR/results_tier1_gemini.json" 2>/dev/null
    cp "$OUTDIR/keyword_scores.csv" "$TIER_DIR/keyword_scores_tier1_gemini.csv" 2>/dev/null
    cp "$OUTDIR/segment_scores.csv" "$TIER_DIR/segment_scores_tier1_gemini.csv" 2>/dev/null

    # --- Step 2: Re-run Stage 6 with heuristic backend ---
    echo "  Running Stage 6 with heuristic backend..."
    rm -f "$OUTDIR/pedagogical_importance.csv"

    python -m pipeline_v4_gpu.main \
        --video "videos/$VID.mp4" \
        --output-root outputs_v4 \
        --stage 6 \
        --importance-backend heuristic

    if [ $? -ne 0 ]; then
        echo "  ERROR: Heuristic Stage 6 failed for $VID"
        # Restore Gemini results
        cp "$TIER_DIR/importance_tier1_gemini.csv" "$OUTDIR/pedagogical_importance.csv"
        continue
    fi

    # Save Tier 3 importance
    cp "$OUTDIR/pedagogical_importance.csv" "$TIER_DIR/importance_tier3_heuristic.csv"
    echo "  Saved Tier 3 (Heuristic) importance"

    # --- Step 3: Re-run Stage 7 with heuristic importance ---
    echo "  Running Stage 7 with heuristic importance..."
    rm -f "$OUTDIR/keyword_scores.csv" "$OUTDIR/segment_scores.csv" "$OUTDIR/results.json"

    python -m pipeline_v4_gpu.main \
        --video "videos/$VID.mp4" \
        --output-root outputs_v4 \
        --stage 7

    # Save Tier 3 scores
    cp "$OUTDIR/results.json" "$TIER_DIR/results_tier3_heuristic.json" 2>/dev/null
    cp "$OUTDIR/keyword_scores.csv" "$TIER_DIR/keyword_scores_tier3_heuristic.csv" 2>/dev/null
    cp "$OUTDIR/segment_scores.csv" "$TIER_DIR/segment_scores_tier3_heuristic.csv" 2>/dev/null

    # --- Step 4: Restore Gemini (Tier 1) as primary ---
    echo "  Restoring Tier 1 (Gemini) as primary..."
    cp "$TIER_DIR/importance_tier1_gemini.csv" "$OUTDIR/pedagogical_importance.csv"
    rm -f "$OUTDIR/keyword_scores.csv" "$OUTDIR/segment_scores.csv" "$OUTDIR/results.json"

    python -m pipeline_v4_gpu.main \
        --video "videos/$VID.mp4" \
        --output-root outputs_v4 \
        --stage 7

    echo "  $VID tier comparison complete"
done

# ═══════════════════════════════════════════════════════════════════════════════
# Analysis: Compare Tiers
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "######################################################################"
echo "# TIER COMPARISON RESULTS"
echo "######################################################################"

python -c "
import pandas as pd, json, os, numpy as np
from scipy.stats import spearmanr, kendalltau

videos = []
for v in ['A0','A5','A1','A3','B0','B1','B3','B5','C0','C1','C3','C5','D0','D1','D3','D5']:
    tc_dir = f'outputs_v4/{v}/tier_comparison'
    if os.path.exists(f'{tc_dir}/importance_tier1_gemini.csv') and \
       os.path.exists(f'{tc_dir}/importance_tier3_heuristic.csv'):
        videos.append(v)

if not videos:
    print('No tier comparison data found.')
    exit()

print(f'Videos with tier comparison: {len(videos)}')
print()

# ─── Table 1: Score Comparison ───
print('═══ TABLE 1: Overall Score by Tier ═══')
print(f'{\"Video\":<8} {\"T1 Score\":>10} {\"T1 Grade\":<12} {\"T3 Score\":>10} {\"T3 Grade\":<12} {\"Delta\":>8}')
print('─' * 62)

all_t1_scores = []
all_t3_scores = []

for vid in videos:
    tc = f'outputs_v4/{vid}/tier_comparison'
    try:
        r1 = json.load(open(f'{tc}/results_tier1_gemini.json'))
        r3 = json.load(open(f'{tc}/results_tier3_heuristic.json'))
        s1 = r1['overall_score']
        s3 = r3['overall_score']
        g1 = r1['overall_grade']
        g3 = r3['overall_grade']
        all_t1_scores.append(s1)
        all_t3_scores.append(s3)
        print(f'{vid:<8} {s1:>10.2f} {g1:<12} {s3:>10.2f} {g3:<12} {s3-s1:>+8.2f}')
    except Exception as e:
        print(f'{vid:<8} Error: {e}')

if all_t1_scores and all_t3_scores:
    print('─' * 62)
    print(f'{\"Mean\":<8} {np.mean(all_t1_scores):>10.2f} {\"\":12} {np.mean(all_t3_scores):>10.2f} {\"\":12} {np.mean(all_t3_scores)-np.mean(all_t1_scores):>+8.2f}')
    print(f'{\"Std\":<8} {np.std(all_t1_scores):>10.2f} {\"\":12} {np.std(all_t3_scores):>10.2f}')

# ─── Table 2: Importance Distribution ───
print()
print('═══ TABLE 2: Importance Rating Distribution ═══')
print(f'{\"Video\":<8} {\"Tier\":>6}  {\"1\":>5} {\"2\":>5} {\"3\":>5} {\"4\":>5} {\"5\":>5}  {\"Mean\":>6} {\"Std\":>5}')
print('─' * 62)

for vid in videos:
    tc = f'outputs_v4/{vid}/tier_comparison'
    try:
        t1 = pd.read_csv(f'{tc}/importance_tier1_gemini.csv')
        t3 = pd.read_csv(f'{tc}/importance_tier3_heuristic.csv')
        for label, df in [('T1', t1), ('T3', t3)]:
            dist = df['importance'].value_counts().reindex([1,2,3,4,5], fill_value=0)
            mean_imp = df['importance'].mean()
            std_imp = df['importance'].std()
            print(f'{vid:<8} {label:>6}  {dist[1]:>5} {dist[2]:>5} {dist[3]:>5} {dist[4]:>5} {dist[5]:>5}  {mean_imp:>6.2f} {std_imp:>5.2f}')
    except Exception as e:
        print(f'{vid:<8} Error: {e}')

# ─── Table 3: Inter-Tier Agreement ───
print()
print('═══ TABLE 3: Inter-Tier Agreement (T1 vs T3) ═══')
print(f'{\"Video\":<8} {\"Spearman\":>10} {\"Kendall\":>10} {\"Exact Match\":>12} {\"±1 Match\":>10} {\"N\":>5}')
print('─' * 58)

all_rho = []
all_exact = []
all_near = []

for vid in videos:
    tc = f'outputs_v4/{vid}/tier_comparison'
    try:
        t1 = pd.read_csv(f'{tc}/importance_tier1_gemini.csv')
        t3 = pd.read_csv(f'{tc}/importance_tier3_heuristic.csv')

        # Merge on segment_id
        merged = pd.merge(t1[['segment_id','importance']], t3[['segment_id','importance']],
                          on='segment_id', suffixes=('_t1','_t3'))

        if len(merged) < 3:
            print(f'{vid:<8} Too few segments ({len(merged)})')
            continue

        rho, p_rho = spearmanr(merged['importance_t1'], merged['importance_t3'])
        tau, p_tau = kendalltau(merged['importance_t1'], merged['importance_t3'])
        exact = (merged['importance_t1'] == merged['importance_t3']).mean() * 100
        near = (abs(merged['importance_t1'] - merged['importance_t3']) <= 1).mean() * 100

        all_rho.append(rho)
        all_exact.append(exact)
        all_near.append(near)

        print(f'{vid:<8} {rho:>10.3f} {tau:>10.3f} {exact:>11.1f}% {near:>9.1f}% {len(merged):>5}')
    except Exception as e:
        print(f'{vid:<8} Error: {e}')

if all_rho:
    print('─' * 58)
    print(f'{\"Mean\":<8} {np.mean(all_rho):>10.3f} {\"\":>10} {np.mean(all_exact):>11.1f}% {np.mean(all_near):>9.1f}%')

# ─── Table 4: Grade Agreement ───
print()
print('═══ TABLE 4: Grade Agreement ═══')
grade_match = 0
grade_total = 0
for vid in videos:
    tc = f'outputs_v4/{vid}/tier_comparison'
    try:
        r1 = json.load(open(f'{tc}/results_tier1_gemini.json'))
        r3 = json.load(open(f'{tc}/results_tier3_heuristic.json'))
        g1, g3 = r1['overall_grade'], r3['overall_grade']
        match = '✓' if g1 == g3 else '✗'
        if g1 == g3:
            grade_match += 1
        grade_total += 1
        print(f'  {vid}: T1={g1}, T3={g3} {match}')
    except:
        pass
if grade_total:
    print(f'  Grade agreement: {grade_match}/{grade_total} ({100*grade_match/grade_total:.0f}%)')

# ─── Summary ───
print()
print('═══ SUMMARY ═══')
if all_rho:
    mean_rho = np.mean(all_rho)
    mean_exact = np.mean(all_exact)
    mean_near = np.mean(all_near)
    print(f'  Mean Spearman ρ (T1 vs T3): {mean_rho:.3f}')
    print(f'  Mean exact match:           {mean_exact:.1f}%')
    print(f'  Mean ±1 match:              {mean_near:.1f}%')
    if mean_rho > 0.7:
        print(f'  → Strong agreement: heuristic is a reliable fallback')
    elif mean_rho > 0.4:
        print(f'  → Moderate agreement: heuristic is a reasonable fallback')
    else:
        print(f'  → Weak agreement: heuristic needs refinement')
if all_t1_scores and all_t3_scores:
    score_diff = abs(np.mean(all_t1_scores) - np.mean(all_t3_scores))
    print(f'  Mean score difference:      {score_diff:.2f} points')
    if score_diff < 3:
        print(f'  → Tier choice has minimal impact on final scores (<3 pts)')
    elif score_diff < 5:
        print(f'  → Tier choice has moderate impact on final scores (3-5 pts)')
    else:
        print(f'  → Tier choice has significant impact on final scores (>5 pts)')
" 2>&1

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Tier Comparison Complete"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════════════"
