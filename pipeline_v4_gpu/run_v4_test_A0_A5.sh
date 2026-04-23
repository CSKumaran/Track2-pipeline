#!/bin/bash
#SBATCH --job-name=v4_test
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=v4_test_A0_A5_%j.log
#SBATCH --error=v4_test_A0_A5_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

# ═══════════════════════════════════════════════════════════════════════════════
# V4 Validation: A0 (baseline) vs A5 (5s delay)
#
# This is the FIRST V4 run. Tests:
#   1. Does V4 code run without errors?
#   2. Does coverage improve over V3? (Target: 34% → 50%+)
#   3. Does score spread improve? (Target: 3.38 → 5+)
#   4. Is monotonicity preserved? (A0 > A5)
#   5. Does visual_on_screen flag populate correctly?
#   6. Does Track E produce matches?
#   7. Does sensitivity analysis complete?
#
# Strategy: Reuse V3 Stage 1 (ASR) — transcripts are identical.
#           Run Stages 2→5→4→6→7 fresh with V4 code.
# ═══════════════════════════════════════════════════════════════════════════════

# === Environment Setup ===
export PATH="/usr/local/cuda-12.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"

CONDA_ENV=/iitjhome/senthil1/.conda/envs/tc_pipeline
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH=/iitjhome/senthil1
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0

cd /iitjhome/senthil1

# Load Gemini API key
if [ -f ~/.gemini_env ]; then
    source ~/.gemini_env
fi

# Clear stale cache
find pipeline_v4_gpu/ -name "__pycache__" -exec rm -rf {} + 2>/dev/null

echo "═══════════════════════════════════════════════════════════════"
echo "  V4 Validation: A0 vs A5"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════════════"

echo ""
echo "=== GPU Check ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"

VIDEOS=("A0" "A5")
# Pipeline order: 1 → 2 → 5 → 4 → 6 → 7
STAGES=(2 5 4 6 7)

for VID in "${VIDEOS[@]}"; do
    echo ""
    echo "######################################################################"
    echo "# Processing: $VID (V4 — full pipeline)"
    echo "######################################################################"
    echo "Start: $(date)"

    # Copy V3 ASR output to V4 output dir (reuse — transcripts are identical)
    mkdir -p "outputs_v4/$VID"
    if [ -f "outputs_v3/$VID/transcript_segments_improved.csv" ]; then
        echo "=== $VID — Copying V3 ASR to V4 output ==="
        cp -n "outputs_v3/$VID/transcript_segments_improved.csv" "outputs_v4/$VID/"
        cp -n "outputs_v3/$VID/transcript_words.csv" "outputs_v4/$VID/" 2>/dev/null
        cp -n "outputs_v3/$VID/transcript_raw.json" "outputs_v4/$VID/" 2>/dev/null
    else
        echo "=== $VID — Stage 1 (ASR) ==="
        python -m pipeline_v4_gpu.main \
            --video "videos/$VID.mp4" \
            --output-root outputs_v4 \
            --stage 1
    fi

    # Clear V4-specific caches (force fresh V4 run)
    rm -f "outputs_v4/$VID/scenes.csv"
    rm -f "outputs_v4/$VID/ocr_per_frame.csv"
    rm -f "outputs_v4/$VID/dinov2_distances.csv"
    rm -f "outputs_v4/$VID/keyword_alignment.csv"
    rm -f "outputs_v4/$VID/segment_alignment.csv"
    rm -f "outputs_v4/$VID/keywords.csv"
    rm -f "outputs_v4/$VID/keyword_scores.csv"
    rm -f "outputs_v4/$VID/segment_scores.csv"
    rm -f "outputs_v4/$VID/results.json"
    rm -f "outputs_v4/$VID/pedagogical_importance.csv"
    rm -f "outputs_v4/$VID/report_dashboard.html"
    rm -rf "outputs_v4/$VID/diagnostics"
    rm -rf "outputs_v4/$VID/sensitivity"

    for STG in "${STAGES[@]}"; do
        echo ""
        echo "=== $VID — Stage $STG ==="
        echo "Time: $(date)"

        GEMINI_ARGS=""
        if [ -n "$GEMINI_API_KEY" ]; then
            GEMINI_ARGS="--gemini-api-key $GEMINI_API_KEY"
        fi

        python -m pipeline_v4_gpu.main \
            --video "videos/$VID.mp4" \
            --output-root outputs_v4 \
            --stage $STG \
            $GEMINI_ARGS \
            --sensitivity-analysis

        if [ $? -ne 0 ]; then
            echo "ERROR: Stage $STG failed for $VID"
            break
        fi
    done

    echo ""
    echo "=== $VID — Dashboard ==="
    ls -la "outputs_v4/$VID/report_dashboard.html" 2>/dev/null && echo "YES" || echo "NO"
    echo "Completed $VID at $(date)"
done

# ═══════════════════════════════════════════════════════════════════════════════
# Results Comparison: V4 A0 vs A5
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "######################################################################"
echo "# V4 RESULTS: A0 vs A5"
echo "######################################################################"

python -c "
import pandas as pd, json, os

videos = ['A0', 'A5']

print('=== V4 SCORE COMPARISON ===')
print(f'{\"Video\":<8} {\"Score\":<10} {\"Grade\":<12} {\"Matched\":<10} {\"Coverage\":<10} {\"Opt%\":<8} {\"TrackE\":<8}')
print('-' * 70)

scores_v4 = {}
for vid in videos:
    try:
        r = json.load(open(f'outputs_v4/{vid}/results.json'))
        s = r['overall_score']
        scores_v4[vid] = s
        kl = r['keyword_level']
        cov = r.get('coverage', {})

        # Count Track E matches
        kw = pd.read_csv(f'outputs_v4/{vid}/keyword_alignment.csv')
        n_e = len(kw[kw['match_case'] == 'E']) if 'match_case' in kw.columns else 0

        print(f'{vid:<8} {s:<10} {r[\"overall_grade\"]:<12} {kl[\"n_matched\"]}/{kl[\"n_total\"]:<7} {cov.get(\"coverage_rate\",0)*100:<9.1f}% {kl[\"pct_Optimal\"]:<7.1f}% {n_e:<8}')
    except Exception as e:
        print(f'{vid:<8} Error: {e}')

# V3 comparison (if available)
print()
print('=== V3 vs V4 COMPARISON ===')
print(f'{\"Video\":<8} {\"V3 Score\":<12} {\"V4 Score\":<12} {\"Delta\":<10} {\"V3 Cov\":<10} {\"V4 Cov\":<10}')
print('-' * 60)

for vid in videos:
    try:
        r3 = json.load(open(f'outputs_v3/{vid}/results.json'))
        r4 = json.load(open(f'outputs_v4/{vid}/results.json'))
        s3 = r3['overall_score']
        s4 = r4['overall_score']
        c3 = r3.get('coverage', {}).get('coverage_rate', 0) * 100
        c4 = r4.get('coverage', {}).get('coverage_rate', 0) * 100
        print(f'{vid:<8} {s3:<12} {s4:<12} {s4-s3:<+10.2f} {c3:<9.1f}% {c4:<9.1f}%')
    except Exception as e:
        print(f'{vid:<8} Error: {e}')

# Spread comparison
print()
if 'A0' in scores_v4 and 'A5' in scores_v4:
    spread_v4 = scores_v4['A0'] - scores_v4['A5']
    print(f'V4 Score spread (A0-A5): {spread_v4:.2f}')
    try:
        r3_a0 = json.load(open('outputs_v3/A0/results.json'))
        r3_a5 = json.load(open('outputs_v3/A5/results.json'))
        spread_v3 = r3_a0['overall_score'] - r3_a5['overall_score']
        print(f'V3 Score spread (A0-A5): {spread_v3:.2f}')
        print(f'Spread improvement:      {spread_v4 - spread_v3:+.2f} points')
    except:
        pass
" 2>&1

# ═══════════════════════════════════════════════════════════════════════════════
# V4-specific checks
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "######################################################################"
echo "# V4 FEATURE VERIFICATION"
echo "######################################################################"

python -c "
import pandas as pd, json, os

for vid in ['A0', 'A5']:
    print(f'')
    print(f'=== {vid} — V4 Feature Check ===')

    # 1. Track E matches
    kw = pd.read_csv(f'outputs_v4/{vid}/keyword_alignment.csv')
    case_dist = kw['match_case'].value_counts().to_dict()
    print(f'  Case distribution: {case_dist}')

    # 2. visual_on_screen flag
    if 'visual_on_screen' in kw.columns:
        n_on = kw['visual_on_screen'].sum()
        n_total = len(kw[kw['match_case'] != 'F'])
        print(f'  visual_on_screen: {n_on}/{n_total} matched keywords ({100*n_on/max(n_total,1):.0f}%)')
    else:
        print(f'  visual_on_screen: COLUMN MISSING — check alignment.py')

    # 3. Gemini concepts
    scenes = pd.read_csv(f'outputs_v4/{vid}/scenes.csv')
    if 'gemini_concepts' in scenes.columns:
        n_concepts = scenes['gemini_concepts'].notna().sum()
        print(f'  Gemini concepts: {n_concepts}/{len(scenes)} scenes have concepts')
    else:
        print(f'  Gemini concepts: COLUMN MISSING — check scene_detection.py')

    # 4. First-mention flags
    kw_ext = pd.read_csv(f'outputs_v4/{vid}/keywords.csv')
    if 'is_first_mention' in kw_ext.columns:
        n_fm = kw_ext['is_first_mention'].sum()
        print(f'  First mentions: {n_fm}/{len(kw_ext)} keywords ({100*n_fm/max(len(kw_ext),1):.0f}%)')
    else:
        print(f'  First mentions: COLUMN MISSING — check keyword_extraction.py')

    # 5. Importance backend
    imp = pd.read_csv(f'outputs_v4/{vid}/pedagogical_importance.csv')
    if 'backend' in imp.columns:
        backend = imp['backend'].iloc[0]
        print(f'  Importance backend: {backend}')
    else:
        print(f'  Importance backend: COLUMN MISSING')

    # 6. Sensitivity analysis
    sens_dir = f'outputs_v4/{vid}/sensitivity'
    if os.path.exists(sens_dir):
        files = os.listdir(sens_dir)
        print(f'  Sensitivity analysis: {len(files)} files in {sens_dir}')
        if os.path.exists(f'{sens_dir}/sensitivity_summary.json'):
            s = json.load(open(f'{sens_dir}/sensitivity_summary.json'))
            ps = s.get('perturbation_stability', {})
            print(f'    Mean rank corr: {ps.get(\"mean_rank_corr\", \"?\")}')
            print(f'    Mean pct same:  {ps.get(\"mean_pct_same\", \"?\")}%')
    else:
        print(f'  Sensitivity analysis: NOT RUN')
" 2>&1

# Per-keyword delta_t comparison
echo ""
echo "######################################################################"
echo "# DELTA_T COMPARISON (A0 vs A5)"
echo "######################################################################"

python -c "
import pandas as pd

a0_kw = pd.read_csv('outputs_v4/A0/keyword_scores.csv')
a5_kw = pd.read_csv('outputs_v4/A5/keyword_scores.csv')

a0_m = a0_kw[a0_kw['match_case'] != 'F'][['keyword_text', 'delta_t', 'S_temporal', 'match_case', 'visual_on_screen']].copy()
a5_m = a5_kw[a5_kw['match_case'] != 'F'][['keyword_text', 'delta_t', 'S_temporal', 'match_case', 'visual_on_screen']].copy()

a0_m.columns = ['keyword', 'dt_A0', 'S_A0', 'case_A0', 'vos_A0']
a5_m.columns = ['keyword', 'dt_A5', 'S_A5', 'case_A5', 'vos_A5']

merged = pd.merge(a0_m, a5_m, on='keyword', how='outer', indicator=True)
both = merged[merged['_merge'] == 'both'].copy()
both['dt_diff'] = both['dt_A5'] - both['dt_A0']
both_sorted = both.sort_values('dt_diff', ascending=False)

print(f'A0 matched: {len(a0_m)}, A5 matched: {len(a5_m)}, In both: {len(both)}')
print()
print(f'{\"keyword\":<35} {\"dt_A0\":>7} {\"dt_A5\":>7} {\"shift\":>7} {\"S_A0\":>7} {\"S_A5\":>7} {\"vos_A0\":>6} {\"vos_A5\":>6}')
print('-' * 85)
for _, row in both_sorted.head(15).iterrows():
    kw = str(row['keyword'])[:33]
    print(f'{kw:<35} {row[\"dt_A0\"]:>7.2f} {row[\"dt_A5\"]:>7.2f} {row[\"dt_diff\"]:>+7.2f} {row[\"S_A0\"]:>7.1f} {row[\"S_A5\"]:>7.1f} {str(row[\"vos_A0\"]):>6} {str(row[\"vos_A5\"]):>6}')
" 2>&1

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  V4 Test Complete"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════════════"
