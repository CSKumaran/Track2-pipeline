#!/bin/bash
#SBATCH --job-name=v4_rescore
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=v4_rescore_%j.log
#SBATCH --error=v4_rescore_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

# ═══════════════════════════════════════════════════════════════════════════════
# V4.1: Re-score ALL 24 videos with Dwell Time Decay
#
# Only Stage 7 (scoring) — alignment data unchanged.
# New params: --dwell-threshold 15 --dwell-tau 20
#
# Validates:
# 1. VE/CTML condition rankings (01>02, 01>03)
# 2. A-series monotonicity (A0>A1>A3>A5)
# 3. B-series monotonicity (B0>B1>B3>B5)
# ═══════════════════════════════════════════════════════════════════════════════

export PATH="/usr/local/cuda-12.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"

CONDA_ENV=/iitjhome/senthil1/.conda/envs/tc_pipeline
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH=/iitjhome/senthil1
export TORCH_FORCE_WEIGHTS_ONLY_LOAD=0

cd /iitjhome/senthil1

find pipeline_v4_gpu/ -name "__pycache__" -exec rm -rf {} + 2>/dev/null

echo "═══════════════════════════════════════════════════════════════"
echo "  V4.1: Re-score ALL videos with Dwell Time Decay"
echo "  threshold=15s, tau=20s"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════════════"

ALL_VIDEOS=(
    "A0" "A1" "A3" "A5"
    "B0" "B1" "B3" "B5"
    "C0" "C1" "C3" "C5"
    "D0" "D1" "D3" "D5"
    "Video_evaluation_01" "Video_evaluation_02" "Video_evaluation_03" "Video_evaluation_04"
    "CTML_03_01" "CTML_03_02" "CTML_03_03" "CTML_03_04"
)

for VID in "${ALL_VIDEOS[@]}"; do
    if [ ! -f "outputs_v4/$VID/keyword_alignment.csv" ]; then
        echo "SKIP $VID — no alignment data"
        continue
    fi

    # Clear old scoring output
    rm -f "outputs_v4/$VID/keyword_scores.csv"
    rm -f "outputs_v4/$VID/segment_scores.csv"
    rm -f "outputs_v4/$VID/results.json"

    echo "Rescoring $VID..."
    python -m pipeline_v4_gpu.main \
        --video "videos/$VID.mp4" \
        --output-root outputs_v4 \
        --stage 7 \
        --dwell-threshold 15.0 \
        --dwell-tau 30.0

    if [ $? -ne 0 ]; then
        echo "ERROR: Stage 7 failed for $VID"
    fi
done

# ═══════════════════════════════════════════════════════════════════════════════
# Results Comparison
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "######################################################################"
echo "# V4.1 DWELL DECAY RESULTS"
echo "######################################################################"

python -c "
import json, os
import pandas as pd
import numpy as np

# === A Series Monotonic Test ===
print('=== A SERIES (Monotonicity) ===')
a_scores = []
for vid in ['A0','A1','A3','A5']:
    try:
        r = json.load(open(f'outputs_v4/{vid}/results.json'))
        s = r['overall_score']
        a_scores.append(s)
        print(f'  {vid}: {s:.2f} ({r[\"overall_grade\"]})')
    except Exception as e:
        print(f'  {vid}: Error: {e}')
if len(a_scores) == 4:
    mono = all(a_scores[i] >= a_scores[i+1] for i in range(3))
    print(f'  Monotonic: {\"YES\" if mono else \"NO\"}')
    print(f'  Spread: {a_scores[0] - a_scores[-1]:.2f}')

# === B Series ===
print()
print('=== B SERIES ===')
b_scores = []
for vid in ['B0','B1','B3','B5']:
    try:
        r = json.load(open(f'outputs_v4/{vid}/results.json'))
        s = r['overall_score']
        b_scores.append(s)
        print(f'  {vid}: {s:.2f}')
    except: pass
if len(b_scores) == 4:
    mono = all(b_scores[i] >= b_scores[i+1] for i in range(3))
    print(f'  Monotonic: {\"YES\" if mono else \"NO\"}')

# === C/D Series ===
for series in ['C', 'D']:
    print(f'')
    print(f'=== {series} SERIES ===')
    for vid in [f'{series}0',f'{series}1',f'{series}3',f'{series}5']:
        try:
            r = json.load(open(f'outputs_v4/{vid}/results.json'))
            print(f'  {vid}: {r[\"overall_score\"]:.2f}')
        except: pass

# === Video_evaluation Conditions ===
conditions = {'01':'Simultaneous','02':'Animation first','03':'Narration first','04':'Segmented'}
for prefix, label in [('Video_evaluation', 'VIDEO EVALUATION'), ('CTML_03', 'CTML_03')]:
    print(f'')
    print(f'=== {label} CONDITIONS ===')
    scores = {}
    for cond, name in conditions.items():
        vid = f'{prefix}_{cond}'
        try:
            r = json.load(open(f'outputs_v4/{vid}/results.json'))
            scores[cond] = r['overall_score']

            # Delta_t stats from keyword_scores
            kw = pd.read_csv(f'outputs_v4/{vid}/keyword_scores.csv')
            matched = kw[kw['delta_t'].notna()]
            vis = kw.get('visual_on_screen', pd.Series(False, index=kw.index))
            n_vis = vis.sum() if hasattr(vis, 'sum') else 0
            dwell_kw = matched[(matched['delta_t'] <= 0) & (vis.reindex(matched.index, fill_value=False))]
            mean_dwell = abs(dwell_kw['delta_t']).mean() if len(dwell_kw) > 0 else 0

            zones = kw['zone'].value_counts().to_dict() if 'zone' in kw.columns else {}
            n_opt = zones.get('Optimal', 0)
            n_sub = zones.get('Suboptimal', 0)
            n_dis = zones.get('Disruptive', 0)
            n_una = zones.get('Unacceptable', 0)

            print(f'  {vid:<24} {name:<18} Score={r[\"overall_score\"]:>6.2f}  Opt={n_opt} Sub={n_sub} Dis={n_dis} Una={n_una}  MeanDwell={mean_dwell:.1f}s')
        except Exception as e:
            print(f'  {vid:<24} Error: {e}')

    if len(scores) >= 2:
        ranking = sorted(scores.items(), key=lambda x: -x[1])
        print(f'  Ranking: {\" > \".join(f\"{c}({s:.1f})\" for c, s in ranking)}')
        print(f'  Spread: {max(scores.values()) - min(scores.values()):.1f} points')
        if scores.get('01', 0) > scores.get('02', 0):
            print(f'  01>02 (Simul>Anim): YES')
        else:
            print(f'  01>02 (Simul>Anim): NO ← PROBLEM')

# === Full Summary Table ===
print()
print('═══ FULL SUMMARY ═══')
print(f'{\"Video\":<24} {\"V4.1 Score\":>10} {\"Grade\":<12} {\"Version\":>8}')
print('-' * 58)
all_vids = ['A0','A1','A3','A5','B0','B1','B3','B5','C0','C1','C3','C5','D0','D1','D3','D5',
            'Video_evaluation_01','Video_evaluation_02','Video_evaluation_03','Video_evaluation_04',
            'CTML_03_01','CTML_03_02','CTML_03_03','CTML_03_04']
for vid in all_vids:
    try:
        r = json.load(open(f'outputs_v4/{vid}/results.json'))
        ver = r.get('pipeline_version', '4.0')
        print(f'{vid:<24} {r[\"overall_score\"]:>10.2f} {r[\"overall_grade\"]:<12} {ver:>8}')
    except:
        pass
" 2>&1

echo ""
echo "=== DONE ==="
echo "End: $(date)"
