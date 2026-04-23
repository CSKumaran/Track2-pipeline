#!/bin/bash
#SBATCH --job-name=v4_sens
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=v4_sensitivity_%j.log
#SBATCH --error=v4_sensitivity_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

# ═══════════════════════════════════════════════════════════════════════════════
# V4: Sensitivity Analysis on ALL completed videos
#
# Run AFTER all videos have been processed.
# Generates per-video sensitivity/ directories with:
#   - weight_perturbation.csv (±50% each feature)
#   - ablation.csv (leave-one-out + V3-equivalent)
#   - scoring_impact.csv (end-to-end score change)
#   - sensitivity_summary.json
# ═══════════════════════════════════════════════════════════════════════════════

# === Environment Setup ===
export PATH="/usr/local/cuda-12.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"

CONDA_ENV=/iitjhome/senthil1/.conda/envs/tc_pipeline
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH=/iitjhome/senthil1

cd /iitjhome/senthil1

echo "═══════════════════════════════════════════════════════════════"
echo "  V4: Sensitivity Analysis"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════════════"

VIDEOS=("A0" "A1" "A3" "A5" "B0" "B1" "B3" "B5" "C0" "C1" "C3" "C5" "D0" "D1" "D3" "D5")

for VID in "${VIDEOS[@]}"; do
    OUTDIR="outputs_v4/$VID"

    if [ ! -f "$OUTDIR/transcript_segments_improved.csv" ]; then
        echo "SKIP $VID — no outputs"
        continue
    fi

    echo ""
    echo "=== $VID ==="
    python -m pipeline_v4_gpu.utils.sensitivity_analysis \
        --output-dir "$OUTDIR"

    if [ $? -ne 0 ]; then
        echo "ERROR: Sensitivity analysis failed for $VID"
    fi
done

# ═══════════════════════════════════════════════════════════════════════════════
# Cross-video stability summary
# ═══════════════════════════════════════════════════════════════════════════════

echo ""
echo "######################################################################"
echo "# CROSS-VIDEO SENSITIVITY SUMMARY"
echo "######################################################################"

python -c "
import json, os

videos = ['A0','A1','A3','A5','B0','B1','B3','B5','C0','C1','C3','C5','D0','D1','D3','D5']

print(f'{\"Video\":<8} {\"Segments\":<10} {\"Mean RankCorr\":<15} {\"MeanSame%\":<12} {\"Score CV%\":<12}')
print('-' * 60)

all_corrs = []
all_cvs = []

for vid in videos:
    path = f'outputs_v4/{vid}/sensitivity/sensitivity_summary.json'
    if not os.path.exists(path):
        continue
    s = json.load(open(path))
    ps = s.get('perturbation_stability', {})
    ss = s.get('scoring_stability', {})

    corr = ps.get('mean_rank_corr', 0)
    same = ps.get('mean_pct_same', 0)
    cv = ss.get('cv_pct', 0)

    all_corrs.append(corr)
    all_cvs.append(cv)

    print(f'{vid:<8} {s[\"n_segments\"]:<10} {corr:<15.3f} {same:<12.1f} {cv:<12.1f}')

if all_corrs:
    print()
    print(f'Across all videos:')
    print(f'  Mean rank correlation: {sum(all_corrs)/len(all_corrs):.3f} (>0.9 = robust)')
    print(f'  Mean score CV:         {sum(all_cvs)/len(all_cvs):.1f}% (<5% = stable)')
" 2>&1

echo ""
echo "=== DONE ==="
echo "End: $(date)"
