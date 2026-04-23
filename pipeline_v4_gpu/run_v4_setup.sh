#!/bin/bash
#SBATCH --job-name=v4_setup
#SBATCH --partition=fat
#SBATCH --gres=gpu:1
#SBATCH --time=00:15:00
#SBATCH --output=v4_setup_%j.log
#SBATCH --error=v4_setup_%j.err
#SBATCH --cpus-per-task=4
#SBATCH --mem-per-cpu=8G

# ═══════════════════════════════════════════════════════════════════════════════
# V4 Setup & Dependency Check
# Run this FIRST before any V4 pipeline jobs.
# Installs new V4 dependencies and verifies the environment.
# ═══════════════════════════════════════════════════════════════════════════════

# === Environment Setup ===
export PATH="/usr/local/cuda-12.2/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-12.2/lib64:$LD_LIBRARY_PATH"

CONDA_ENV=/iitjhome/senthil1/.conda/envs/tc_pipeline
export PATH="$CONDA_ENV/bin:$PATH"
export PYTHONPATH=/iitjhome/senthil1

cd /iitjhome/senthil1

echo "═══════════════════════════════════════════════════════════════"
echo "  V4 Pipeline Setup & Dependency Check"
echo "  $(date)"
echo "═══════════════════════════════════════════════════════════════"

# === 1. GPU Check ===
echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# === 2. Install V4 dependencies ===
echo ""
echo "=== Installing V4 Dependencies ==="

# rapidfuzz: required for Track E (Gemini concept fuzzy matching)
pip install rapidfuzz --quiet 2>&1 | tail -1
python -c "import rapidfuzz; print(f'rapidfuzz: {rapidfuzz.__version__}')" 2>&1

# scipy: required for sensitivity analysis (spearmanr)
pip install scipy --quiet 2>&1 | tail -1
python -c "import scipy; print(f'scipy: {scipy.__version__}')" 2>&1

# spacy model: required for instructional verb detection
python -c "import spacy; spacy.load('en_core_web_sm')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "Installing spaCy en_core_web_sm..."
    python -m spacy download en_core_web_sm --quiet
fi
python -c "import spacy; nlp=spacy.load('en_core_web_sm'); print(f'spaCy model: en_core_web_sm OK')" 2>&1

# === 3. Verify ALL V4 dependencies ===
echo ""
echo "=== Dependency Verification ==="
python -c "
deps = {
    'torch': 'torch',
    'whisperx': 'whisperx',
    'pandas': 'pandas',
    'numpy': 'numpy',
    'cv2': 'cv2',
    'surya': 'surya',
    'open_clip': 'open_clip',
    'sentence_transformers': 'sentence_transformers',
    'sklearn': 'sklearn',
    'keybert': 'keybert',
    'rapidfuzz': 'rapidfuzz',       # V4 NEW
    'scipy': 'scipy',               # V4 NEW
    'spacy': 'spacy',               # V4 NEW (for dep parsing)
    'google.generativeai': 'google.generativeai',
}

ok = 0
fail = 0
for name, module in deps.items():
    try:
        __import__(module)
        print(f'  ✓ {name}')
        ok += 1
    except ImportError:
        print(f'  ✗ {name} — MISSING')
        fail += 1

print(f'')
print(f'  {ok}/{ok+fail} dependencies OK')
if fail > 0:
    print(f'  WARNING: {fail} missing — pipeline may fail')
else:
    print(f'  All dependencies satisfied')
"

# === 4. Verify V4 code parses ===
echo ""
echo "=== V4 Code Parse Check ==="
python -c "
import ast, os, sys
errors = []
for root, dirs, files in os.walk('pipeline_v4_gpu'):
    for f in files:
        if f.endswith('.py'):
            path = os.path.join(root, f)
            try:
                ast.parse(open(path, encoding='utf-8').read())
            except SyntaxError as e:
                errors.append(f'{path}: {e}')

if errors:
    print(f'ERRORS in {len(errors)} files:')
    for e in errors:
        print(f'  {e}')
    sys.exit(1)
else:
    n = sum(1 for r,d,fs in os.walk('pipeline_v4_gpu') for f in fs if f.endswith('.py'))
    print(f'  All {n} Python files parse OK')
"

# === 5. Check Gemini API key ===
echo ""
echo "=== Gemini API Key ==="
if [ -f ~/.gemini_env ]; then
    source ~/.gemini_env
    if [ -n "$GEMINI_API_KEY" ]; then
        echo "  API key loaded (${#GEMINI_API_KEY} chars)"
        # Quick connectivity test
        python -c "
import google.generativeai as genai
import os
genai.configure(api_key=os.environ['GEMINI_API_KEY'])
model = genai.GenerativeModel('gemini-2.5-flash')
r = model.generate_content('Say OK')
print(f'  Gemini test: {r.text.strip()[:20]}')
" 2>&1
    else
        echo "  WARNING: GEMINI_API_KEY is empty"
    fi
else
    echo "  WARNING: ~/.gemini_env not found"
    echo "  V4 will fall back to heuristic for Stage 6 and skip Gemini frame understanding"
fi

# === 6. Check V3 outputs (ASR can be reused) ===
echo ""
echo "=== V3 ASR Outputs (reusable for V4) ==="
for VID in A0 A1 A3 A5 B0 B1 B3 B5 C0 C1 C3 C5 D0 D1 D3 D5; do
    if [ -f "outputs_v3/$VID/transcript_segments_improved.csv" ]; then
        echo "  $VID: ASR ✓"
    else
        echo "  $VID: ASR ✗ (will need Stage 1)"
    fi
done

# === 7. Create outputs_v4 directory ===
echo ""
echo "=== Output Directory ==="
mkdir -p outputs_v4
echo "  outputs_v4/ ready"

echo ""
echo "═══════════════════════════════════════════════════════════════"
echo "  Setup complete. Ready to run V4 pipeline."
echo "  $(date)"
echo "═══════════════════════════════════════════════════════════════"
