#!/usr/bin/env bash
# =============================================================
# setup.sh — One-command environment setup
#
# HOW TO USE (for beginners):
#   1. Open a terminal in the rrin_project folder
#   2. Make this script executable: chmod +x setup.sh
#   3. Run it: ./setup.sh
#
# WHAT THIS SCRIPT DOES:
#   - Creates a Python virtual environment called .venv
#   - Installs all required packages from requirements.txt
#   - Copies .env.example to .env if you haven't done it yet
#   - Creates the output directories
#   - Prints next steps
# =============================================================

set -e  # Stop if any command fails

echo ""
echo "=============================================="
echo " RRIN — Environment Setup"
echo "=============================================="
echo ""

# ---- Check Python version ---------------------------------
PYTHON_VERSION=$(python3 --version 2>&1 | grep -oP '\d+\.\d+' | head -1)
REQUIRED_MAJOR=3
REQUIRED_MINOR=10

python3 -c "
import sys
major, minor = sys.version_info[:2]
if major < 3 or (major == 3 and minor < 10):
    print(f'ERROR: Python 3.10+ is required. You have {major}.{minor}.')
    sys.exit(1)
print(f'Python {major}.{minor} — OK')
"

# ---- Create virtual environment ---------------------------
echo ""
echo "[1/5] Creating virtual environment (.venv)..."
if [ -d ".venv" ]; then
    echo "  .venv already exists — skipping creation."
else
    python3 -m venv .venv
    echo "  Created .venv/"
fi

# ---- Activate virtual environment -------------------------
echo ""
echo "[2/5] Activating virtual environment..."
source .venv/bin/activate
echo "  Activated."
echo "  (To activate manually in the future: source .venv/bin/activate)"

# ---- Install packages -------------------------------------
echo ""
echo "[3/5] Installing packages from requirements.txt..."
echo "  (This takes 3-10 minutes on first run)"
pip install --upgrade pip --quiet
pip install -r requirements.txt
echo "  All packages installed."

# ---- Check for GPU ----------------------------------------
echo ""
echo "[4/5] Checking for GPU..."
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'  GPU detected: {torch.cuda.get_device_name(0)}')
    print(f'  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB')
else:
    print('  No GPU detected. Training will run on CPU (very slow).')
    print('  For free GPU: use Kaggle (kaggle.com) or Google Colab (colab.research.google.com)')
"

# ---- Create .env file -------------------------------------
echo ""
echo "[5/5] Setting up .env file..."
if [ -f ".env" ]; then
    echo "  .env already exists — not overwriting."
else
    cp .env.example .env
    echo "  Copied .env.example → .env"
    echo ""
    echo "  ACTION REQUIRED: Open .env in a text editor and fill in:"
    echo "    KAGGLE_USERNAME = your_kaggle_username"
    echo "    KAGGLE_KEY      = your_kaggle_api_key"
    echo "  Get these from: https://www.kaggle.com → Settings → API → Create New Token"
fi

# ---- Create output directories ----------------------------
mkdir -p metadata checkpoints logs data output
echo "  Created output directories: metadata/ checkpoints/ logs/ data/ output/"

# ---- Done -------------------------------------------------
echo ""
echo "=============================================="
echo " Setup complete!"
echo "=============================================="
echo ""
echo " NEXT STEPS:"
echo ""
echo " 1. Fill in your Kaggle credentials in .env"
echo "    (See: https://www.kaggle.com → Settings → API → Create New Token)"
echo ""
echo " 2. Download the datasets:"
echo "    python scripts/kaggle_setup.py"
echo ""
echo " 3. Update config.yaml — set the correct paths under 'dataset_paths:'"
echo "    (Open config.yaml in any text editor)"
echo ""
echo " 4. Start training:"
echo "    source .venv/bin/activate   ← activate environment first"
echo "    python main.py              ← start training"
echo ""
echo " OR — use the free Kaggle notebook instead (no local storage needed):"
echo "    See README.md → Option A: Train on Kaggle (Recommended)"
echo ""
