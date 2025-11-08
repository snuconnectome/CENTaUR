#!/bin/bash
# CENTaUR on ARM64 DGX-Spark

set -e

echo "=== CENTaUR ARM64 Setup ==="
cd ~/git/CENTaUR

# Create Python virtual environment
if [ ! -d "dgx-venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv dgx-venv
fi

source dgx-venv/bin/activate

echo "Installing dependencies..."
pip install --upgrade pip

# Install PyTorch from default PyPI (ARM64 compatible)
echo "Installing PyTorch for ARM64..."
pip install torch torchvision torchaudio

# Install other dependencies
pip install transformers accelerate peft tqdm

# Note: bitsandbytes may not work on ARM64, try without it first
echo "Attempting to install bitsandbytes..."
pip install bitsandbytes || echo "WARNING: bitsandbytes not available on ARM64"

# Check GPU and PyTorch
echo "=== System Check ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
python3 << 'PYEOF'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
PYEOF

# Run feature extraction test
echo "=== Running Feature Extraction Test (10 samples) ==="
python scripts/extract_centaur_features.py \
    --model qwen25 \
    --n_samples 10

echo "=== Test Complete ==="
ls -lh features_*.pt 2>/dev/null || echo "No feature files yet"
