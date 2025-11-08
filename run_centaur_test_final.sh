#!/bin/bash
# CENTaUR Quick Test on DGX-Spark (CUDA 13.0 compatible)

set -e

echo "=== CENTaUR Quick Test Setup ==="
cd ~/git/CENTaUR

# Create Python virtual environment if it doesn't exist
if [ ! -d "dgx-venv" ]; then
    echo "Creating Python virtual environment..."
    python3 -m venv dgx-venv
fi

# Activate environment
source dgx-venv/bin/activate

# Install dependencies
echo "Installing dependencies..."
pip install --upgrade pip

# Install PyTorch for CUDA 12.1+ (compatible with CUDA 13.0)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other dependencies
pip install transformers accelerate bitsandbytes peft tqdm

# Check GPU availability
echo "=== GPU Check ==="
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Test feature extraction with small sample
echo "=== Running Feature Extraction Test (10 samples) ==="
python scripts/extract_centaur_features.py \
    --model qwen25 \
    --n_samples 10

echo "=== Test Complete ==="
echo "Check the output files:"
ls -lh features_*.pt 2>/dev/null || echo "No feature files yet"
