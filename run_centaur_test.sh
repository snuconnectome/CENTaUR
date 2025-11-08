#!/bin/bash
# CENTaUR Quick Test on DGX-Spark
# Run this in the tmux session 'git'

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
pip install torch transformers accelerate bitsandbytes peft tqdm

# Test feature extraction with small sample
echo "=== Running Feature Extraction Test (10 samples) ==="
python scripts/extract_centaur_features.py \
    --model qwen25 \
    --n_samples 10 \
    --is_local

echo "=== Test Complete ==="
echo "Check the output in the current directory"
echo "To run full experiment, modify the script parameters"
