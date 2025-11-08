#!/bin/bash
# CENTaUR on ARM64 DGX-Spark - Base Model Only

set -e

echo "=== CENTaUR Base Model Setup ==="
cd ~/git/CENTaUR

# Activate existing environment
source dgx-venv/bin/activate

# Check environment
echo "=== Environment Check ==="
python3 << 'PYEOF'
import torch
print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
PYEOF

# GPU check
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv

# Run feature extraction with BASE MODEL (no adapter)
echo "=== Running Feature Extraction (Base Model, 10 samples) ==="
python scripts/extract_centaur_features.py \
    --model qwen25-base \
    --n_samples 10

echo "=== Test Complete ==="
ls -lh features_*.pt 2>/dev/null || echo "No feature files yet"
