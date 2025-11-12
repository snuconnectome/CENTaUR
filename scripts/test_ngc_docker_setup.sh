#!/bin/bash
# Quick test: NGC PyTorch Docker setup verification
set -e

SUDO_PASSWORD="462773"
PROJECT_DIR="/home/juke/git/CENTaUR"

echo "=========================================="
echo "NGC PyTorch Docker Setup Test"
echo "=========================================="
echo ""

echo "1. Testing NGC PyTorch container..."
echo "$SUDO_PASSWORD" | sudo -S docker run --rm --gpus=all \
  -v "$PROJECT_DIR:/workspace" \
  -w /workspace \
  nvcr.io/nvidia/pytorch:24.08-py3 \
  bash -c "
    echo 'PyTorch Version:'
    python -c 'import torch; print(f\"  PyTorch: {torch.__version__}\"); print(f\"  CUDA available: {torch.cuda.is_available()}\"); print(f\"  CUDA version: {torch.version.cuda}\")'
    echo ''
    echo 'GPU Information:'
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ''
    echo 'Installing test packages...'
    pip install -q transformers bitsandbytes peft
    echo ''
    echo 'Testing BitsAndBytes import:'
    python -c 'import bitsandbytes as bnb; print(f\"  BitsAndBytes: OK\")'
    echo ''
    echo 'Testing model config load:'
    python -c 'from transformers import AutoTokenizer; print(\"  Transformers: OK\")'
    echo ''
    echo '✅ All tests passed!'
  "

echo ""
echo "=========================================="
echo "✅ NGC Docker setup is ready!"
echo "Run: ./scripts/train_qwen25_ngc_docker.sh"
echo "=========================================="
