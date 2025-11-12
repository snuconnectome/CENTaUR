#!/bin/bash
# Train Qwen2.5-32B QLoRA using NGC PyTorch Docker
set -e

SUDO_PASSWORD="462773"
PROJECT_DIR="/home/juke/git/CENTaUR"
WORK_DIR="/workspace"

echo "=========================================="
echo "Qwen2.5-32B QLoRA Training with NGC PyTorch"
echo "=========================================="
echo ""

# Check if NGC PyTorch image exists
echo "Checking NGC PyTorch image..."
if ! echo "$SUDO_PASSWORD" | sudo -S docker images | grep -q "nvcr.io/nvidia/pytorch.*24.08"; then
    echo "❌ NGC PyTorch 24.08 image not found"
    echo "Run: sudo docker pull nvcr.io/nvidia/pytorch:24.08-py3"
    exit 1
fi
echo "✅ NGC PyTorch image found"
echo ""

# Check GPU
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Create logs directory
mkdir -p logs

echo "Starting NGC Docker container for training..."
echo ""

# Run training in NGC Docker container
echo "$SUDO_PASSWORD" | sudo -S docker run --rm \
  --gpus=all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$PROJECT_DIR:$WORK_DIR" \
  -w "$WORK_DIR" \
  nvcr.io/nvidia/pytorch:24.08-py3 \
  bash -c "
    set -e

    echo '=========================================='
    echo 'NGC PyTorch Container Environment'
    echo '=========================================='
    python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA: {torch.cuda.is_available()}\"); print(f\"CUDA version: {torch.version.cuda}\")'
    echo ''

    echo 'Installing required packages...'
    pip install -q transformers accelerate peft bitsandbytes datasets pyyaml
    echo '✅ Packages installed'
    echo ''

    echo 'Starting training...'
    python ko_centaur/training/train_qwen25_32b_qlora.py \
      --config ko_centaur/configs/training_qwen25_32b_qlora.yaml
  " 2>&1 | tee logs/qwen25-32b-qlora_ngc.log

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully"
else
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo "Check logs/qwen25-32b-qlora_ngc.log for details"
fi
echo "=========================================="

exit $EXIT_CODE
