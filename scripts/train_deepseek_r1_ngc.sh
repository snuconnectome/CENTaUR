#!/bin/bash
# Train DeepSeek-R1-Distill-Qwen-32B QLoRA using NGC PyTorch
# Uses ngc-python wrapper

set -e

echo "=========================================="
echo "DeepSeek-R1-32B QLoRA Training (NGC PyTorch)"
echo "=========================================="
echo ""

# Check GPU
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader
echo ""

# Create logs directory
mkdir -p logs

echo "Starting training with NGC PyTorch..."
echo ""

# Install required packages and run training in NGC container
SUDO_PASSWORD="462773"
NGC_IMAGE="nvcr.io/nvidia/pytorch:24.08-py3"

echo "$SUDO_PASSWORD" | sudo -S docker run --rm \
  --gpus=all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$(pwd):/workspace" \
  -w /workspace \
  -e PYTHONUNBUFFERED=1 \
  $NGC_IMAGE \
  bash -c "
    set -e
    echo '=========================================='
    echo 'Installing required packages...'
    echo '=========================================='
    pip install transformers accelerate peft bitsandbytes datasets pyyaml
    echo '✅ Packages installed'
    echo ''
    echo '=========================================='
    echo 'Starting training...'
    echo '=========================================='
    python ko_centaur/training/train_deepseek_r1_qwen32b_qlora.py \
      --config ko_centaur/configs/training_deepseek_r1_qwen32b_qlora.yaml
  " 2>&1 | tee logs/deepseek-r1-32b-qlora_ngc_$(date +%Y%m%d_%H%M%S).log

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully"
else
    echo "❌ Training failed with exit code: $EXIT_CODE"
fi
echo "=========================================="

exit $EXIT_CODE
