#!/bin/bash
set -e

echo "=== CENTaUR Environment Setup (ARM64) ==="
echo "Starting at: $(date)"

cd ~/git/CENTaUR

# Activate venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch for ARM64
echo "Installing PyTorch for ARM64..."
pip install torch torchvision torchaudio

# Install other requirements
echo "Installing other packages..."
pip install transformers peft accelerate datasets scipy numpy tqdm sentencepiece

# Try to install bitsandbytes (may not work on ARM)
pip install bitsandbytes || echo "⚠️ bitsandbytes not available on ARM64"

# Verify installation
echo ""
echo "=== Verification ==="
python -c "
import torch
import transformers
import peft
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ PEFT: {peft.__version__}')
if torch.cuda.is_available():
    print(f'✅ CUDA version: {torch.version.cuda}')
    print(f'✅ GPU: {torch.cuda.get_device_name(0)}')
else:
    print('⚠️ CUDA not available - will run on CPU')
"

echo ""
echo "=== Installation Complete at: $(date) ==="
