#!/bin/bash
set -e

echo "=== CENTaUR Environment Setup ==="
echo "Starting at: $(date)"

cd ~/git/CENTaUR

# Activate venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install PyTorch with CUDA support
echo "Installing PyTorch with CUDA..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install other requirements
echo "Installing other packages..."
pip install -r requirements.txt

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
"

echo ""
echo "=== Installation Complete at: $(date) ==="
