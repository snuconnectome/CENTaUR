#!/bin/bash

# Ko-CENTaUR Server Environment Setup Script
# Location: /scratch/connectome/connectome1/ko-centaur

set -e

SCRATCH_BASE="/scratch/connectome/connectome1"
WORK_DIR="$SCRATCH_BASE/ko-centaur"
CONDA_PATH="$SCRATCH_BASE/miniconda3"
ENV_NAME="ko-centaur"

echo "=== Ko-CENTaUR Server Setup ==="
echo "Work directory: $WORK_DIR"
echo "Conda path: $CONDA_PATH"
echo ""

# Setup directories
mkdir -p $WORK_DIR/pip-cache
mkdir -p $WORK_DIR/tmp
mkdir -p $WORK_DIR/models
mkdir -p $WORK_DIR/data
mkdir -p $WORK_DIR/cache

# Set environment variables
export TMPDIR=$WORK_DIR/tmp
export PIP_CACHE_DIR=$WORK_DIR/pip-cache
export HF_HOME=$WORK_DIR/cache
export TORCH_HOME=$WORK_DIR/models
export TRANSFORMERS_CACHE=$WORK_DIR/cache

echo "✅ Directories created"
echo "✅ Environment variables set"
echo ""

# Activate conda environment
source $CONDA_PATH/bin/activate $ENV_NAME
echo "✅ Conda environment activated: $ENV_NAME"
echo ""

# Install PyTorch with CUDA 12.1
echo "📦 Installing PyTorch 2.1.0 with CUDA 12.1..."
pip install --no-cache-dir torch==2.1.0 torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu121

echo ""
echo "✅ PyTorch installed"
echo ""

# Install HuggingFace and ML packages
echo "📦 Installing Transformers, PEFT, Datasets..."
pip install --no-cache-dir \
    transformers==4.36.0 \
    datasets==2.15.0 \
    peft==0.7.0 \
    accelerate \
    bitsandbytes \
    scipy \
    scikit-learn \
    pandas \
    "numpy<2"

echo ""
echo "✅ All packages installed"
echo ""

# Verify installation
echo "🔍 Verifying installation..."
python3 << 'EOF'
import torch
import transformers
import peft
import datasets

print("\n✅ Package versions:")
print(f"  PyTorch: {torch.__version__}")
print(f"  Transformers: {transformers.__version__}")
print(f"  PEFT: {peft.__version__}")
print(f"  Datasets: {datasets.__version__}")
print(f"\n✅ CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  CUDA version: {torch.version.cuda}")
    print(f"  GPU count: {torch.cuda.device_count()}")
    for i in range(torch.cuda.device_count()):
        print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
EOF

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To use this environment:"
echo "  source $CONDA_PATH/bin/activate $ENV_NAME"
echo "  export TMPDIR=$WORK_DIR/tmp"
echo "  export HF_HOME=$WORK_DIR/cache"
echo "  export TORCH_HOME=$WORK_DIR/models"
echo ""
