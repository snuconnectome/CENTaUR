#!/bin/bash
# Fix PyTorch CUDA version mismatch
# Server has CUDA 11.4 driver, but PyTorch 2.5.1+cu121 requires 12.1

source /scratch/connectome/connectome1/miniconda3/bin/activate ko-centaur

echo "Current PyTorch version:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}')"

echo ""
echo "Uninstalling current PyTorch..."
pip uninstall -y torch torchvision torchaudio

echo ""
echo "Installing PyTorch 2.5.1 with CUDA 11.8 (compatible with driver 11.4)..."
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu118

echo ""
echo "Verifying installation:"
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.version.cuda}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU count: {torch.cuda.device_count()}')"

echo ""
echo "✅ PyTorch CUDA version fixed!"
