#!/bin/bash
#SBATCH --job-name=extract_qwen25
#SBATCH --partition=octopus
#SBATCH --nodelist=node1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/connectome/connectome1/ko-centaur/logs/extract_qwen25_%j.out
#SBATCH --error=/scratch/connectome/connectome1/ko-centaur/logs/extract_qwen25_%j.err

# ============================================================
# CENTaUR Feature Extraction - Qwen2.5-32B
# ============================================================
#
# Purpose: Extract last-layer hidden states from fine-tuned
#          Qwen2.5-32B for downstream binomial regression
#
# Expected:
#   - GPU Memory: ~21GB (4-bit quantized)
#   - Time: 1-2 hours for 100 samples
#   - Output: features (100, 3072) tensor
#
# ============================================================

set -e  # Exit on error

echo "=================================="
echo "CENTaUR Feature Extraction"
echo "Model: Qwen2.5-32B-QLoRA"
echo "=================================="
echo ""

# Environment
cd /scratch/connectome/connectome1/ko-centaur
source ~/.bashrc
conda activate centaur

echo "Environment:"
echo "  Working dir: $(pwd)"
echo "  Conda env: $CONDA_DEFAULT_ENV"
echo "  Python: $(which python)"
echo "  CUDA: $CUDA_VISIBLE_DEVICES"
echo ""

# GPU Info
echo "GPU Information:"
nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv
echo ""

# Run extraction
echo "Starting extraction..."
echo "-----------------------------------"

python scripts/extract_centaur_features.py \
    --model qwen25

echo "-----------------------------------"
echo "Extraction complete!"
echo ""

# Show output
OUTPUT_FILE="/scratch/connectome/connectome1/ko-centaur/data/features/centaur_features_qwen25.pth"
if [ -f "$OUTPUT_FILE" ]; then
    echo "Output file created:"
    ls -lh "$OUTPUT_FILE"
    echo ""

    # Quick verification
    python -c "
import torch
data = torch.load('$OUTPUT_FILE')
print('Feature shape:', data['features'].shape)
print('Labels shape:', data['labels'].shape)
print('Hidden dim:', data['hidden_dim'])
print('Model:', data['model_name'])
"
else
    echo "ERROR: Output file not found!"
    exit 1
fi

echo ""
echo "=================================="
echo "Job Complete!"
echo "=================================="
