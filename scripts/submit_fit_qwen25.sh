#!/bin/bash
#SBATCH --job-name=fit_qwen25
#SBATCH --partition=octopus
#SBATCH --nodelist=node1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/scratch/connectome/connectome1/ko-centaur/logs/fit_qwen25_%j.out
#SBATCH --error=/scratch/connectome/connectome1/ko-centaur/logs/fit_qwen25_%j.err

# ============================================================
# CENTaUR 100-fold LOO Cross-Validation - Qwen2.5-32B
# ============================================================
#
# Purpose: Fit binomial regression with 100-fold LOO CV
#          following original Binz & Schulz (2023) methodology
#
# Expected:
#   - CPU only (no GPU needed for regression)
#   - Time: 1-2 hours for 100 folds
#   - Output: NLL metric comparable to original paper
#
# ============================================================

set -e  # Exit on error

echo "=================================="
echo "CENTaUR 100-fold LOO CV"
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
echo ""

# Check if features exist
FEATURES_FILE="data/features/centaur_features_qwen25.pth"
if [ ! -f "$FEATURES_FILE" ]; then
    echo "ERROR: Features file not found!"
    echo "Please run feature extraction first:"
    echo "  sbatch scripts/submit_extract_qwen25.sh"
    exit 1
fi

echo "Features file found:"
ls -lh "$FEATURES_FILE"
echo ""

# Run LOO CV
echo "Starting 100-fold LOO Cross-Validation..."
echo "-----------------------------------"

python scripts/fit_centaur_loo_cv.py \
    --model qwen25

echo "-----------------------------------"
echo "Cross-validation complete!"
echo ""

# Show output
OUTPUT_FILE="data/results/loo_cv_results_qwen25.pth"
if [ -f "$OUTPUT_FILE" ]; then
    echo "Results file created:"
    ls -lh "$OUTPUT_FILE"
    echo ""

    # Quick verification
    python -c "
import torch
data = torch.load('$OUTPUT_FILE')
print('='*60)
print('RESULTS SUMMARY')
print('='*60)
print(f'Model: {data[\"model_name\"]}')
print(f'Samples: {data[\"n_samples\"]}')
print(f'Hidden dim: {data[\"hidden_dim\"]}')
print(f'\nNegative Log-Likelihood (NLL):')
print(f'  Average NLL: {data[\"avg_nll\"]:.4f} ± {data[\"std_nll\"]:.4f}')
print(f'  Total NLL:   {data[\"total_nll\"]:.2f}')
print(f'\nAccuracy (reference):')
print(f'  Accuracy: {data[\"accuracy\"]:.1%}')
print(f'\nBenchmark Comparison:')
print(f'  Random baseline:  ~120,000 NLL')
print(f'  LLaMA-65B (orig): ~30,000 NLL')
print(f'  Qwen2.5-32B:      {data[\"total_nll\"]:.0f} NLL')
print('='*60)
"
else
    echo "ERROR: Output file not found!"
    exit 1
fi

echo ""
echo "=================================="
echo "Job Complete!"
echo "=================================="
