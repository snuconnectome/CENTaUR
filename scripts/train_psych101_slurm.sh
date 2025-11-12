#!/bin/bash
#SBATCH --job-name=ko-centaur-train
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/connectome/connectome1/ko-centaur/logs/slurm-%j.out
#SBATCH --error=/scratch/connectome/connectome1/ko-centaur/logs/slurm-%j.err

# ======================================================================
# Ko-CENTaUR Training via SLURM
# ======================================================================

echo "=========================================="
echo "Ko-CENTaUR SLURM Training Job"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Activate conda environment
source /scratch/connectome/connectome1/miniconda3/bin/activate ko-centaur

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Navigate to working directory
cd /scratch/connectome/connectome1/ko-centaur

# Run training script
echo "Starting training..."
python /Users/jiookcha/Documents/git/CENTaUR/ko_centaur/training/train_psych101_full_slurm.py

echo ""
echo "=========================================="
echo "Training job completed"
echo "End time: $(date)"
echo "=========================================="
