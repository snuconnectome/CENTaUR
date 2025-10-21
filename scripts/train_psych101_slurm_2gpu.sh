#!/bin/bash
#SBATCH --job-name=ko-centaur-2gpu
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx:2
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=/scratch/connectome/connectome1/ko-centaur/logs/slurm-%j.out
#SBATCH --error=/scratch/connectome/connectome1/ko-centaur/logs/slurm-%j.err

# ======================================================================
# Ko-CENTaUR Training via SLURM (2 GPUs with DDP)
# ======================================================================

echo "=========================================="
echo "Ko-CENTaUR SLURM Training Job (2 GPUs)"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Set TMPDIR to use scratch space (avoid root filesystem full issue)
export TMPDIR=/scratch/connectome/connectome1/ko-centaur/tmp

# Set PyTorch CUDA memory allocator settings to reduce fragmentation
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

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

# DDP automatically activated by Trainer when multiple GPUs detected
echo "Starting training with DDP (2 GPUs)..."
python train_psych101_full_slurm.py

echo ""
echo "=========================================="
echo "Training job completed"
echo "End time: $(date)"
echo "=========================================="
