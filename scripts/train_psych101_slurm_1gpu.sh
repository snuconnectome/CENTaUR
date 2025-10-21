#!/bin/bash
#SBATCH --job-name=ko-centaur-1gpu
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx:1
#SBATCH --mem=64G
#SBATCH --time=100:00:00
#SBATCH --output=/scratch/connectome/connectome1/ko-centaur/logs/slurm-%j.out
#SBATCH --error=/scratch/connectome/connectome1/ko-centaur/logs/slurm-%j.err

# ======================================================================
# Ko-CENTaUR Training via SLURM (Single GPU - Stable)
# ======================================================================

echo "=========================================="
echo "Ko-CENTaUR SLURM Training Job (1 GPU)"
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

# CRITICAL: Force single GPU usage (prevent accelerate from spawning multi-GPU workers)
# This was the root cause of 237 it/s → 0.12 it/s speed degradation
export CUDA_VISIBLE_DEVICES=0

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

# Single GPU training (OPTIMIZED: 4-bit quantization - faster than 8-bit)
echo "Starting training with 4-bit quantization..."
echo "Configuration: batch_size=2, gradient_accumulation=4, max_seq_length=1024"
echo "Optimizations: 4-bit quantization (faster than 8-bit)"
echo "Starting fresh (no checkpoint resume)"
echo ""
python ko_centaur/training/train_psych101_full_slurm.py

echo ""
echo "=========================================="
echo "Training job completed"
echo "End time: $(date)"
echo "=========================================="
