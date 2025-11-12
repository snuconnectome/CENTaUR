#!/bin/bash
#SBATCH --job-name=extract_qwen25_base
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/connectome/connectome1/ko-centaur/logs/extract_qwen25_base_%j.out
#SBATCH --error=/scratch/connectome/connectome1/ko-centaur/logs/extract_qwen25_base_%j.err

echo "===================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "Start Time: $(date)"
echo "===================================================="

# Activate conda environment
source /scratch/connectome/connectome1/miniconda3/bin/activate ko-centaur

# Change to project directory
cd /scratch/connectome/connectome1/ko-centaur

# Run feature extraction for Qwen2.5-32B-Base
echo "Extracting features from Qwen2.5-32B-Instruct (Base, no fine-tuning)..."
python scripts/extract_centaur_features.py --model qwen25-base

echo "===================================================="
echo "End Time: $(date)"
echo "===================================================="
