#!/bin/bash
#SBATCH --job-name=extract_deepseek_base
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/connectome/connectome1/ko-centaur/logs/extract_deepseek_base_%j.out
#SBATCH --error=/scratch/connectome/connectome1/ko-centaur/logs/extract_deepseek_base_%j.err

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

# Run feature extraction for DeepSeek-R1-32B-Base
echo "Extracting features from DeepSeek-R1-Distill-Qwen-32B (Base, no fine-tuning)..."
python scripts/extract_centaur_features.py --model deepseek-base

echo "===================================================="
echo "End Time: $(date)"
echo "===================================================="
