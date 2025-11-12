#!/bin/bash
#SBATCH --job-name=extract_exaone_base
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/connectome/connectome1/ko-centaur/logs/extract_exaone_base_%j.out
#SBATCH --error=/scratch/connectome/connectome1/ko-centaur/logs/extract_exaone_base_%j.err

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

# Run feature extraction for EXAONE-3.0-7.8B-Instruct-Base
echo "Extracting features from EXAONE-3.0-7.8B-Instruct (Base)..."
python scripts/extract_centaur_features.py --model exaone-base

echo "===================================================="
echo "End Time: $(date)"
echo "===================================================="
