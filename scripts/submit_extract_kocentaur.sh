#!/bin/bash
#SBATCH --job-name=extract_kocentaur
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00
#SBATCH --output=/scratch/connectome/connectome1/ko-centaur/logs/extract_kocentaur_%j.out
#SBATCH --error=/scratch/connectome/connectome1/ko-centaur/logs/extract_kocentaur_%j.err

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

# Run feature extraction for Ko-CENTaUR (EXAONE + Psych-101)
echo "Extracting features from Ko-CENTaUR (EXAONE-3.0-7.8B + Psych-101)..."
python scripts/extract_centaur_features.py --model ko-centaur

echo "===================================================="
echo "End Time: $(date)"
echo "===================================================="
