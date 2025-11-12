#!/bin/bash
#SBATCH --job-name=horizon-qwen25
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=3:00:00
#SBATCH --output=/scratch/connectome/connectome1/ko-centaur/logs/horizon_qwen25_%j.out

echo "Starting Qwen2.5-32B Horizon Task evaluation"
echo "Time: $(date)"
echo "Node: $(hostname)"

# Activate conda environment
source /scratch/connectome/connectome1/miniconda3/etc/profile.d/conda.sh
conda activate ko-centaur

cd /scratch/connectome/connectome1/ko-centaur

# Evaluate on both experiments with 10 participants each
python scripts/evaluate_horizon_task.py --model qwen25 --dataset exp1 --max-participants 10
python scripts/evaluate_horizon_task.py --model qwen25 --dataset exp2 --max-participants 10

echo "Evaluation complete"
echo "Time: $(date)"
