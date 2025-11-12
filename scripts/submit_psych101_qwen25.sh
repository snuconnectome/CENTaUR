#!/bin/bash
#SBATCH --job-name=psych101-qwen25
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=4:00:00
#SBATCH --output=/scratch/connectome/connectome1/ko-centaur/logs/psych101_qwen25_%j.out

echo "Starting Qwen2.5-32B Psych-101 Multi-Task Evaluation"
echo "Time: $(date)"
echo "Node: $(hostname)"

# Activate conda environment
source /scratch/connectome/connectome1/miniconda3/etc/profile.d/conda.sh
conda activate ko-centaur

cd /scratch/connectome/connectome1/ko-centaur

# Evaluate all Phase 2 & 3 tasks (100 samples per task)
python scripts/evaluate_psych101_tasks.py \
    --model qwen25 \
    --tasks twostep nback iowa_gambling intertemporal decisions_description decisions_experience \
    --max-samples 100

echo "Evaluation complete"
echo "Time: $(date)"
