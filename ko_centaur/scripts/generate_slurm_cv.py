#!/usr/bin/env python3
"""
SLURM Cross-Validation Script Generator

Generates SLURM array job scripts for parallel 100-fold LOO cross-validation.
Configured for connectome server.

Usage:
    python scripts/generate_slurm_cv.py \
        --job_name kocentaur_loo_cv \
        --n_folds 100 \
        --time_limit "04:00:00" \
        --memory "32GB" \
        --gpus_per_task 1 \
        --checkpoint /scratch/connectome/connectome1/ko-centaur/models/ko_centaur_checkpoint \
        --dataset /scratch/connectome/connectome1/ko-centaur/data/raw/psych101_train.jsonl \
        --baselines exaone-base llama-3.2-3b \
        --output slurm_cv_job.sh
"""

import argparse
from pathlib import Path


def generate_slurm_script(
    job_name: str,
    n_folds: int,
    time_limit: str,
    memory: str,
    gpus_per_task: int,
    checkpoint: str,
    dataset: str,
    baselines: list,
    output_dir: str,
    partition: str = "debug"
):
    """Generate SLURM array job script"""

    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --partition={partition}
#SBATCH --array=0-{n_folds - 1}
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:rtx:{gpus_per_task}
#SBATCH --mem={memory}
#SBATCH --time={time_limit}
#SBATCH --output={output_dir}/logs/slurm-%A_%a.out
#SBATCH --error={output_dir}/logs/slurm-%A_%a.err

# ======================================================================
# Ko-CENTaUR LOO Cross-Validation (SLURM Array Job)
# ======================================================================

echo "=========================================="
echo "Ko-CENTaUR LOO CV - Fold $SLURM_ARRAY_TASK_ID"
echo "=========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Array Task ID: $SLURM_ARRAY_TASK_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Start time: $(date)"
echo ""

# Set TMPDIR to use scratch space
export TMPDIR={output_dir}/tmp
mkdir -p $TMPDIR

# Set PyTorch CUDA memory allocator settings
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Activate conda environment
source /scratch/connectome/connectome1/miniconda3/bin/activate ko-centaur

# Verify environment
echo "Python: $(which python)"
echo "Python version: $(python --version)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"
echo "GPU count: $(python -c 'import torch; print(torch.cuda.device_count())')"
echo ""

# Navigate to working directory
cd {output_dir.rsplit('/', 3)[0]}

# Create logs directory
mkdir -p {output_dir}/logs

# Run single fold
echo "Running fold $SLURM_ARRAY_TASK_ID..."
echo "Checkpoint: {checkpoint}"
echo "Dataset: {dataset}"
echo "Baselines: {' '.join(baselines)}"
echo ""

python ko_centaur/scripts/run_full_eval.py \\
    --checkpoint {checkpoint} \\
    --dataset {dataset} \\
    --baselines {' '.join(baselines)} \\
    --n_samples {n_folds} \\
    --fold_id $SLURM_ARRAY_TASK_ID \\
    --output_dir {output_dir} \\
    --skip_features

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "Fold $SLURM_ARRAY_TASK_ID completed successfully"
else
    echo "Fold $SLURM_ARRAY_TASK_ID failed with exit code $EXIT_CODE"
fi
echo "End time: $(date)"
echo "=========================================="

exit $EXIT_CODE
"""

    return script


def main():
    parser = argparse.ArgumentParser(
        description='Generate SLURM array job script for Ko-CENTaUR LOO CV'
    )
    parser.add_argument(
        '--job_name',
        type=str,
        default='kocentaur_loo_cv',
        help='SLURM job name'
    )
    parser.add_argument(
        '--n_folds',
        type=int,
        default=100,
        help='Number of folds (default: 100 for LOO)'
    )
    parser.add_argument(
        '--time_limit',
        type=str,
        default='04:00:00',
        help='Time limit per fold (default: 04:00:00)'
    )
    parser.add_argument(
        '--memory',
        type=str,
        default='32GB',
        help='Memory per task (default: 32GB)'
    )
    parser.add_argument(
        '--gpus_per_task',
        type=int,
        default=1,
        help='GPUs per task (default: 1)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='/scratch/connectome/connectome1/ko-centaur/models/ko_centaur_checkpoint',
        help='Path to Ko-CENTaUR checkpoint'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='/scratch/connectome/connectome1/ko-centaur/data/raw/psych101_train.jsonl',
        help='Path to Psych-101 dataset'
    )
    parser.add_argument(
        '--baselines',
        nargs='+',
        default=['exaone-base'],
        help='Baseline model names'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/scratch/connectome/connectome1/ko-centaur/results/full_eval',
        help='Output directory'
    )
    parser.add_argument(
        '--partition',
        type=str,
        default='debug',
        help='SLURM partition (default: debug)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='slurm_cv_job.sh',
        help='Output script filename'
    )

    args = parser.parse_args()

    # Generate script
    script = generate_slurm_script(
        job_name=args.job_name,
        n_folds=args.n_folds,
        time_limit=args.time_limit,
        memory=args.memory,
        gpus_per_task=args.gpus_per_task,
        checkpoint=args.checkpoint,
        dataset=args.dataset,
        baselines=args.baselines,
        output_dir=args.output_dir,
        partition=args.partition
    )

    # Save script
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        f.write(script)

    # Make executable
    output_path.chmod(0o755)

    print(f"✓ SLURM script generated: {output_path}")
    print()
    print("To submit:")
    print(f"  sbatch {output_path}")
    print()
    print("To monitor:")
    print(f"  squeue -u $USER")
    print(f"  tail -f {args.output_dir}/logs/slurm-*.out")
    print()
    print("After completion, collect results:")
    print(f"  python scripts/collect_cv_results.py --input_dir {args.output_dir} --n_folds {args.n_folds}")


if __name__ == "__main__":
    main()
