#!/bin/bash
#SBATCH --job-name=qwen25-32b-qlora
#SBATCH --partition=debug
#SBATCH --nodelist=node3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:geforce:4
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/connectome/connectome1/ko-centaur/logs/qwen25_32b_%j.out
#SBATCH --error=/scratch/connectome/connectome1/ko-centaur/logs/qwen25_32b_%j.err

echo "=========================================="
echo "Qwen2.5-32B QLoRA Training Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# Check GPU availability
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Activate conda environment
echo "Activating conda environment..."
source /scratch/connectome/connectome1/miniconda3/etc/profile.d/conda.sh
conda activate ko-centaur

# Verify environment
echo ""
echo "Python environment:"
python --version
echo ""

echo "Library versions:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "import bitsandbytes; print(f'BitsAndBytes: {bitsandbytes.__version__}')"
echo ""

# Verify model directory exists
MODEL_DIR="/scratch/connectome/connectome1/ko-centaur/models/qwen2.5-32b-instruct"
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ ERROR: Model directory not found: $MODEL_DIR"
    echo "Please ensure Qwen2.5-32B model is downloaded first"
    exit 1
fi

# Check if model files are complete
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "❌ ERROR: Model files incomplete (config.json missing)"
    echo "Download may still be in progress"
    exit 1
fi

echo "✅ Model directory verified: $MODEL_DIR"
echo ""

# Create output directory
OUTPUT_DIR="/scratch/connectome/connectome1/ko-centaur/outputs/qwen25-32b-qlora"
mkdir -p "$OUTPUT_DIR"
echo "✅ Output directory: $OUTPUT_DIR"
echo ""

# Start training
echo "=========================================="
echo "Starting Qwen2.5-32B QLoRA Training"
echo "GPUs: 4 (SLURM-allocated)"
echo "Quantization: NF4 4-bit"
echo "Expected GPU Memory: 18-22GB per GPU"
echo "Expected Duration: 12-15 hours (4 GPU)"
echo "=========================================="
echo ""

cd /scratch/connectome/connectome1/ko-centaur

# Let SLURM handle GPU allocation
python train_qwen25_32b_qlora.py \
    --config configs/training_qwen25_32b_qlora.yaml

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully"
    echo "LoRA adapters saved to: $OUTPUT_DIR"
else
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo "Check error log for details"
fi
echo "=========================================="

exit $EXIT_CODE
