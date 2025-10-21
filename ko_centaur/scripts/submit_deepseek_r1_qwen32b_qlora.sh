#!/bin/bash
#SBATCH --job-name=deepseek-r1-qwen32b-qlora
#SBATCH --partition=debug
#SBATCH --nodelist=node3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:geforce:4
#SBATCH --mem=200G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/connectome/connectome1/ko-centaur/logs/deepseek_r1_qwen32b_%j.out
#SBATCH --error=/scratch/connectome/connectome1/ko-centaur/logs/deepseek_r1_qwen32b_%j.err

echo "=========================================="
echo "DeepSeek-R1-Distill-Qwen-32B QLoRA Training Job"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

echo "🏆 Model: DeepSeek-R1-Distill-Qwen-32B"
echo "   Performance: Outperforms OpenAI o1-mini"
echo "   Parameters: 32B"
echo "   Base: Qwen2.5-32B architecture"
echo "   Distilled from: 671B DeepSeek-R1"
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

# Verify model directory exists (in /home)
MODEL_DIR="/home/connectome/connectome1/models/deepseek-r1-distill-qwen-32b"
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ ERROR: Model directory not found: $MODEL_DIR"
    echo "Please ensure DeepSeek-R1-Distill-Qwen-32B model is downloaded first"
    echo "Run: sbatch ko_centaur/scripts/download_deepseek_r1_qwen32b.sh"
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
OUTPUT_DIR="/scratch/connectome/connectome1/ko-centaur/outputs/deepseek-r1-qwen32b-qlora"
mkdir -p "$OUTPUT_DIR"
echo "✅ Output directory: $OUTPUT_DIR"
echo ""

# Start training
echo "=========================================="
echo "Starting DeepSeek-R1-Distill-Qwen-32B QLoRA Training"
echo "GPUs: 4 (SLURM-allocated)"
echo "Quantization: NF4 4-bit"
echo "Expected GPU Memory: 18-22GB per GPU"
echo "Expected Duration: 12-15 hours (4 GPU)"
echo "Benchmark Performance: Better than o1-mini!"
echo "=========================================="
echo ""

cd /scratch/connectome/connectome1/ko-centaur

# Let SLURM handle GPU allocation
python ko_centaur/training/train_deepseek_r1_qwen32b_qlora.py \
    --config ko_centaur/configs/training_deepseek_r1_qwen32b_qlora.yaml

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Training completed successfully"
    echo "LoRA adapters saved to: $OUTPUT_DIR"
    echo ""
    echo "🏆 Model Performance (from benchmarks):"
    echo "   - AIME 2024: 86.0 (vs o1-mini 63.6)"
    echo "   - AIME 2025: 76.3"
    echo "   - GPQA Diamond: 61.1"
    echo "   - Superior reasoning capabilities!"
else
    echo "❌ Training failed with exit code: $EXIT_CODE"
    echo "Check error log for details"
fi
echo "=========================================="

exit $EXIT_CODE
