#!/bin/bash
#SBATCH --job-name=test-gpt-oss-nf4
#SBATCH --partition=debug
#SBATCH --nodelist=node3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:geforce:1
#SBATCH --mem=50G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/connectome/connectome1/ko-centaur/logs/test_gpt_oss_nf4_%j.out
#SBATCH --error=/scratch/connectome/connectome1/ko-centaur/logs/test_gpt_oss_nf4_%j.err

echo "=========================================="
echo "GPT-OSS-20B NF4 Validation Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# Verify model directory exists
MODEL_DIR="/scratch/connectome/connectome1/ko-centaur/models/gpt-oss-20b-nf4"
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ ERROR: Model directory not found: $MODEL_DIR"
    exit 1
fi

# Check if model files are complete
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "❌ ERROR: Model files incomplete (config.json missing)"
    echo "Download may still be in progress"
    exit 1
fi

# Check for safetensors files
SAFETENSORS_COUNT=$(ls $MODEL_DIR/model*.safetensors 2>/dev/null | wc -l)
if [ "$SAFETENSORS_COUNT" -eq 0 ]; then
    echo "❌ ERROR: No safetensors files found"
    echo "Download may still be in progress"
    exit 1
fi

echo "✅ Model directory verified: $MODEL_DIR"
echo "✅ Found $SAFETENSORS_COUNT safetensors files"
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
python -c "import bitsandbytes; print(f'BitsAndBytes: {bitsandbytes.__version__}')"
echo ""

# GPU status
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Run validation test
echo "=========================================="
echo "Running NF4 Validation Test"
echo "Expected: 3 tests (loading, extraction, batching)"
echo "=========================================="
echo ""

cd /scratch/connectome/connectome1/ko-centaur

python ko_centaur/training/test_gpt_oss_nf4.py \
    --model-path "$MODEL_DIR"

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Validation test PASSED"
    echo "NF4 model ready for CENTaUR workflow"
    echo "Next: Create query.py for full dataset processing"
else
    echo "❌ Validation test FAILED"
    echo "Check error log for details"
fi
echo "=========================================="

exit $EXIT_CODE
