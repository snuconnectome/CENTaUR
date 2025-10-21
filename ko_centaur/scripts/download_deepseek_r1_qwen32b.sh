#!/bin/bash
#SBATCH --job-name=deepseek-r1-qwen32b-download
#SBATCH --partition=debug
#SBATCH --nodelist=node3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/connectome/connectome1/ko-centaur/logs/deepseek_r1_qwen32b_download_%j.out
#SBATCH --error=/scratch/connectome/connectome1/ko-centaur/logs/deepseek_r1_qwen32b_download_%j.err

echo "=========================================="
echo "DeepSeek-R1-Distill-Qwen-32B Download"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

echo "✅ Model: DeepSeek-R1-Distill-Qwen-32B"
echo "   Parameters: 32B"
echo "   Performance: Outperforms OpenAI o1-mini"
echo "   Expected size: ~65GB"
echo "   4-bit VRAM requirement: ~20GB (fits single GPU)"
echo ""

# Check /home storage (261GB free)
echo "Checking /home storage..."
df -h /home/connectome/connectome1
echo ""

# Create model directory in /home (better space)
MODEL_DIR="/home/connectome/connectome1/models"
mkdir -p "$MODEL_DIR"

# Verify write permission
if [ ! -w "$MODEL_DIR" ]; then
    echo "❌ ERROR: No write permission to $MODEL_DIR"
    exit 1
fi

echo "✅ Model directory: $MODEL_DIR"
echo "Available space: $(df -h /home/connectome/connectome1 | tail -1 | awk '{print $4}')"
echo ""

# Activate conda environment
echo "Activating conda environment..."
source /scratch/connectome/connectome1/miniconda3/etc/profile.d/conda.sh
conda activate ko-centaur

# Verify hf command
if ! command -v hf &> /dev/null; then
    echo "❌ ERROR: hf command not found"
    echo "Trying to install huggingface_hub..."
    pip install -U "huggingface_hub[cli]"
fi

echo ""
echo "=========================================="
echo "Starting DeepSeek-R1-Distill-Qwen-32B download"
echo "Model: deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
echo "Expected duration: 1-2 hours"
echo "Expected size: ~65GB"
echo "=========================================="
echo ""

cd "$MODEL_DIR"

# Download with progress
hf download deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --local-dir deepseek-r1-distill-qwen-32b \
    --local-dir-use-symlinks False

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Download completed successfully"
    echo ""
    echo "Verifying files..."

    # Check safetensors files
    SAFETENSORS_COUNT=$(ls $MODEL_DIR/deepseek-r1-distill-qwen-32b/model-*.safetensors 2>/dev/null | wc -l)

    echo "✅ Found $SAFETENSORS_COUNT safetensors files"

    # Check config files
    if [ -f "$MODEL_DIR/deepseek-r1-distill-qwen-32b/config.json" ] && \
       [ -f "$MODEL_DIR/deepseek-r1-distill-qwen-32b/tokenizer.json" ]; then
        echo "✅ Config and tokenizer files present"
    else
        echo "❌ ERROR: Missing config or tokenizer files"
        EXIT_CODE=1
    fi

    # Show total size
    echo ""
    echo "Total size:"
    du -sh $MODEL_DIR/deepseek-r1-distill-qwen-32b

    echo ""
    echo "🚀 Next: Run validation test with:"
    echo "   sbatch ko_centaur/scripts/submit_test_deepseek_r1_qwen32b_nf4.sh"
    echo ""
    echo "Expected performance (from benchmarks):"
    echo "   AIME 2024: 86.0 (better than o1-mini's 63.6)"
    echo "   AIME 2025: 76.3"
    echo "   GPQA Diamond: 61.1"
else
    echo "❌ Download failed with exit code: $EXIT_CODE"
    echo "Check error log for details"
fi
echo "=========================================="

exit $EXIT_CODE
