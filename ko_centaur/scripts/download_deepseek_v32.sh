#!/bin/bash
#SBATCH --job-name=deepseek-v32-download
#SBATCH --partition=debug
#SBATCH --nodelist=node3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=50G
#SBATCH --time=12:00:00
#SBATCH --output=/scratch/connectome/connectome1/ko-centaur/logs/deepseek_v32_download_%j.out
#SBATCH --error=/scratch/connectome/connectome1/ko-centaur/logs/deepseek_v32_download_%j.err

echo "=========================================="
echo "DeepSeek-V3.2-Exp Download"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# WARNING: This is a VERY LARGE model (685B parameters)
echo "⚠️  WARNING: DeepSeek-V3.2-Exp is 685B parameters"
echo "   Full precision download: ~1.3TB"
echo "   Expected download size: ~300-400GB (safetensors)"
echo "   4-bit VRAM requirement: ~171GB"
echo ""

# Check /scratch storage
echo "Checking /scratch storage..."
df -h /scratch/connectome/connectome1
echo ""

# Create model directory in /scratch (largest partition)
MODEL_DIR="/scratch/connectome/connectome1/ko-centaur/models"
mkdir -p "$MODEL_DIR"

# Verify write permission
if [ ! -w "$MODEL_DIR" ]; then
    echo "❌ ERROR: No write permission to $MODEL_DIR"
    exit 1
fi

echo "✅ Model directory: $MODEL_DIR"
echo "Available space: $(df -h /scratch/connectome/connectome1 | tail -1 | awk '{print $4}')"
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
echo "Starting DeepSeek-V3.2-Exp download"
echo "Model: deepseek-ai/DeepSeek-V3.2-Exp"
echo "Expected duration: 3-6 hours (depending on network)"
echo "Expected size: ~300-400GB"
echo "=========================================="
echo ""

cd "$MODEL_DIR"

# Download with progress
hf download deepseek-ai/DeepSeek-V3.2-Exp \
    --local-dir deepseek-v3.2-exp \
    --local-dir-use-symlinks False

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Download completed successfully"
    echo ""
    echo "Verifying files..."

    # Check safetensors files
    SAFETENSORS_COUNT=$(ls $MODEL_DIR/deepseek-v3.2-exp/model-*.safetensors 2>/dev/null | wc -l)

    echo "✅ Found $SAFETENSORS_COUNT safetensors files"

    # Check config files
    if [ -f "$MODEL_DIR/deepseek-v3.2-exp/config.json" ] && \
       [ -f "$MODEL_DIR/deepseek-v3.2-exp/tokenizer.json" ]; then
        echo "✅ Config and tokenizer files present"
    else
        echo "❌ ERROR: Missing config or tokenizer files"
        EXIT_CODE=1
    fi

    # Show total size
    echo ""
    echo "Total size:"
    du -sh $MODEL_DIR/deepseek-v3.2-exp

    echo ""
    echo "⚠️  NEXT: Validation test will require ~171GB VRAM with 4-bit quantization"
    echo "   Your 4x GeForce GPUs (96GB total) may not be sufficient"
    echo "   Consider using 8-bit quantization or distributed inference"
else
    echo "❌ Download failed with exit code: $EXIT_CODE"
    echo "Check error log for details"
fi
echo "=========================================="

exit $EXIT_CODE
