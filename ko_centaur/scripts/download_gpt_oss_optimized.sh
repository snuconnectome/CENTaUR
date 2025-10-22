#!/bin/bash
#SBATCH --job-name=gpt-oss-download
#SBATCH --partition=debug
#SBATCH --nodelist=node1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=20G
#SBATCH --time=04:00:00
#SBATCH --output=/scratch/connectome/connectome1/ko-centaur/logs/gpt_oss_download_%j.out
#SBATCH --error=/scratch/connectome/connectome1/ko-centaur/logs/gpt_oss_download_%j.err

echo "=========================================="
echo "GPT-OSS-20B Optimized Download"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# Check /home storage
echo "Checking /home storage..."
df -h /home/connectome/connectome1
echo ""

# Create model directory in /home (261GB free)
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
echo "Starting optimized download"
echo "Excluding: original/* (13.7GB), metal/* (unnecessary)"
echo "Expected: ~14GB (3 safetensors + configs)"
echo "=========================================="
echo ""

cd "$MODEL_DIR"

# Download with exclusions
hf download openai/gpt-oss-20b \
    --exclude "original/*" "metal/*" \
    --local-dir gpt-oss-20b \
    --resume-download

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Download completed successfully"
    echo ""
    echo "Verifying files..."

    # Check safetensors files
    SAFETENSORS_COUNT=$(ls $MODEL_DIR/gpt-oss-20b/model-*.safetensors 2>/dev/null | wc -l)

    if [ "$SAFETENSORS_COUNT" -eq 3 ]; then
        echo "✅ Found 3 safetensors files"
        ls -lh $MODEL_DIR/gpt-oss-20b/model-*.safetensors
    else
        echo "⚠️  WARNING: Expected 3 safetensors files, found $SAFETENSORS_COUNT"
    fi

    # Check config files
    if [ -f "$MODEL_DIR/gpt-oss-20b/config.json" ] && \
       [ -f "$MODEL_DIR/gpt-oss-20b/tokenizer.json" ]; then
        echo "✅ Config and tokenizer files present"
    else
        echo "❌ ERROR: Missing config or tokenizer files"
        EXIT_CODE=1
    fi

    # Show total size
    echo ""
    echo "Total size:"
    du -sh $MODEL_DIR/gpt-oss-20b

    echo ""
    echo "🚀 Next: Run NF4 validation test with:"
    echo "   sbatch ko_centaur/scripts/submit_test_gpt_oss_nf4.sh"
else
    echo "❌ Download failed with exit code: $EXIT_CODE"
    echo "Check error log for details"
fi
echo "=========================================="

exit $EXIT_CODE
