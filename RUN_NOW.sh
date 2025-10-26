#!/bin/bash
#
# Ko-CENTaUR Pipeline - One-Click Execution Script
# Run this on the server: bash RUN_NOW.sh
#

set -e  # Exit on error

echo "╔════════════════════════════════════════════════════════════╗"
echo "║          Ko-CENTaUR Evaluation Pipeline                    ║"
echo "║          Starting Complete Execution                       ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

# ============================================================
# 0. Environment Setup
# ============================================================

echo "📦 Step 0: Environment Setup"
echo "─────────────────────────────────────────────────────────────"

# Check we're in the right directory
if [ ! -f "scripts/extract_centaur_features.py" ]; then
    echo "❌ ERROR: Not in ko-centaur directory!"
    echo "   Please run: cd /scratch/connectome/connectome1/ko-centaur"
    exit 1
fi

# Activate conda environment
source ~/.bashrc
conda activate centaur

echo "✅ Working directory: $(pwd)"
echo "✅ Conda environment: $CONDA_DEFAULT_ENV"
echo "✅ Python: $(which python)"
echo ""

# Create directories
mkdir -p data/features data/results logs

# Verify packages
python -c "import torch, transformers, peft" || {
    echo "❌ ERROR: Required packages not found!"
    echo "   Please install: pip install torch transformers peft"
    exit 1
}

echo "✅ All packages available"
echo ""

# ============================================================
# 1. Feature Extraction - Qwen2.5
# ============================================================

echo "🚀 Step 1: Feature Extraction - Qwen2.5-32B"
echo "─────────────────────────────────────────────────────────────"

if [ -f "data/features/centaur_features_qwen25.pth" ]; then
    echo "⚠️  Features already exist, skipping extraction"
else
    echo "Submitting Qwen2.5 feature extraction job..."
    JOB1=$(sbatch scripts/submit_extract_qwen25.sh | awk '{print $4}')
    echo "✅ Job submitted: $JOB1"

    echo "Waiting for completion (this takes 1-2 hours)..."
    while squeue -j $JOB1 2>/dev/null | grep -q $JOB1; do
        echo "  $(date +%H:%M:%S) - Job $JOB1 still running..."
        sleep 60
    done

    # Verify output
    if [ -f "data/features/centaur_features_qwen25.pth" ]; then
        echo "✅ Qwen2.5 features extracted successfully!"
        python -c "
import torch
data = torch.load('data/features/centaur_features_qwen25.pth')
print(f'   Shape: {data[\"features\"].shape}')
"
    else
        echo "❌ ERROR: Feature extraction failed!"
        echo "   Check logs: logs/extract_qwen25_${JOB1}.err"
        exit 1
    fi
fi

echo ""

# ============================================================
# 2. Feature Extraction - DeepSeek
# ============================================================

echo "🚀 Step 2: Feature Extraction - DeepSeek-R1"
echo "─────────────────────────────────────────────────────────────"

if [ -f "data/features/centaur_features_deepseek.pth" ]; then
    echo "⚠️  Features already exist, skipping extraction"
else
    echo "Submitting DeepSeek feature extraction job..."
    JOB2=$(sbatch scripts/submit_extract_deepseek.sh | awk '{print $4}')
    echo "✅ Job submitted: $JOB2"

    echo "Waiting for completion (this takes 1-2 hours)..."
    while squeue -j $JOB2 2>/dev/null | grep -q $JOB2; do
        echo "  $(date +%H:%M:%S) - Job $JOB2 still running..."
        sleep 60
    done

    # Verify output
    if [ -f "data/features/centaur_features_deepseek.pth" ]; then
        echo "✅ DeepSeek features extracted successfully!"
        python -c "
import torch
data = torch.load('data/features/centaur_features_deepseek.pth')
print(f'   Shape: {data[\"features\"].shape}')
"
    else
        echo "❌ ERROR: Feature extraction failed!"
        echo "   Check logs: logs/extract_deepseek_${JOB2}.err"
        exit 1
    fi
fi

echo ""

# ============================================================
# 3. LOO CV - Qwen2.5
# ============================================================

echo "🧮 Step 3: 100-fold LOO CV - Qwen2.5"
echo "─────────────────────────────────────────────────────────────"

if [ -f "data/results/loo_cv_results_qwen25.pth" ]; then
    echo "⚠️  Results already exist, skipping CV"
else
    echo "Submitting Qwen2.5 LOO CV job..."
    JOB3=$(sbatch scripts/submit_fit_qwen25.sh | awk '{print $4}')
    echo "✅ Job submitted: $JOB3"

    echo "Waiting for completion (this takes 1-2 hours)..."
    while squeue -j $JOB3 2>/dev/null | grep -q $JOB3; do
        echo "  $(date +%H:%M:%S) - Job $JOB3 still running..."
        sleep 60
    done

    # Verify output
    if [ -f "data/results/loo_cv_results_qwen25.pth" ]; then
        echo "✅ Qwen2.5 LOO CV complete!"
    else
        echo "❌ ERROR: LOO CV failed!"
        echo "   Check logs: logs/fit_qwen25_${JOB3}.err"
        exit 1
    fi
fi

echo ""

# ============================================================
# 4. LOO CV - DeepSeek
# ============================================================

echo "🧮 Step 4: 100-fold LOO CV - DeepSeek"
echo "─────────────────────────────────────────────────────────────"

if [ -f "data/results/loo_cv_results_deepseek.pth" ]; then
    echo "⚠️  Results already exist, skipping CV"
else
    echo "Submitting DeepSeek LOO CV job..."
    JOB4=$(sbatch scripts/submit_fit_deepseek.sh | awk '{print $4}')
    echo "✅ Job submitted: $JOB4"

    echo "Waiting for completion (this takes 1-2 hours)..."
    while squeue -j $JOB4 2>/dev/null | grep -q $JOB4; do
        echo "  $(date +%H:%M:%S) - Job $JOB4 still running..."
        sleep 60
    done

    # Verify output
    if [ -f "data/results/loo_cv_results_deepseek.pth" ]; then
        echo "✅ DeepSeek LOO CV complete!"
    else
        echo "❌ ERROR: LOO CV failed!"
        echo "   Check logs: logs/fit_deepseek_${JOB4}.err"
        exit 1
    fi
fi

echo ""

# ============================================================
# 5. Display Results
# ============================================================

echo "╔════════════════════════════════════════════════════════════╗"
echo "║                    FINAL RESULTS                           ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""

python -c "
import torch

# Load results
qwen_data = torch.load('data/results/loo_cv_results_qwen25.pth')
deep_data = torch.load('data/results/loo_cv_results_deepseek.pth')

print('Ko-CENTaUR Evaluation Results')
print('='*70)
print()

# Qwen2.5 Results
print('📊 Qwen2.5-32B-QLoRA:')
print(f'   Average NLL:  {qwen_data[\"avg_nll\"]:.4f} ± {qwen_data[\"std_nll\"]:.4f}')
print(f'   Total NLL:    {qwen_data[\"total_nll\"]:.0f}')
print(f'   Accuracy:     {qwen_data[\"accuracy\"]:.1%}')
print()

# DeepSeek Results
print('📊 DeepSeek-R1-32B-QLoRA:')
print(f'   Average NLL:  {deep_data[\"avg_nll\"]:.4f} ± {deep_data[\"std_nll\"]:.4f}')
print(f'   Total NLL:    {deep_data[\"total_nll\"]:.0f}')
print(f'   Accuracy:     {deep_data[\"accuracy\"]:.1%}')
print()

# Benchmark Comparison
print('📈 Benchmark Comparison:')
print('   Random Baseline:  ~120,000 NLL')
print('   LLaMA-65B (orig): ~30,000 NLL')
print(f'   Qwen2.5-32B:      {qwen_data[\"total_nll\"]:.0f} NLL')
print(f'   DeepSeek-R1:      {deep_data[\"total_nll\"]:.0f} NLL')
print()

# Winner
qwen_nll = qwen_data['total_nll']
deep_nll = deep_data['total_nll']
if qwen_nll < deep_nll:
    improvement = (deep_nll - qwen_nll) / deep_nll * 100
    print(f'🏆 Winner: Qwen2.5-32B ({improvement:.1f}% better than DeepSeek)')
else:
    improvement = (qwen_nll - deep_nll) / qwen_nll * 100
    print(f'🏆 Winner: DeepSeek-R1 ({improvement:.1f}% better than Qwen2.5)')

print()
print('='*70)
"

echo ""
echo "╔════════════════════════════════════════════════════════════╗"
echo "║              PIPELINE COMPLETE! 🎉                         ║"
echo "╚════════════════════════════════════════════════════════════╝"
echo ""
echo "📁 Output files:"
echo "   - data/features/centaur_features_qwen25.pth"
echo "   - data/features/centaur_features_deepseek.pth"
echo "   - data/results/loo_cv_results_qwen25.pth"
echo "   - data/results/loo_cv_results_deepseek.pth"
echo ""
echo "📊 Next steps:"
echo "   - Save results to permanent location"
echo "   - Update documentation with actual NLL values"
echo "   - Analyze alpha selection patterns"
echo "   - Compare detailed per-sample performance"
echo ""
