# Ko-CENTaUR Server Deployment Guide

**Last Updated**: 2025-10-26
**Purpose**: Step-by-step guide for running CENTaUR evaluation pipeline on server

---

## Pre-Deployment Checklist

### 1. Local Preparation (✅ COMPLETE)

- [x] Feature extraction scripts created
- [x] LOO CV scripts created
- [x] SLURM submission scripts ready
- [x] All scripts syntax validated
- [x] Documentation updated
- [x] Git commits complete

### 2. Server Prerequisites

**Check these on server before starting**:

```bash
# Connect to server
ssh server

# Navigate to project directory
cd /scratch/connectome/connectome1/ko-centaur

# Pull latest changes
git pull origin feature/ko-centaur-llm-strategy

# Verify conda environment
conda activate centaur
python --version  # Should be Python 3.10+

# Check required packages
python -c "import torch, transformers, peft; print('✅ All packages available')"

# Verify directory structure
ls -la scripts/extract_centaur_features.py
ls -la scripts/fit_centaur_loo_cv.py
ls -la models.py

# Create output directories
mkdir -p data/features
mkdir -p data/results
mkdir -p logs
```

---

## Execution Pipeline

### Phase 1: Feature Extraction (GPU Required)

**Expected Time**: 1-2 hours per model
**GPU Memory**: ~21GB (NF4 quantized)
**Output**: Features (100, 3072) tensors

#### 1A. Submit Qwen2.5 Feature Extraction

```bash
cd /scratch/connectome/connectome1/ko-centaur

# Submit job
sbatch scripts/submit_extract_qwen25.sh

# Get job ID
squeue -u $USER

# Monitor progress
tail -f logs/extract_qwen25_<JOB_ID>.out

# Watch GPU usage (optional)
watch -n 5 nvidia-smi
```

**Expected Output File**:
- `data/features/centaur_features_qwen25.pth`

**Verification**:
```bash
# After job completes, verify output
python -c "
import torch
data = torch.load('data/features/centaur_features_qwen25.pth')
print(f'Features: {data[\"features\"].shape}')
print(f'Labels: {data[\"labels\"].shape}')
print(f'Hidden dim: {data[\"hidden_dim\"]}')
print(f'Model: {data[\"model_name\"]}')
"
```

Expected output:
```
Features: torch.Size([100, 3072])
Labels: torch.Size([100])
Hidden dim: 3072
Model: Qwen2.5-32B-QLoRA
```

#### 1B. Submit DeepSeek Feature Extraction

```bash
# Same process for DeepSeek
sbatch scripts/submit_extract_deepseek.sh

# Monitor
tail -f logs/extract_deepseek_<JOB_ID>.out

# Verify output
python -c "
import torch
data = torch.load('data/features/centaur_features_deepseek.pth')
print(f'Features: {data[\"features\"].shape}')
print(f'Model: {data[\"model_name\"]}')
"
```

---

### Phase 2: 100-fold LOO Cross-Validation (CPU Only)

**Expected Time**: 1-2 hours per model
**CPU Memory**: ~32GB
**Output**: NLL metrics + full results

#### 2A. Submit Qwen2.5 LOO CV

**Prerequisites**: Must have completed Phase 1A

```bash
# Verify features exist
ls -lh data/features/centaur_features_qwen25.pth

# Submit LOO CV job
sbatch scripts/submit_fit_qwen25.sh

# Monitor progress
tail -f logs/fit_qwen25_<JOB_ID>.out

# Watch for progress updates
grep "Fold" logs/fit_qwen25_<JOB_ID>.out | tail -5
```

**Expected Output File**:
- `data/results/loo_cv_results_qwen25.pth`

**Verification**:
```bash
# After job completes, check results
python -c "
import torch
data = torch.load('data/results/loo_cv_results_qwen25.pth')
print('='*60)
print('QWEN2.5-32B RESULTS')
print('='*60)
print(f'Average NLL: {data[\"avg_nll\"]:.4f} ± {data[\"std_nll\"]:.4f}')
print(f'Total NLL: {data[\"total_nll\"]:.2f}')
print(f'Accuracy: {data[\"accuracy\"]:.1%}')
print(f'\\nBenchmark Comparison:')
print(f'  Random baseline: ~120,000 NLL')
print(f'  LLaMA-65B (orig): ~30,000 NLL')
print(f'  Qwen2.5-32B: {data[\"total_nll\"]:.0f} NLL')
print('='*60)
"
```

#### 2B. Submit DeepSeek LOO CV

```bash
# Verify features exist
ls -lh data/features/centaur_features_deepseek.pth

# Submit job
sbatch scripts/submit_fit_deepseek.sh

# Monitor
tail -f logs/fit_deepseek_<JOB_ID>.out

# Check results
python -c "
import torch
data = torch.load('data/results/loo_cv_results_deepseek.pth')
print('='*60)
print('DEEPSEEK-R1 RESULTS')
print('='*60)
print(f'Average NLL: {data[\"avg_nll\"]:.4f} ± {data[\"std_nll\"]:.4f}')
print(f'Total NLL: {data[\"total_nll\"]:.2f}')
print(f'Accuracy: {data[\"accuracy\"]:.1%}')
print(f'\\nBenchmark Comparison:')
print(f'  Random baseline: ~120,000 NLL')
print(f'  LLaMA-65B (orig): ~30,000 NLL')
print(f'  DeepSeek-R1: {data[\"total_nll\"]:.0f} NLL')
print('='*60)
"
```

---

## Monitoring Commands

### Check Job Status

```bash
# List all your jobs
squeue -u $USER

# Detailed job info
scontrol show job <JOB_ID>

# Check job efficiency
seff <JOB_ID>
```

### Monitor Logs

```bash
# Real-time log monitoring
tail -f logs/extract_qwen25_<JOB_ID>.out
tail -f logs/fit_qwen25_<JOB_ID>.out

# Search for errors
grep -i error logs/*.err

# Check progress
grep "Extracting\|Fold" logs/fit_*.out
```

### GPU Monitoring (Feature Extraction Only)

```bash
# Watch GPU usage
watch -n 5 nvidia-smi

# Check specific GPU
nvidia-smi -i 0 --query-gpu=utilization.gpu,memory.used,memory.total --format=csv
```

---

## Troubleshooting

### Issue 1: Feature Extraction Fails

**Symptoms**: Job exits with error, out of memory

**Debug Steps**:
```bash
# Check error log
cat logs/extract_qwen25_<JOB_ID>.err

# Common issues:
# 1. GPU OOM -> Model already loaded elsewhere
nvidia-smi  # Check if GPU is available

# 2. Model not found -> Check paths
ls -la /home/connectome/connectome1/models/deepseek-r1-distill-qwen-32b

# 3. Adapter not found -> Check LoRA path
ls -la /scratch/connectome/connectome1/ko-centaur/outputs/qwen25-32b-qlora
```

**Solutions**:
- Wait for GPU to be free
- Reduce `--n_samples` for testing
- Check conda environment is activated

### Issue 2: LOO CV Fails

**Symptoms**: Job exits early, features not found

**Debug Steps**:
```bash
# Check if features exist
ls -lh data/features/centaur_features_qwen25.pth

# Check feature file integrity
python -c "
import torch
data = torch.load('data/features/centaur_features_qwen25.pth')
print(f'Features shape: {data[\"features\"].shape}')
print(f'No NaN: {not data[\"features\"].isnan().any()}')
"

# Check error log
cat logs/fit_qwen25_<JOB_ID>.err
```

**Solutions**:
- Re-run feature extraction if corrupted
- Check models.py is in project root
- Verify Python path in script

### Issue 3: Slow Progress

**Symptoms**: Jobs taking longer than expected

**Check**:
```bash
# For feature extraction - check GPU usage
nvidia-smi

# For LOO CV - check CPU/memory
top -u $USER

# Check if other jobs are running
squeue -u $USER
```

**Solutions**:
- Normal for first run (model loading)
- LOO CV is CPU-intensive, 1-2 hours is expected
- Use `--n_samples 10` for testing

---

## Expected Timeline

| Phase | Task | Time | Output Size |
|-------|------|------|-------------|
| 1A | Qwen2.5 Feature Extraction | 1-2 hours | ~5MB |
| 1B | DeepSeek Feature Extraction | 1-2 hours | ~5MB |
| 2A | Qwen2.5 LOO CV | 1-2 hours | ~100KB |
| 2B | DeepSeek LOO CV | 1-2 hours | ~100KB |
| **Total** | **End-to-End** | **4-8 hours** | **~10MB** |

**Optimization**: Run Phase 1A and 1B in parallel (if 2 GPUs available)

---

## Quick Start Script

Save this as `run_pipeline.sh` for easy execution:

```bash
#!/bin/bash
# Ko-CENTaUR Complete Pipeline

echo "Ko-CENTaUR Evaluation Pipeline"
echo "=============================="

# Phase 1: Feature Extraction
echo "Phase 1: Submitting feature extraction jobs..."
JOB1=$(sbatch scripts/submit_extract_qwen25.sh | awk '{print $4}')
JOB2=$(sbatch scripts/submit_extract_deepseek.sh | awk '{print $4}')

echo "Submitted jobs: $JOB1 (Qwen2.5), $JOB2 (DeepSeek)"
echo "Waiting for feature extraction to complete..."

# Wait for jobs to complete
while squeue -j $JOB1,$JOB2 | grep -q $JOB1; do
    echo "  $(date): Jobs still running..."
    sleep 60
done

echo "Feature extraction complete!"

# Phase 2: LOO CV
echo "Phase 2: Submitting LOO CV jobs..."
JOB3=$(sbatch scripts/submit_fit_qwen25.sh | awk '{print $4}')
JOB4=$(sbatch scripts/submit_fit_deepseek.sh | awk '{print $4}')

echo "Submitted jobs: $JOB3 (Qwen2.5), $JOB4 (DeepSeek)"
echo "Waiting for LOO CV to complete..."

# Wait for jobs to complete
while squeue -j $JOB3,$JOB4 | grep -q $JOB3; do
    echo "  $(date): Jobs still running..."
    sleep 60
done

echo "LOO CV complete!"

# Display results
echo "=============================="
echo "FINAL RESULTS"
echo "=============================="

python -c "
import torch

# Qwen2.5 results
qwen_data = torch.load('data/results/loo_cv_results_qwen25.pth')
print(f'Qwen2.5-32B: {qwen_data[\"total_nll\"]:.0f} NLL')

# DeepSeek results
deep_data = torch.load('data/results/loo_cv_results_deepseek.pth')
print(f'DeepSeek-R1: {deep_data[\"total_nll\"]:.0f} NLL')

print(f'\nBenchmark:')
print(f'  Random: ~120,000 NLL')
print(f'  LLaMA-65B: ~30,000 NLL')
"

echo "=============================="
echo "Pipeline complete!"
```

Make executable: `chmod +x run_pipeline.sh`

---

## Post-Execution

### Save Results

```bash
# Copy results to permanent storage
cp data/results/*.pth ~/ko-centaur-results/

# Create summary report
python -c "
import torch
import json

qwen = torch.load('data/results/loo_cv_results_qwen25.pth')
deep = torch.load('data/results/loo_cv_results_deepseek.pth')

summary = {
    'qwen25': {
        'avg_nll': float(qwen['avg_nll']),
        'total_nll': float(qwen['total_nll']),
        'accuracy': float(qwen['accuracy'])
    },
    'deepseek': {
        'avg_nll': float(deep['avg_nll']),
        'total_nll': float(deep['total_nll']),
        'accuracy': float(deep['accuracy'])
    }
}

with open('centaur_results_summary.json', 'w') as f:
    json.dump(summary, f, indent=2)

print('✅ Summary saved to centaur_results_summary.json')
"
```

### Update Documentation

```bash
# Update CLAUDE.md with actual results
# Update README.md status to "✅ Complete"
```

---

## Support

If you encounter issues:
1. Check logs in `logs/` directory
2. Review troubleshooting section above
3. Consult CLAUDE.md for detailed documentation
4. Check original paper methodology in `claudedocs/EVALUATION_METHODOLOGY_ANALYSIS.md`

---

**Ready to Execute!** 🚀
