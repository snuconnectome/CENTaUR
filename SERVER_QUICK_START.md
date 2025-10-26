# Ko-CENTaUR Server Quick Start

**Copy and paste these commands to run the complete pipeline**

---

## 1️⃣ Connect to Server & Prepare

```bash
# Connect to server
ssh server

# Navigate to project
cd /scratch/connectome/connectome1/ko-centaur

# Pull latest changes
git pull origin feature/ko-centaur-llm-strategy

# Activate environment
source ~/.bashrc
conda activate centaur

# Verify everything is ready
python -c "import torch, transformers, peft; print('✅ Packages OK')"
ls -la scripts/extract_centaur_features.py
ls -la scripts/fit_centaur_loo_cv.py

# Create directories
mkdir -p data/features data/results logs
```

---

## 2️⃣ Run Feature Extraction (GPU, ~1-2h each)

```bash
# Submit Qwen2.5 feature extraction
sbatch scripts/submit_extract_qwen25.sh

# Submit DeepSeek feature extraction
sbatch scripts/submit_extract_deepseek.sh

# Check job status
squeue -u $USER

# Monitor progress (Ctrl+C to exit)
tail -f logs/extract_qwen25_*.out
```

---

## 3️⃣ Verify Feature Extraction

```bash
# After jobs complete, verify outputs
python -c "
import torch
qwen = torch.load('data/features/centaur_features_qwen25.pth')
deep = torch.load('data/features/centaur_features_deepseek.pth')
print(f'✅ Qwen2.5: {qwen[\"features\"].shape}')
print(f'✅ DeepSeek: {deep[\"features\"].shape}')
"
```

Expected output:
```
✅ Qwen2.5: torch.Size([100, 3072])
✅ DeepSeek: torch.Size([100, 3072])
```

---

## 4️⃣ Run LOO Cross-Validation (CPU, ~1-2h each)

```bash
# Submit Qwen2.5 LOO CV
sbatch scripts/submit_fit_qwen25.sh

# Submit DeepSeek LOO CV
sbatch scripts/submit_fit_deepseek.sh

# Monitor progress
tail -f logs/fit_qwen25_*.out
```

---

## 5️⃣ View Results

```bash
# After jobs complete, view results
python -c "
import torch

print('='*70)
print('Ko-CENTaUR RESULTS')
print('='*70)

# Qwen2.5
qwen = torch.load('data/results/loo_cv_results_qwen25.pth')
print(f'\\nQwen2.5-32B-QLoRA:')
print(f'  Average NLL: {qwen[\"avg_nll\"]:.4f} ± {qwen[\"std_nll\"]:.4f}')
print(f'  Total NLL:   {qwen[\"total_nll\"]:.0f}')
print(f'  Accuracy:    {qwen[\"accuracy\"]:.1%}')

# DeepSeek
deep = torch.load('data/results/loo_cv_results_deepseek.pth')
print(f'\\nDeepSeek-R1-32B-QLoRA:')
print(f'  Average NLL: {deep[\"avg_nll\"]:.4f} ± {deep[\"std_nll\"]:.4f}')
print(f'  Total NLL:   {deep[\"total_nll\"]:.0f}')
print(f'  Accuracy:    {deep[\"accuracy\"]:.1%}')

print(f'\\nBenchmark Comparison:')
print(f'  Random Baseline:  ~120,000 NLL')
print(f'  LLaMA-65B (orig): ~30,000 NLL')
print(f'  Qwen2.5-32B:      {qwen[\"total_nll\"]:.0f} NLL')
print(f'  DeepSeek-R1:      {deep[\"total_nll\"]:.0f} NLL')

print('='*70)
"
```

---

## 🔍 Monitoring Commands

```bash
# Check all your jobs
squeue -u $USER

# Watch GPU usage
watch -n 5 nvidia-smi

# Check logs for errors
grep -i error logs/*.err

# Monitor progress
grep "Fold\|Extracting" logs/fit_*.out | tail -10
```

---

## ⚠️ Troubleshooting

**If feature extraction fails**:
```bash
# Check GPU availability
nvidia-smi

# Check error log
cat logs/extract_qwen25_*.err

# Try with fewer samples first
python scripts/extract_centaur_features.py --model qwen25 --n_samples 10
```

**If LOO CV fails**:
```bash
# Verify features exist
ls -lh data/features/centaur_features_qwen25.pth

# Check error log
cat logs/fit_qwen25_*.err

# Manually test
python scripts/fit_centaur_loo_cv.py --model qwen25
```

---

## 📊 Expected Timeline

| Task | Time | GPU/CPU |
|------|------|---------|
| Qwen2.5 Feature Extraction | 1-2h | GPU |
| DeepSeek Feature Extraction | 1-2h | GPU |
| Qwen2.5 LOO CV | 1-2h | CPU |
| DeepSeek LOO CV | 1-2h | CPU |
| **Total** | **4-8h** | Both |

**Tip**: Run both feature extractions in parallel if 2 GPUs available!

---

## 📁 Output Files

After completion, you'll have:

```
data/
├── features/
│   ├── centaur_features_qwen25.pth     (~5MB)
│   └── centaur_features_deepseek.pth   (~5MB)
└── results/
    ├── loo_cv_results_qwen25.pth       (~100KB)
    └── loo_cv_results_deepseek.pth     (~100KB)
```

---

## ✅ Success Criteria

Your pipeline is successful if:

1. ✅ All 4 jobs complete without errors
2. ✅ Features shape: `(100, 3072)`
3. ✅ NLL values are reasonable (not NaN/Inf)
4. ✅ Both models have results in `data/results/`

---

## 🎯 Next Steps After Completion

1. Save results to permanent location
2. Update documentation with actual NLL values
3. Compare against original LLaMA-65B baseline (~30,000)
4. Analyze which model performs better on cognitive modeling

---

**Ready to start! 🚀**

For detailed guide: See `claudedocs/SERVER_DEPLOYMENT_GUIDE.md`
