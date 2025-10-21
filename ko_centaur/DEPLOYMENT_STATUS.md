# Ko-CENTaUR Deployment Status

**Date**: 2025-10-11
**Server**: connectome1@147.47.200.154
**Base Path**: `/scratch/connectome/connectome1/ko-centaur/`

---

## ✅ Deployment Complete

### 1. Code Upload
- ✅ Production scripts uploaded to `/scratch/connectome/connectome1/ko-centaur/scripts/`
- ✅ Evaluation modules: `baselines/`, `evaluation/`
- ✅ Test suite: `tests/`
- ✅ Documentation: `TDD_IMPLEMENTATION_STATUS.md`, `DEPLOYMENT_GUIDE.md`, `scripts/README.md`

### 2. Dataset Preparation
- ✅ Psych-101 dataset uploaded: `data/raw/psych101_train.jsonl` (820MB, 60,092 samples)
- ✅ Directory structure created

### 3. Model Checkpoint
- ✅ Ko-CENTaUR checkpoint located: `models/exaone-psych101-full/checkpoint-19000`
- ✅ BaselineModelManager configured to use this checkpoint

### 4. Environment
- ✅ Conda environment: `ko-centaur` (Python 3.10.18)
- ✅ PyTorch 2.6.0+cu118 with CUDA support
- ✅ Transformers 4.57.0, PEFT 0.17.1

---

## 🔄 Current Status

### Quick Evaluation (Requires User Action)
**Previous Attempts**:
- ✅ Attempt 1: Fixed `checkpoint_path` parameter error
- ❌ Attempt 2: Failed due to SSH disconnection (exit code 255)

**Issue**: SSH connections drop during long-running jobs, killing processes

**Solution**: Use persistent session methods (screen/tmux/nohup) as documented in `RUN_EVALUATION.md`

**Next Action** (User Required):
```bash
# Method 1: Screen (Recommended)
ssh server
screen -S kocentaur_eval
cd /scratch/connectome/connectome1/ko-centaur
source /scratch/connectome/connectome1/miniconda3/bin/activate ko-centaur
python scripts/run_quick_eval.py \
    --checkpoint models/exaone-psych101-full/checkpoint-19000 \
    --dataset data/raw/psych101_train.jsonl \
    --baseline exaone-base \
    --n_samples 10 \
    --output_dir results/quick_eval \
    --device cuda
# Detach: Ctrl+A, then D
# Reattach: screen -r kocentaur_eval
```

**Status**: ⏳ Waiting for user to run with persistent session method

---

## 📋 Next Steps (User Action Required)

### 1. Monitor Quick Evaluation
```bash
# SSH to server
ssh server

# Check if evaluation completed
ls -lh /scratch/connectome/connectome1/ko-centaur/results/quick_eval/

# View results
cat /scratch/connectome/connectome1/ko-centaur/results/quick_eval/quick_eval_*.json
```

### 2. Run Full 50-Sample Evaluation
Once quick eval (10 samples) completes successfully:
```bash
cd /scratch/connectome/connectome1/ko-centaur
source /scratch/connectome/connectome1/miniconda3/bin/activate ko-centaur

python scripts/run_quick_eval.py \
    --checkpoint models/exaone-psych101-full/checkpoint-19000 \
    --dataset data/raw/psych101_train.jsonl \
    --baseline exaone-base \
    --n_samples 50 \
    --output_dir results/quick_eval_50
```

**Expected**: ~5 minutes for 50 samples

### 3. Full Evaluation (100-fold LOO CV)

#### Option A: Local Execution (~4-8 hours)
```bash
python scripts/run_full_eval.py \
    --checkpoint models/exaone-psych101-full/checkpoint-19000 \
    --dataset data/raw/psych101_train.jsonl \
    --baselines exaone-base \
    --n_samples 100 \
    --output_dir results/full_eval
```

#### Option B: SLURM Parallel (~30-60 minutes)

**Step 1**: Extract and cache features (one-time, ~2-4 hours)
```bash
python scripts/run_full_eval.py \
    --checkpoint models/exaone-psych101-full/checkpoint-19000 \
    --dataset data/raw/psych101_train.jsonl \
    --baselines exaone-base \
    --n_samples 100 \
    --fold_id 0 \
    --output_dir results/full_eval
```

**Step 2**: Generate SLURM array job script
```bash
python scripts/generate_slurm_cv.py \
    --job_name kocentaur_loo_cv \
    --n_folds 100 \
    --checkpoint models/exaone-psych101-full/checkpoint-19000 \
    --dataset data/raw/psych101_train.jsonl \
    --baselines exaone-base \
    --output_dir results/full_eval \
    --output slurm_cv_job.sh
```

**Step 3**: Submit SLURM job
```bash
sbatch slurm_cv_job.sh
```

**Step 4**: Monitor progress
```bash
# Check job status
squeue -u connectome1

# View logs
tail -f results/full_eval/logs/slurm-*.out

# Count completed folds
ls results/full_eval/ko-centaur/fold_*.pth | wc -l
```

**Step 5**: Collect results
```bash
python scripts/collect_cv_results.py \
    --input_dir results/full_eval \
    --n_folds 100 \
    --output results/full_eval/aggregated_results.pth
```

### 4. Generate Statistical Reports
```bash
python scripts/generate_reports.py \
    --results_file results/full_eval/aggregated_results.pth \
    --output_dir results/reports
```

**Outputs**:
- `results/reports/statistical_report.txt` (plain text)
- `results/reports/statistical_report.md` (markdown)
- `results/reports/statistical_report.json` (structured data)

---

## 🔧 Troubleshooting

### Issue: Import Errors
**Solution**: Modules are at `/scratch/connectome/connectome1/ko-centaur/baselines` and `evaluation` (root level, not under `ko_centaur/`)

### Issue: OOM Errors
**Solution**: Enable 4-bit quantization by editing `baselines/load_baselines.py`:
```python
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    load_in_4bit=True,  # Add this
    ...
)
```

### Issue: SLURM Job Fails
**Check logs**:
```bash
cat results/full_eval/logs/slurm-JOBID_FOLDID.err
```

**Common causes**:
1. Features not cached → Run fold 0 first
2. GPU OOM → Reduce batch size or enable quantization
3. Missing dependencies → Check `pip list` in environment

### Issue: Missing Folds
**Find missing**:
```bash
python -c "
import os
folds = set(range(100))
completed = {int(f.split('_')[1].split('.')[0])
             for f in os.listdir('results/full_eval/ko-centaur')
             if f.startswith('fold_')}
missing = sorted(folds - completed)
print(f'Missing folds: {missing}')
"
```

**Rerun specific folds**:
```bash
sbatch --array=5,12,47 slurm_cv_job.sh
```

---

## 📊 Expected Performance

| Task | Time | GPU |
|------|------|-----|
| Quick Eval (10 samples) | 2-3 min | 1x RTX |
| Quick Eval (50 samples) | 5-10 min | 1x RTX |
| Feature Extraction (100 samples) | 2-4 hours | 1x RTX |
| Single LOO fold | 2-3 min | 1x RTX |
| All 100 folds (local) | 4-8 hours | 1x RTX |
| All 100 folds (SLURM) | 30-60 min | 100x RTX |

---

## 📁 Server Directory Structure

```
/scratch/connectome/connectome1/ko-centaur/
├── baselines/               ← Baseline model loading
│   ├── load_baselines.py
│   ├── download_models.py
│   └── download_verification.py
├── evaluation/              ← Evaluation pipeline
│   ├── extract_features.py
│   ├── quick_eval.py
│   ├── compare_statistical.py
│   └── cross_validation.py
├── scripts/                 ← Production scripts
│   ├── run_quick_eval.py
│   ├── run_full_eval.py
│   ├── generate_slurm_cv.py
│   ├── collect_cv_results.py
│   ├── generate_reports.py
│   └── README.md
├── tests/                   ← Test suite (142 tests)
├── data/
│   └── raw/
│       └── psych101_train.jsonl  ← Dataset (820MB)
├── models/
│   └── exaone-psych101-full/
│       └── checkpoint-19000/     ← Ko-CENTaUR weights
└── results/                 ← Evaluation outputs
    ├── quick_eval/
    ├── full_eval/
    └── reports/
```

---

## ✅ Deployment Checklist

- [x] Code uploaded to server
- [x] Dataset uploaded (820MB, 60,092 samples)
- [x] Checkpoint configured (checkpoint-19000)
- [x] Environment verified (Python 3.10.18, PyTorch 2.6.0, CUDA)
- [x] Dependencies installed (transformers, peft, scipy, sklearn)
- [x] Directory structure created
- [x] Scripts configured for server paths
- [x] Checkpoint path error fixed in baselines/load_baselines.py
- [x] Module import errors resolved (baselines/, evaluation/ at root)
- [x] Persistent session guide created (RUN_EVALUATION.md)
- [ ] Quick evaluation completed (waiting for user with screen/tmux)
- [ ] Full evaluation executed
- [ ] Statistical reports generated

---

## 📚 Documentation

- **Implementation Guide**: `TDD_IMPLEMENTATION_STATUS.md`
- **Deployment Guide**: `DEPLOYMENT_GUIDE.md`
- **Scripts Usage**: `scripts/README.md`
- **This Status**: `DEPLOYMENT_STATUS.md`

---

**Last Updated**: 2025-10-11 15:06 KST
**Quick Eval Status**: Running in background (models loading)
