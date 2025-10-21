# Ko-CENTaUR Production Evaluation Scripts

Production-ready scripts for running Ko-CENTaUR evaluation pipeline on connectome server.

**All scripts configured for**: `/scratch/connectome/connectome1/ko-centaur/`

---

## 📋 Quick Start

### 1. Quick Evaluation (5 minutes, 50 samples)

```bash
# SSH to server
ssh server
cd /scratch/connectome/connectome1/ko-centaur

# Run quick evaluation
python ko_centaur/scripts/run_quick_eval.py \
    --checkpoint models/ko_centaur_checkpoint \
    --dataset data/raw/psych101_train.jsonl \
    --baseline exaone-base \
    --n_samples 50 \
    --output_dir results/quick_eval
```

### 2. Full Evaluation - Local (4-8 hours)

```bash
# Extract features once (slow, ~2-4 hours)
python ko_centaur/scripts/run_full_eval.py \
    --checkpoint models/ko_centaur_checkpoint \
    --dataset data/raw/psych101_train.jsonl \
    --baselines exaone-base llama-3.2-3b \
    --n_samples 100 \
    --output_dir results/full_eval

# Features are cached, subsequent runs use --skip_features
```

### 3. Full Evaluation - SLURM Parallel (30-60 minutes)

**Step 1: Extract features**
```bash
# First run extracts and caches features
python ko_centaur/scripts/run_full_eval.py \
    --checkpoint models/ko_centaur_checkpoint \
    --dataset data/raw/psych101_train.jsonl \
    --baselines exaone-base \
    --n_samples 100 \
    --fold_id 0 \
    --output_dir results/full_eval
```

**Step 2: Generate SLURM script**
```bash
python ko_centaur/scripts/generate_slurm_cv.py \
    --job_name kocentaur_loo_cv \
    --n_folds 100 \
    --checkpoint models/ko_centaur_checkpoint \
    --dataset data/raw/psych101_train.jsonl \
    --baselines exaone-base llama-3.2-3b \
    --output_dir results/full_eval \
    --output slurm_cv_job.sh
```

**Step 3: Submit SLURM array job**
```bash
sbatch slurm_cv_job.sh
```

**Step 4: Monitor progress**
```bash
# Check job status
squeue -u connectome1

# View logs
tail -f results/full_eval/logs/slurm-*.out

# Check completion
ls results/full_eval/ko-centaur/fold_*.pth | wc -l
```

**Step 5: Collect results**
```bash
python ko_centaur/scripts/collect_cv_results.py \
    --input_dir results/full_eval \
    --n_folds 100 \
    --output results/full_eval/aggregated_results.pth
```

**Step 6: Generate reports**
```bash
python ko_centaur/scripts/generate_reports.py \
    --results_file results/full_eval/aggregated_results.pth \
    --output_dir results/reports
```

---

## 📄 Script Details

### `run_quick_eval.py`

**Purpose**: Fast 50-sample validation of evaluation pipeline

**Usage**:
```bash
python scripts/run_quick_eval.py \
    --checkpoint PATH_TO_CHECKPOINT \
    --dataset PATH_TO_DATASET \
    --baseline MODEL_NAME \
    --n_samples 50 \
    --output_dir OUTPUT_DIR
```

**Parameters**:
- `--checkpoint`: Ko-CENTaUR checkpoint directory (LoRA adapters)
- `--dataset`: Psych-101 JSONL dataset path
- `--baseline`: Baseline model name (exaone-base, llama-3.2-3b, etc.)
- `--n_samples`: Number of samples (default: 50)
- `--output_dir`: Output directory for results
- `--device`: Device (cuda or cpu, default: cuda)

**Output**:
- `quick_eval_YYYYMMDD_HHMMSS.json`: Evaluation results with accuracies and comparison

**Expected Runtime**: ~5 minutes

---

### `run_full_eval.py`

**Purpose**: Complete 100-fold LOO cross-validation with nested hyperparameter tuning

**Usage**:
```bash
# All folds (local execution)
python scripts/run_full_eval.py \
    --checkpoint PATH_TO_CHECKPOINT \
    --dataset PATH_TO_DATASET \
    --baselines MODEL1 MODEL2 \
    --n_samples 100 \
    --output_dir OUTPUT_DIR

# Single fold (for SLURM array jobs)
python scripts/run_full_eval.py \
    --checkpoint PATH_TO_CHECKPOINT \
    --dataset PATH_TO_DATASET \
    --baselines MODEL1 \
    --n_samples 100 \
    --fold_id $SLURM_ARRAY_TASK_ID \
    --output_dir OUTPUT_DIR \
    --skip_features  # Use cached features
```

**Parameters**:
- `--checkpoint`: Ko-CENTaUR checkpoint directory
- `--dataset`: Psych-101 JSONL dataset path
- `--baselines`: Space-separated baseline model names
- `--n_samples`: Number of samples (default: 100)
- `--fold_id`: Specific fold to run (default: None = all folds)
- `--output_dir`: Output directory
- `--skip_features`: Skip feature extraction (use cached)
- `--device`: Device (cuda or cpu)

**Output**:
- `features/MODEL_NAME_features.pth`: Cached features
- `MODEL_NAME/fold_N.pth`: Results for each fold
- `MODEL_NAME/aggregated_results.pth`: Aggregated statistics

**Expected Runtime**:
- Feature extraction: 2-4 hours (one-time)
- Per fold: 2-3 minutes
- All folds locally: 4-8 hours
- All folds via SLURM (parallel): 30-60 minutes

---

### `generate_slurm_cv.py`

**Purpose**: Generate SLURM array job script for parallel execution

**Usage**:
```bash
python scripts/generate_slurm_cv.py \
    --job_name JOB_NAME \
    --n_folds 100 \
    --time_limit "04:00:00" \
    --memory "32GB" \
    --gpus_per_task 1 \
    --checkpoint PATH_TO_CHECKPOINT \
    --dataset PATH_TO_DATASET \
    --baselines MODEL1 MODEL2 \
    --output_dir OUTPUT_DIR \
    --output slurm_cv_job.sh
```

**Parameters**:
- `--job_name`: SLURM job name (default: kocentaur_loo_cv)
- `--n_folds`: Number of folds (default: 100)
- `--time_limit`: Time limit per fold (default: 04:00:00)
- `--memory`: Memory per task (default: 32GB)
- `--gpus_per_task`: GPUs per fold (default: 1)
- `--checkpoint`: Ko-CENTaUR checkpoint path
- `--dataset`: Dataset path
- `--baselines`: Baseline models
- `--output_dir`: Output directory
- `--partition`: SLURM partition (default: debug)
- `--output`: Output script filename

**Output**:
- `.sh` file ready for `sbatch` submission

---

### `collect_cv_results.py`

**Purpose**: Aggregate fold results into final evaluation metrics

**Usage**:
```bash
python scripts/collect_cv_results.py \
    --input_dir RESULTS_DIR \
    --n_folds 100 \
    --output AGGREGATED_RESULTS.pth \
    --models MODEL1 MODEL2  # Optional, auto-detects if omitted
```

**Parameters**:
- `--input_dir`: Directory containing fold results
- `--n_folds`: Expected number of folds (default: 100)
- `--output`: Output path (default: input_dir/aggregated_results.pth)
- `--models`: Model names (default: auto-detect from directories)

**Output**:
- `aggregated_results.pth`: PyTorch file with all statistics
- `aggregated_results.json`: JSON version for easy inspection

**Output Format**:
```python
{
    'model_results': {
        'model_name': {
            'mean_accuracy': float,
            'std_accuracy': float,
            'accuracies': list,
            'log_likelihoods': list,
            ...
        }
    },
    'pairwise_comparisons': {
        'model_a vs model_b': {
            't_statistic': float,
            'p_value': float,
            'cohens_d': float,
            ...
        }
    },
    'best_model': str,
    'ranking': list
}
```

---

### `generate_reports.py`

**Purpose**: Create comprehensive statistical analysis reports

**Usage**:
```bash
python scripts/generate_reports.py \
    --results_file AGGREGATED_RESULTS.pth \
    --output_dir REPORTS_DIR \
    --alpha 0.05
```

**Parameters**:
- `--results_file`: Path to aggregated results (.pth)
- `--output_dir`: Output directory for reports
- `--alpha`: Significance level (default: 0.05)

**Output**:
- `statistical_report.txt`: Plain text comprehensive report
- `statistical_report.md`: Markdown formatted report
- `statistical_report.json`: JSON structured report

**Report Sections**:
1. Model Performance Summary
2. Pairwise Statistical Comparisons
3. Model Ranking with Confidence Intervals
4. Detailed Statistics per Model

---

## 🔄 Complete Workflow

### Workflow 1: Quick Validation

```bash
# 1. Quick evaluation (5 minutes)
python scripts/run_quick_eval.py \
    --checkpoint models/ko_centaur_checkpoint \
    --dataset data/raw/psych101_train.jsonl \
    --baseline exaone-base \
    --n_samples 50

# 2. Inspect results
cat results/quick_eval/quick_eval_*.json
```

### Workflow 2: Full Evaluation (Local)

```bash
# 1. Run full evaluation (4-8 hours)
python scripts/run_full_eval.py \
    --checkpoint models/ko_centaur_checkpoint \
    --dataset data/raw/psych101_train.jsonl \
    --baselines exaone-base llama-3.2-3b \
    --n_samples 100

# 2. Collect results
python scripts/collect_cv_results.py \
    --input_dir results/full_eval \
    --n_folds 100

# 3. Generate reports
python scripts/generate_reports.py \
    --results_file results/full_eval/aggregated_results.pth \
    --output_dir results/reports

# 4. View reports
cat results/reports/statistical_report.txt
```

### Workflow 3: Full Evaluation (SLURM Parallel)

```bash
# 1. Extract features (one-time, ~2-4 hours)
python scripts/run_full_eval.py \
    --checkpoint models/ko_centaur_checkpoint \
    --dataset data/raw/psych101_train.jsonl \
    --baselines exaone-base llama-3.2-3b \
    --n_samples 100 \
    --fold_id 0  # Just fold 0 to cache features

# 2. Generate SLURM script
python scripts/generate_slurm_cv.py \
    --n_folds 100 \
    --checkpoint models/ko_centaur_checkpoint \
    --dataset data/raw/psych101_train.jsonl \
    --baselines exaone-base llama-3.2-3b \
    --output slurm_cv_job.sh

# 3. Submit job
sbatch slurm_cv_job.sh

# 4. Monitor (wait for completion)
watch -n 10 'squeue -u connectome1 | tail -n 20'

# 5. Collect results
python scripts/collect_cv_results.py \
    --input_dir results/full_eval \
    --n_folds 100

# 6. Generate reports
python scripts/generate_reports.py \
    --results_file results/full_eval/aggregated_results.pth \
    --output_dir results/reports
```

---

## 🛠 Troubleshooting

### Issue: Feature Extraction OOM

**Solution**:
```bash
# Enable 4-bit quantization
# Edit run_full_eval.py, add to BaselineModelManager:
BaselineModelManager(model_name, load_in_4bit=True)
```

### Issue: SLURM Job Fails

**Check logs**:
```bash
# View error logs
cat results/full_eval/logs/slurm-*_*.err

# Check specific fold
cat results/full_eval/logs/slurm-JOBID_FOLDID.out
```

**Common causes**:
- Features not cached: Run fold 0 first to cache features
- GPU out of memory: Reduce batch size or enable quantization
- Missing dependencies: Check `pip install peft bitsandbytes`

### Issue: Missing Folds

**Check completion**:
```bash
# Count completed folds
ls results/full_eval/ko-centaur/fold_*.pth | wc -l

# Find missing folds
python -c "
import os
folds = set(range(100))
completed = {int(f.split('_')[1].split('.')[0]) for f in os.listdir('results/full_eval/ko-centaur') if f.startswith('fold_')}
missing = sorted(folds - completed)
print(f'Missing: {missing}')
"
```

**Rerun missing folds**:
```bash
# Resubmit specific folds
sbatch --array=5,12,47 slurm_cv_job.sh
```

---

## 📊 Expected Output Structure

```
results/
├── quick_eval/
│   └── quick_eval_20241011_223045.json
├── full_eval/
│   ├── features/
│   │   ├── ko_centaur_features.pth
│   │   ├── exaone-base_features.pth
│   │   └── llama-3.2-3b_features.pth
│   ├── ko-centaur/
│   │   ├── fold_0.pth
│   │   ├── fold_1.pth
│   │   ├── ...
│   │   ├── fold_99.pth
│   │   └── aggregated_results.pth
│   ├── exaone-base/
│   │   └── [same structure]
│   ├── logs/
│   │   ├── slurm-12345_0.out
│   │   └── slurm-12345_0.err
│   ├── aggregated_results.pth
│   └── aggregated_results.json
└── reports/
    ├── statistical_report.txt
    ├── statistical_report.md
    └── statistical_report.json
```

---

## ⏱ Performance Benchmarks

| Task | Local (1x RTX 3090) | SLURM (100x RTX) |
|------|---------------------|------------------|
| Quick Eval (50 samples) | 5 min | N/A |
| Feature Extraction (100 samples) | 2-4 hours | 2-4 hours (one-time) |
| Single Fold CV | 2-3 min | 2-3 min |
| All Folds (100) | 4-8 hours | 30-60 min |
| Report Generation | 1 min | 1 min |
| **Total (first run)** | **6-12 hours** | **3-5 hours** |
| **Total (cached features)** | **4-8 hours** | **30-60 min** |

---

## 📚 References

- ICLR 2023 CENTaUR paper: Binz & Schulz
- TDD Implementation: `../TDD_IMPLEMENTATION_STATUS.md`
- Deployment Guide: `../DEPLOYMENT_GUIDE.md`
- Connectome Server Guide: `../../docs/SLURM_GUIDE.md`
