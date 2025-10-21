# Ko-CENTaUR Evaluation Execution Guide

**Server**: connectome1@147.47.200.154
**Status**: Code deployed, ready for execution

---

## 🚨 Important: Run Jobs with nohup or screen

SSH sessions that disconnect will kill running processes. Use one of these methods:

### Method 1: nohup (Recommended for simple jobs)

```bash
ssh server
cd /scratch/connectome/connectome1/ko-centaur
source /scratch/connectome/connectome1/miniconda3/bin/activate ko-centaur

# Run with nohup (survives SSH disconnection)
nohup python scripts/run_quick_eval.py \
    --checkpoint models/exaone-psych101-full/checkpoint-19000 \
    --dataset data/raw/psych101_train.jsonl \
    --baseline exaone-base \
    --n_samples 10 \
    --output_dir results/quick_eval \
    --device cuda \
    > quick_eval.log 2>&1 &

# Get process ID
echo $!

# Monitor progress
tail -f quick_eval.log

# Or check status later
ps aux | grep run_quick_eval
ls -lh results/quick_eval/
```

### Method 2: screen (Recommended for interactive monitoring)

```bash
ssh server

# Start screen session
screen -S kocentaur_eval

# Inside screen session
cd /scratch/connectome/connectome1/ko-centaur
source /scratch/connectome/connectome1/miniconda3/bin/activate ko-centaur

python scripts/run_quick_eval.py \
    --checkpoint models/exaone-psych101-full/checkpoint-19000 \
    --dataset data/raw/psych101_train.jsonl \
    --baseline exaone-base \
    --n_samples 10 \
    --output_dir results/quick_eval \
    --device cuda

# Detach from screen: Ctrl+A, then D
# Reattach later: screen -r kocentaur_eval
# List sessions: screen -ls
# Kill session: screen -X -S kocentaur_eval quit
```

### Method 3: tmux (Alternative to screen)

```bash
ssh server

# Start tmux session
tmux new -s kocentaur_eval

# Inside tmux
cd /scratch/connectome/connectome1/ko-centaur
source /scratch/connectome/connectome1/miniconda3/bin/activate ko-centaur

python scripts/run_quick_eval.py \
    --checkpoint models/exaone-psych101-full/checkpoint-19000 \
    --dataset data/raw/psych101_train.jsonl \
    --baseline exaone-base \
    --n_samples 10 \
    --output_dir results/quick_eval \
    --device cuda

# Detach from tmux: Ctrl+B, then D
# Reattach later: tmux attach -t kocentaur_eval
# List sessions: tmux ls
```

---

## 📋 Evaluation Workflow

### Step 1: Quick Validation (10 samples, ~5 min)

```bash
# With nohup
nohup python scripts/run_quick_eval.py \
    --checkpoint models/exaone-psych101-full/checkpoint-19000 \
    --dataset data/raw/psych101_train.jsonl \
    --baseline exaone-base \
    --n_samples 10 \
    --output_dir results/quick_eval_10 \
    --device cuda \
    > quick_eval_10.log 2>&1 &

# Monitor
tail -f quick_eval_10.log

# Check results when done
cat results/quick_eval_10/quick_eval_*.json | python -m json.tool
```

### Step 2: Full Quick Eval (50 samples, ~10 min)

```bash
nohup python scripts/run_quick_eval.py \
    --checkpoint models/exaone-psych101-full/checkpoint-19000 \
    --dataset data/raw/psych101_train.jsonl \
    --baseline exaone-base \
    --n_samples 50 \
    --output_dir results/quick_eval_50 \
    --device cuda \
    > quick_eval_50.log 2>&1 &
```

### Step 3: Full Evaluation - Local (100 folds, 4-8 hours)

**Only if you want local sequential execution:**

```bash
# Use screen/tmux for long jobs!
screen -S kocentaur_full

cd /scratch/connectome/connectome1/ko-centaur
source /scratch/connectome/connectome1/miniconda3/bin/activate ko-centaur

python scripts/run_full_eval.py \
    --checkpoint models/exaone-psych101-full/checkpoint-19000 \
    --dataset data/raw/psych101_train.jsonl \
    --baselines exaone-base \
    --n_samples 100 \
    --output_dir results/full_eval_local \
    --device cuda

# Detach: Ctrl+A D (screen) or Ctrl+B D (tmux)
```

### Step 4: Full Evaluation - SLURM Parallel (Recommended, 30-60 min)

#### 4a. Extract and Cache Features (one-time)

```bash
# Use screen for this 2-4 hour job
screen -S feature_extract

cd /scratch/connectome/connectome1/ko-centaur
source /scratch/connectome/connectome1/miniconda3/bin/activate ko-centaur

python scripts/run_full_eval.py \
    --checkpoint models/exaone-psych101-full/checkpoint-19000 \
    --dataset data/raw/psych101_train.jsonl \
    --baselines exaone-base \
    --n_samples 100 \
    --fold_id 0 \
    --output_dir results/full_eval_slurm \
    --device cuda

# Detach: Ctrl+A D
```

**Monitor feature extraction:**
```bash
screen -r feature_extract  # Reattach to check progress
ls -lh results/full_eval_slurm/features/  # Check cached features
```

#### 4b. Generate SLURM Script

```bash
cd /scratch/connectome/connectome1/ko-centaur
source /scratch/connectome/connectome1/miniconda3/bin/activate ko-centaur

python scripts/generate_slurm_cv.py \
    --job_name kocentaur_loo_cv \
    --n_folds 100 \
    --time_limit "02:00:00" \
    --memory "32GB" \
    --gpus_per_task 1 \
    --checkpoint models/exaone-psych101-full/checkpoint-19000 \
    --dataset data/raw/psych101_train.jsonl \
    --baselines exaone-base \
    --output_dir results/full_eval_slurm \
    --partition debug \
    --output slurm_kocentaur_cv.sh
```

#### 4c. Submit SLURM Job

```bash
# Submit array job (100 parallel folds)
sbatch slurm_kocentaur_cv.sh

# Monitor job status
watch -n 5 'squeue -u connectome1 | tail -n 20'

# Check specific fold logs
tail -f results/full_eval_slurm/logs/slurm-*.out
cat results/full_eval_slurm/logs/slurm-*_0.err  # Check fold 0 errors

# Count completed folds
ls results/full_eval_slurm/ko-centaur/fold_*.pth | wc -l

# Cancel if needed
scancel --name=kocentaur_loo_cv
```

#### 4d. Collect Results (after SLURM completes)

```bash
cd /scratch/connectome/connectome1/ko-centaur
source /scratch/connectome/connectome1/miniconda3/bin/activate ko-centaur

python scripts/collect_cv_results.py \
    --input_dir results/full_eval_slurm \
    --n_folds 100 \
    --output results/full_eval_slurm/aggregated_results.pth
```

### Step 5: Generate Reports

```bash
python scripts/generate_reports.py \
    --results_file results/full_eval_slurm/aggregated_results.pth \
    --output_dir results/reports
```

**View reports:**
```bash
# Plain text
cat results/reports/statistical_report.txt

# Markdown
cat results/reports/statistical_report.md

# JSON (structured)
python -m json.tool results/reports/statistical_report.json | less
```

---

## 🔍 Monitoring and Debugging

### Check Running Processes
```bash
# All Python processes
ps aux | grep python | grep connectome1

# Specific script
ps aux | grep run_quick_eval

# GPU usage
nvidia-smi
watch -n 1 nvidia-smi  # Real-time monitoring
```

### Check Disk Space
```bash
df -h /scratch/connectome/connectome1/ko-centaur/
du -sh results/
```

### Check Logs
```bash
# Quick eval log
tail -f quick_eval.log

# SLURM logs
tail -f results/full_eval_slurm/logs/slurm-*.out

# Error logs
cat results/full_eval_slurm/logs/slurm-*_*.err | less
```

### Verify Checkpoint
```bash
ls -lh models/exaone-psych101-full/checkpoint-19000/
python -c "from transformers import AutoModel; print('Checkpoint valid')"
```

---

## ⚠️ Common Issues

### Issue: SSH Connection Drops
**Solution**: Always use nohup, screen, or tmux for long-running jobs

### Issue: "No module named 'baselines'"
**Solution**: Check that baselines/ and evaluation/ are at:
```bash
ls /scratch/connectome/connectome1/ko-centaur/baselines/
ls /scratch/connectome/connectome1/ko-centaur/evaluation/
```

### Issue: "No labels found in dataset"
**Note**: This is expected for Psych-101 dataset. The evaluation pipeline uses task_description fields.

### Issue: GPU Out of Memory
**Solutions**:
1. Enable 4-bit quantization in baselines/load_baselines.py
2. Reduce batch size
3. Use smaller model or fewer samples

### Issue: Process Killed
**Check**:
```bash
dmesg | tail -50  # Look for OOM killer
cat /var/log/messages | tail -50  # System logs
```

---

## 📊 Expected Output Structure

```
results/
├── quick_eval_10/
│   └── quick_eval_20251011_150538.json
├── quick_eval_50/
│   └── quick_eval_20251011_151234.json
├── full_eval_slurm/
│   ├── features/
│   │   ├── ko_centaur_features.pth
│   │   └── exaone-base_features.pth
│   ├── ko-centaur/
│   │   ├── fold_0.pth ... fold_99.pth
│   │   └── aggregated_results.pth
│   ├── exaone-base/
│   │   └── [same structure]
│   ├── logs/
│   │   ├── slurm-12345_0.out ... slurm-12345_99.out
│   │   └── slurm-12345_0.err ... slurm-12345_99.err
│   ├── aggregated_results.pth
│   └── aggregated_results.json
└── reports/
    ├── statistical_report.txt
    ├── statistical_report.md
    └── statistical_report.json
```

---

## 🚀 Quick Start Commands

**For quick validation (right now):**
```bash
ssh server
screen -S quickeval
cd /scratch/connectome/connectome1/ko-centaur
source /scratch/connectome/connectome1/miniconda3/bin/activate ko-centaur
python scripts/run_quick_eval.py \
    --checkpoint models/exaone-psych101-full/checkpoint-19000 \
    --dataset data/raw/psych101_train.jsonl \
    --baseline exaone-base \
    --n_samples 10 \
    --output_dir results/quick_eval \
    --device cuda
# Detach: Ctrl+A D
# Check later: screen -r quickeval
```

**For full evaluation (recommended):**
1. Run feature extraction in screen (2-4 hours)
2. Generate SLURM script
3. Submit sbatch (30-60 minutes for 100 folds)
4. Collect results
5. Generate reports

---

**Documentation**:
- Deployment Status: `DEPLOYMENT_STATUS.md`
- Implementation Guide: `TDD_IMPLEMENTATION_STATUS.md`
- Scripts Reference: `scripts/README.md`
