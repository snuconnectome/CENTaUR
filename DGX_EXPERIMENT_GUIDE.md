# DGX-Spark CENTaUR Experiments Guide

## Setup Complete ✅

- **Branch**: `feature/dgx-experiments`
- **Location**: `dgx-spark:~/git/CENTaUR/`
- **Tmux Session**: `git` (already in CENTaUR directory)

## Quick Start

### 1. Connect to tmux session
```bash
ssh dgx-spark -t "tmux a -t git"
```

### 2. Run quick test (10 samples)
```bash
./run_centaur_test.sh
```

This will:
- Create Python virtual environment (`dgx-venv`)
- Install dependencies (torch, transformers, etc.)
- Run feature extraction test with 10 samples
- Takes ~5-10 minutes depending on GPU availability

### 3. Check GPU availability
```bash
nvidia-smi
```

### 4. Run full experiment
```bash
# Activate environment
source dgx-venv/bin/activate

# Feature extraction (GPU required)
python scripts/extract_centaur_features.py --model qwen25 --n_samples 100

# LOO Cross-Validation (CPU only)
python scripts/fit_centaur_loo_cv.py --model qwen25
```

## Experiment Options

### Quick Test (10 samples)
```bash
python scripts/extract_centaur_features.py --model qwen25 --n_samples 10 --is_local
```

### Medium Scale (100 samples)
```bash
python scripts/extract_centaur_features.py --model qwen25 --n_samples 100
```

### Full Dataset (~1000 samples)
```bash
python scripts/extract_centaur_features.py --model qwen25
```

### Use DeepSeek model instead
```bash
python scripts/extract_centaur_features.py --model deepseek --n_samples 10
```

## Models Available

- **qwen25**: Qwen2.5-32B-Instruct (fine-tuned)
- **deepseek**: DeepSeek-R1-Distill-Qwen-32B (fine-tuned)

## Expected Output

Feature extraction creates:
- `features_<model>_<timestamp>.pt` - Extracted hidden states
- Progress logs in terminal

LOO CV creates:
- Model checkpoints for each fold
- Performance metrics (NLL scores)

## Monitoring

### Check progress in tmux
```bash
Ctrl+B, D  # Detach from session
ssh dgx-spark -t "tmux a -t git"  # Reattach later
```

### Monitor GPU usage
```bash
watch -n 1 nvidia-smi
```

### Check results
```bash
ls -lh features_*.pt
cat results_*.json
```

## Troubleshooting

### Out of GPU memory
- Reduce batch size or n_samples
- Use CPU-only mode: `--device cpu`

### Missing models
- Check model paths in scripts
- Models should be in `/data/models/` or similar

### Python errors
- Reinstall environment: `rm -rf dgx-venv && ./run_centaur_test.sh`

## Next Steps

After feature extraction:
1. Run LOO cross-validation
2. Compare with baseline (Random ≈ 120K, LLaMA-65B ≈ 30K)
3. Generate reports

## Git Workflow

```bash
# Stage your changes
git add .

# Commit
git commit -m "feat: run dgx-spark experiments"

# Push to remote (if needed)
git push origin feature/dgx-experiments
```
