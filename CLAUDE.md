# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

CENTaUR is a research codebase for turning large language models into cognitive models, based on the ICLR 2023 paper by Binz & Schulz. The project analyzes human decision-making using LLaMA model embeddings as cognitive representations.

**Note**: This codebase is outdated and primarily for reference. For current work in this domain, see the [tutorial paper](https://osf.io/preprints/psyarxiv/f7stn) and [updated implementation](https://github.com/Zak-Hussain/LLM4BeSci).

## Architecture

### Core Components

**LLaMA Integration** (`llama/`, `inference.py`):
- Custom LLaMA implementation with distributed loading via `accelerate`
- `LLaMAInference`: Wrapper for loading models and generating text with temperature/top-p sampling
- `Transformer`: Base model with rotary embeddings, RMS normalization, SwiGLU feedforward
- Models use cached KV attention for efficient autoregressive generation
- The `hl` attribute stores last-layer hidden states used as cognitive features

**Cognitive Models** (`models.py`):
- `BinomialRegression`: Base model fitting logits from LLaMA features to human choices
- `TemperatureBinomialRegression`: Temperature-scaled softmax over token probabilities (tokens 1/2)
- `JointBinomialRegression`: Jointly fits two tasks with temperature parameter and clipping
- `MixedEffectBinomialRegression`: Adds random effects per participant group
- All use LBFGS optimization with L2 regularization (alpha parameter)

### Three Experimental Domains

**choices13k/** - Risky choice prediction on 13K gambling problems:
- `query.py`: Generates prompts describing probabilistic payoffs, extracts hidden states
- `fit_centaur.py`: Fits binomial regression to LLaMA features with nested cross-validation
- `fit_beast.py`: Baseline cognitive model comparison (BEASTsd/)
- Uses last-layer representations as features for choice prediction

**HorizonTask/** - Sequential decision-making (explore-exploit):
- `query.py`: Builds contextual prompts with observation histories and trial budgets
- `fit_centaur.py`: Fixed-effects binomial regression on LLaMA features
- `fit_mixed_centaur.py`: Mixed-effects model with per-participant random effects
- `fit_hybrid.py`: Combines symbolic task representations with LLaMA features
- Multiple datasets: `exp1`, `exp2` with forced/free choice trials

**ExperientialSymbolicTask/** - Learning from experience vs. description:
- `query.py`: Mixes experiential learning history with symbolic gamble descriptions
- `fit_joint.py`: Jointly fits description-based (DfD) and experience-based (DfE) tasks
- Tests transfer of LLaMA representations across learning contexts

### Workflow Pattern

All experimental domains follow this pipeline:

1. **Query (`query.py`)**:
   - Loads LLaMA model with `LLaMAInference(llama_path, model)`
   - Formats task-specific prompts (probabilities, histories, goals)
   - Generates text with `temperature=0.0` (deterministic)
   - Extracts `llama.generator.model.hl` (last hidden state) as features
   - Saves features + human actions to `data/model={model}_*.pth`

2. **Fit (`fit_*.py`)**:
   - Loads pre-extracted features from query phase
   - Performs nested cross-validation for regularization (alpha)
   - Fits binomial regression: features → logits → choice probabilities
   - Computes log-likelihood on held-out test fold
   - Saves results to `data/loo_*.pth`

3. **Plot/Eval (`plot_*.py`, `eval.py`)**:
   - Aggregates cross-validation results
   - Compares model performance (CENTaUR vs baselines)
   - Generates visualizations of predictions

## Common Commands

### Extract LLaMA Features

```bash
# HorizonTask
cd HorizonTask
python query.py --llama-path /path/to/llama --model 7B --dataset exp1

# choices13k
cd choices13k
python query.py --llama-path /path/to/llama --model 7B

# ExperientialSymbolicTask
cd ExperientialSymbolicTask
python query.py --llama-path /path/to/llama --model 7B
```

Models: `7B`, `13B`, `30B`, `65B` (must match directory names in llama-path)

### Fit Cognitive Models

```bash
# Single fold cross-validation
python fit_centaur.py --model 7B --foldid 0

# All folds (typically 0-99)
for i in {0..99}; do
    python fit_centaur.py --model 7B --foldid $i
done
```

### Extract Last Layer Weights

```bash
# From repository root
python last_layer.py --llama-path /path/to/llama
# Saves output layer weights for all models to data/last_layer_{model}.pth
```

## Key Implementation Details

### LLaMA Model Setup
- Requires LLaMA weights with `state_dict.pth`, `params.json`, `tokenizer.model`
- Uses `load_checkpoint_and_dispatch` for multi-GPU distributed loading
- `device_map="auto"` automatically distributes layers across available GPUs
- `no_split_module_classes=["TransformerBlock"]` keeps attention blocks intact

### Feature Extraction Pattern
```python
llama = LLaMAInference(llama_path, model, max_batch_size=2)
results, _ = llama.generate([prompt], temperature=0.0, top_p=1, max_length=1)
features = llama.generator.model.hl.squeeze().detach().cpu()
```
- `max_length=1`: Only generates one token (choice between "1" or "2")
- Hidden state from last token position used as cognitive representation
- Temperature=0.0 for deterministic generation during feature extraction

### Cross-Validation Structure
- Outer loop: 100-fold leave-one-out for test evaluation
- Inner loop: 11-fold nested CV for hyperparameter selection (alpha)
- Alpha grid: `[0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]`
- Features normalized using training set statistics before fitting

### Prompt Engineering
- All prompts end with partial completion: `"A: Machine"` (model completes with "1" or "2")
- Use `replace_right()` to format lists with proper "and" placement
- Horizon task includes explicit trial budget: "X additional choices"
- Goals always mention maximizing dollars to align with human incentives

## Server Infrastructure

### SSH Access
- **Compute Server**: Use SSH alias `server` to connect
  ```bash
  ssh server
  ```
- **Hardware**: 7x RTX GPUs (24GB each), 250-500GB CPU RAM
- **Work Directory**: `/scratch/connectome/connectome1/ko-centaur`
- **Partition**: `octopus` (SLURM)
- **Node**: `node1` (primary GPU node)

### Ko-CENTaUR Project

**Location**: `/scratch/connectome/connectome1/ko-centaur`

**Goal**: Replicate original CENTaUR methodology (Binz & Schulz, 2023) using modern Korean-capable LLMs

**Models**:
1. **Qwen2.5-32B-Instruct** with QLoRA fine-tuning
2. **DeepSeek-R1-Distill-Qwen-32B** with QLoRA fine-tuning

**Evaluation Pipeline** (Original CENTaUR Methodology):

```
1. Feature Extraction (scripts/extract_centaur_features.py)
   ├─ Load fine-tuned model (base + LoRA adapter)
   ├─ Forward pass on prompts (NO generation)
   ├─ Extract last-layer hidden states (3072-dim)
   └─ Save features → data/features/*.pth

2. 100-fold LOO Cross-Validation (scripts/fit_centaur_loo_cv.py)
   ├─ Load extracted features
   ├─ Nested 11-fold CV for alpha selection
   ├─ Fit BinomialRegression (legacy/models.py)
   └─ Compute negative log-likelihood (NLL)

3. Benchmark Comparison
   ├─ Random baseline: NLL ≈ 120,000
   ├─ LLaMA-65B (original): NLL ≈ 30,000
   ├─ Qwen2.5-32B (ours): NLL = ?
   └─ DeepSeek-R1 (ours): NLL = ?
```

**Current Status** (2025-10-26):
- ✅ Fine-tuning: Qwen2.5-32B, DeepSeek-R1 trained on Choices13k
- ✅ Feature extraction scripts: Production-ready with NF4 quantization
- ✅ SLURM scripts: `submit_extract_qwen25.sh`, `submit_extract_deepseek.sh`
- ⏳ LOO CV implementation: In progress
- ⏳ Benchmark results: Pending feature extraction

**Key Files**:
- `scripts/extract_centaur_features.py`: Main extraction script
- `scripts/fit_centaur_loo_cv.py`: Cross-validation (to be finalized)
- `claudedocs/EVALUATION_METHODOLOGY_ANALYSIS.md`: Detailed methodology comparison
- `legacy/models.py`: Original BinomialRegression implementation

### SLURM Commands

**Complete Pipeline**:
```bash
# Step 1: Extract features (requires GPU)
sbatch scripts/submit_extract_qwen25.sh
sbatch scripts/submit_extract_deepseek.sh

# Step 2: 100-fold LOO CV (CPU only, after features are extracted)
sbatch scripts/submit_fit_qwen25.sh
sbatch scripts/submit_fit_deepseek.sh

# Monitor jobs
tail -f /scratch/connectome/connectome1/ko-centaur/logs/extract_*.out
tail -f /scratch/connectome/connectome1/ko-centaur/logs/fit_*.out
```

**General SLURM**:
```bash
# Check job status
squeue -u $USER

# Monitor GPU (for feature extraction)
watch -n 1 nvidia-smi

# Cancel job
scancel <job_id>
```

**Quick Test** (Local):
```bash
# Test feature extraction with 10 samples
python scripts/extract_centaur_features.py --model qwen25 --n_samples 10

# Test LOO CV (requires extracted features)
python scripts/fit_centaur_loo_cv.py --model qwen25
```

## Critical Implementation Requirements

### ⚠️ MANDATORY: GPU Initialization in Training Scripts

**ALL QLoRA training scripts MUST include GPU initialization code at the start of main().**

Without this code, `Accelerate` will fail to detect GPUs properly, resulting in:
```
TypeError: device() received an invalid combination of arguments - got (NoneType)
```

**Required GPU initialization template:**
```python
def main():
    # ... load config ...
    
    # ============================================================================
    # GPU 사용 강제 확인 (MANDATORY!)
    # ============================================================================
    print("\n[0/6] GPU 사용 확인 및 강제 설정...")
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA를 사용할 수 없습니다! GPU가 필요합니다.")
    
    num_gpus = torch.cuda.device_count()
    print(f"✅ CUDA 사용 가능: {num_gpus}개 GPU 감지")
    
    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")
    
    # CUDA_VISIBLE_DEVICES가 설정되어 있으면 사용, 없으면 모든 GPU 사용
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible:
        print(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")
    else:
        print(f"   CUDA_VISIBLE_DEVICES: 설정 안됨 (모든 GPU 사용)")
    
    # 기본 디바이스를 GPU로 설정
    device = torch.device("cuda:0")
    print(f"   기본 디바이스: {device}")
    print("✅ GPU 사용 준비 완료")
    
    # ... continue with tokenizer, dataset, model loading ...
```

**Why this is critical:**
1. Forces early GPU detection before model loading
2. Ensures `Accelerate` properly initializes with CUDA devices
3. Prevents device mapping failures that cause `NoneType` errors
4. Provides clear error messages if GPUs are unavailable

**Affected scripts:**
- `ko_centaur/training/train_qwen25_32b_qlora.py` ✅ HAS IT
- `ko_centaur/training/train_deepseek_r1_qwen32b_qlora.py` ✅ FIXED (added 2025-11-09)
- **ANY NEW** QLoRA training script MUST include this!

**Checklist for new training scripts:**
- [ ] Import `os` module
- [ ] Add GPU initialization code in `main()` before tokenizer loading
- [ ] Test with NGC PyTorch container
- [ ] Verify training starts without device errors

