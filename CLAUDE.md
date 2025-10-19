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

### Ko-CENTaUR Project (EXAONE Training)
- **Location**: `/scratch/connectome/connectome1/ko-centaur`
- **Model**: LGAI-EXAONE/EXAONE-4.0.1-32B (32B parameters)
- **Training Method**: DeepSpeed ZeRO-3 Infinity with activation checkpointing
- **Data**: `data/risky_choice_train.jsonl`
- **Expected GPU Memory**: 14-17GB per GPU (with checkpointing)
- **Training Speed**: 10-50x slower than baseline (due to activation recomputation)

### SLURM Commands
```bash
# Submit job
sbatch submit_exaone40_infinity.sh

# Check job status
squeue -u $USER

# Monitor GPU
watch -n 1 nvidia-smi

# View logs
tail -f logs/infinity_*.out
tail -f logs/gpu_monitor_*.log

# Cancel job
scancel <job_id>
```
