# ZeRO-3 Infinity Implementation Plan

## Status: Planning Complete
Date: 2025-10-19

## Goal
Train EXAONE 4.0-32B on 7x RTX 24GB GPUs using DeepSpeed ZeRO-3 Infinity

## Key Strategy
- Activation checkpointing: Reduce activation memory 15-20GB → 3-5GB
- Aggressive parameter management: 1e8 max live parameters (10x stricter)
- Memory efficient linear layers
- NVMe offload ready (if CPU RAM insufficient)

## Expected Outcome
- GPU memory per device: 14-17GB (within 24GB limit)
- Training speed: 10-50x slower than baseline
- Success probability: ~50%

## Implementation Phases
1. Environment setup (30min)
2. DeepSpeed config creation (20min)
3. Training script modification (30min)
4. Validation run (1hr)
5. Full training launch (monitoring)

## Files to Create
- `/scratch/connectome/connectome1/ko-centaur/configs/deepspeed_zero3_infinity.json`
- `/scratch/connectome/connectome1/ko-centaur/configs/training_exaone40_infinity.yaml`
- `/scratch/connectome/connectome1/ko-centaur/train_exaone40_infinity.py`
- `/scratch/connectome/connectome1/ko-centaur/submit_exaone40_infinity.sh`

## Risk Factors
- Activation checkpointing overhead may exceed acceptable limits
- NVMe offload may trigger if CPU RAM insufficient
- Numerical stability with aggressive parameter swapping

## Fallback Options
1. QLoRA (85% success, faster, adapter-only)
2. EXAONE 7.8B (100% success, smaller model)
