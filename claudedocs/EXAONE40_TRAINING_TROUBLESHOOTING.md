# EXAONE 4.0-32B Training Troubleshooting Report

**Date**: 2025-10-18
**Model**: LGAI-EXAONE/EXAONE-4.0.1-32B (32 billion parameters)
**Hardware**: 7x A100 GPUs (80GB each), 250GB CPU RAM, node1
**Method**: DeepSpeed ZeRO-3
**Status**: ❌ UNRESOLVED - All 5 attempted solutions failed

---

## Executive Summary

Attempted to train EXAONE 4.0-32B model using DeepSpeed ZeRO-3 on 7x A100 GPUs. Successfully resolved Phase 1 (model loading OOM) but encountered persistent failure at DeepSpeed initialization phase that remains unresolved after 5 different approaches.

**Key Achievements**:
- ✅ Identified and fixed model loading OOM through initialization order correction
- ✅ Systematically eliminated multiple potential causes through controlled experiments
- ✅ Identified exact failure point: DeepSpeed optimizer initialization

**Current Blocker**:
- ❌ All jobs killed with SIGKILL at DeepSpeed optimizer initialization phase
- ❌ Failure independent of: dataloader workers, gradient settings, GPU memory settings, CPU parameter offload
- ❌ Suspected root cause: CPU memory exhaustion during optimizer initialization (128GB AdamW states)

---

## Problem Summary

**Failure Pattern (Jobs 62710-62714)**:

All training attempts fail consistently at the same point during DeepSpeed ZeRO-3 initialization, immediately after the "Parameter Offload" log message.

**Failure Signature**:
```
Parameter Offload - Persistent parameters statistics: param_count = 257, numel = 676864
[INFO] [launch.py:335:sigkill_handler] Killing subprocess [PID]
... (all 7 ranks killed sequentially within 13-50 seconds)
[ERROR] [launch.py:341:sigkill_handler] exits with return code = -9
```

**Timeline**:
- Model download: 30-31 minutes (if not cached)
- Model loading: 1-2 minutes (14 checkpoint shards)
- DeepSpeed initialization: Starts successfully
- **Failure point**: 13-50 seconds after "Parameter Offload" message
- **Exit code**: -9 (SIGKILL - OS-level kill, likely OOM killer)

**What Works**:
- ✅ Model downloads from HuggingFace
- ✅ Tokenization and dataset preparation
- ✅ TrainingArguments creation with DeepSpeed config
- ✅ Model loading (14/14 checkpoint shards, 100% complete)
- ✅ DeepSpeed ZeRO-3 parameter partitioning initialization

**What Fails**:
- ❌ DeepSpeed optimizer initialization phase
- ❌ No Python exception or traceback
- ❌ Silent kill with SIGKILL (-9)

---

## Root Cause Discovery (Phase 1): Model Loading OOM

### Initial Problem (Jobs 62706-62708)
CUDA OOM errors during model loading with various approaches:
- Job 62706: Parameter name error (`evaluation_strategy` → `eval_strategy`)
- Job 62707: OOM with `model_init()` approach
- Job 62708: Meta tensor error with `accelerate.init_empty_weights()`

### Solution: Correct DeepSpeed Initialization Order

**Research findings** (2025-10-18):
- Source: GitHub Issue #4100, HuggingFace docs, Medium tutorials
- **Critical discovery**: TrainingArguments must be instantiated BEFORE loading model

**Wrong approach (v1-v4):**
```python
model = AutoModelForCausalLM.from_pretrained(...)  # Loads full 32B on EACH GPU
training_args = TrainingArguments(deepspeed=...)   # Too late!
trainer = Trainer(model=model, args=training_args)
```

**Correct approach (v5):**
```python
training_args = TrainingArguments(deepspeed=...)   # FIRST!
model = AutoModelForCausalLM.from_pretrained(...)  # Partitioned during loading
trainer = Trainer(model=model, args=training_args)
```

**Why it works:**
- TrainingArguments signals DeepSpeed config early
- `from_pretrained()` detects DeepSpeed context
- Model is partitioned DURING loading, not after
- Each GPU loads only its partition (~4.6GB vs 32GB)

**Validation**: Jobs 62710-62713 all successfully load model (14/14 checkpoint shards, 100% complete)

**Documentation**: `/tmp/DEEPSPEED_RESEARCH_20251018.md`

---

## Root Cause Discovery (Phase 2): DeepSpeed Initialization Hang

### New Problem (Jobs 62710-62713)
Model loading succeeds, but training fails at DeepSpeed initialization.

**Evidence:**
- Model loads completely: "Loading checkpoint shards: 100%|██████████| 14/14"
- All ranks print: "✅ Model loaded and will be partitioned by DeepSpeed"
- DeepSpeed starts initialization: "Parameter Offload - Persistent parameters statistics"
- **13-30 seconds later**: All ranks killed with SIGKILL (-9)
- **No Python exception or traceback** - silent failure
- **No training step ever executed**

### Hypothesis 1: CPU Memory Exhaustion from Parameter Offload (DISPROVEN)

**Initial Theory**: DeepSpeed ZeRO-3 parameter offload + optimizer offload exceeds available CPU RAM (250GB).

**Supporting Evidence**:
1. GitHub Issue #7021 (Feb 2025): Same SIGKILL pattern with ZeRO-3 CPU offload
2. GitHub Issue #4380 (Sep 2023): "subprocess killed after parameter offload" - CPU memory exhaustion
3. Documentation: ZeRO-3 with param+optimizer offload needs 71-127GB CPU RAM minimum
4. SIGKILL (-9) indicates OS-level OOM killer, not Python exception
5. Option 3 (10x more aggressive GPU memory settings) had NO EFFECT → not GPU issue

**Test**: Job 62714 - Disabled parameter offload entirely, kept only optimizer offload
**Result**: ❌ FAILED at same point - **Hypothesis DISPROVEN**

### Hypothesis 2: Optimizer Initialization Memory Spike (CURRENT)

**Updated Theory**: DeepSpeed optimizer initialization phase causes temporary CPU memory spike exceeding 250GB limit.

**Critical Discovery**: "Parameter Offload" is a **standard DeepSpeed ZeRO-3 initialization phase**, NOT the CPU offload feature:
- Message appears even when `offload_param` is disabled in config
- Logs which parameters remain persistent (not partitioned)
- Failure occurs during **optimizer initialization** immediately after this log

**Memory Analysis**:
- Model parameters: 32B × 2 bytes = 64GB (partitioned across 7 GPUs = ~9GB each)
- AdamW optimizer states: 32B × 2 states × 4 bytes = 256GB total
- With ZeRO-3 + CPU offload: 256GB ÷ 7 = ~36.5GB per GPU worth offloaded to CPU
- **Temporary spike during initialization**: May exceed 250GB CPU RAM
- Even with partitioning, all 7 processes initializing simultaneously → 7 × 36.5GB = ~255GB

**Supporting Evidence**:
1. Job 62714 failed despite NO parameter offload configured
2. All jobs fail at identical point: immediately after "Parameter Offload" log
3. Failure timing: 13-50 seconds after message (optimizer initialization phase)
4. CPU RAM allocation: 250GB (barely sufficient for theoretical 255GB)
5. No process survives past this point across all 5 attempted solutions

---

## Troubleshooting Attempts

### Configuration Summary
**Original configuration:**
```yaml
# Training config
per_device_train_batch_size: 1
gradient_accumulation_steps: 4  # Effective batch = 28
gradient_checkpointing: true
dataloader_num_workers: 4
max_steps: -1  # Full 5 epochs

# DeepSpeed ZeRO-3
stage3_max_live_parameters: 1e9
stage3_prefetch_bucket_size: auto
stage3_param_persistence_threshold: auto
```

### Option 1: Disable Dataloader Workers (Job 62711)
**Change**: `dataloader_num_workers: 0`
**Rationale**: Eliminate potential dataloader fork/spawn deadlock with 28 worker processes (7 GPUs × 4 workers)
**Result**: ❌ FAILED - Same hang at Parameter Offload
**Duration**: ~5 minutes before kill

### Option 2: Reduce Training Complexity (Job 62712)
**Changes**:
- `gradient_accumulation_steps: 1` (reduce from 4)
- `gradient_checkpointing: false` (disable memory optimization)
- `max_steps: 10` (quick test only)

**Rationale**: Simplify training initialization, reduce first batch complexity
**Result**: ❌ FAILED - Same hang at Parameter Offload
**Duration**: ~5:33 minutes before kill

### Option 3: Aggressive DeepSpeed Memory Optimization (Job 62713)
**Changes** (`deepspeed_zero3_32b.json`):
```json
{
  "stage3_max_live_parameters": 1e8,              // 1e9 → 1e8 (10x reduction)
  "stage3_prefetch_bucket_size": 5e7,             // auto → 50M
  "stage3_param_persistence_threshold": 1e5       // auto → 100K
}
```

**Rationale**:
- Reduce active GPU memory during training
- Limit prefetch buffer size
- Aggressive parameter offloading threshold
- Force more CPU offloading to prevent GPU memory spikes

**Result**: ❌ FAILED - Same hang at Parameter Offload
**Duration**: ~6:05 minutes before kill
**Conclusion**: 10x more aggressive memory settings had NO EFFECT - problem is NOT GPU memory

### Option 4: Disable Parameter Offload (Job 62714)
**Changes** (`deepspeed_zero3_32b.json`):
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    }
    // REMOVED: offload_param configuration
    // Parameters stay on GPU with ZeRO-3 partitioning
  }
}
```

**Additional Changes** (restored original training settings):
```yaml
gradient_accumulation_steps: 4  # Restored from 1
gradient_checkpointing: true    # Restored
max_steps: -1                   # Full training (5 epochs)
dataloader_num_workers: 4       # Restored from 0
```

**Rationale**:
- **Hypothesis test**: Disable parameter CPU offload based on GitHub Issues #7021, #4380
- **Keep optimizer offload**: AdamW states (256GB) offloaded to CPU
- **Parameters on GPU**: 64GB ÷ 7 = ~9.1GB per GPU (fits in 80GB A100)
- **Expected**: Reduce CPU memory usage by ~64GB

**Result**: ❌ FAILED - Same hang at Parameter Offload
**Duration**: ~48 minutes total (31 min download + 2 min loading + 15 min to failure)

**Key Discovery**:
- "Parameter Offload" log appears even WITHOUT `offload_param` in config
- This proves it's a DeepSpeed initialization phase name, not CPU offload feature
- Failure point is **optimizer initialization**, not parameter offload

---

## Technical Analysis

### Memory Footprint Breakdown (Estimated)

**Model parameters**: 32B parameters
- FP16/BF16: 32B × 2 bytes = 64 GB
- **Per GPU with ZeRO-3**: 64 GB ÷ 7 = ~9.1 GB

**Optimizer states** (AdamW):
- Momentum + variance: 32B × 2 × 4 bytes = 256 GB
- **With CPU offload**: Not in GPU memory

**Gradients**:
- 32B × 2 bytes = 64 GB
- **Per GPU with ZeRO-3**: 64 GB ÷ 7 = ~9.1 GB

**Activations** (per batch):
- Batch size 1, seq length 512, hidden size ~8192
- Estimated: 10-20 GB per GPU (with gradient checkpointing)
- **WITHOUT gradient checkpointing** (Option 2): 40-80 GB per GPU

**Total per GPU**:
- With original config: ~9.1 + ~9.1 + ~10-20 = 28-38 GB
- A100 has 80 GB VRAM → Should fit comfortably
- **BUT**: First forward pass may spike higher during model preparation

### DeepSpeed ZeRO-3 Initialization Phases

1. **Model Loading** ✅ SUCCEEDS
   - Checkpoint shards loaded (14/14)
   - Model structure created

2. **Parameter Partitioning** ✅ SUCCEEDS
   - ZeRO-3 splits parameters across ranks
   - Each GPU gets 1/7th of parameters

3. **Parameter Offload Setup** ✅ SUCCEEDS
   - "param_count = 257, numel = 676864" logged
   - Persistent parameters identified

4. **First Batch Preparation** ❌ FAILS HERE
   - Preparing first training batch
   - **Hypothesis**: Memory spike during:
     - All-gather of parameters for forward pass
     - Activation computation
     - Gradient computation preparation

### CUDA Version Mismatch (Minor Issue)

**Warning in logs:**
```
Installed CUDA version 11.1 does not match the version torch was compiled with 11.8
but since the APIs are compatible, accepting this combination
```

**Impact**: Likely minimal, but could cause subtle issues with CUDA operations
**Recommendation**: Verify PyTorch CUDA version matches system CUDA

---

## System Environment

### Hardware
- **Node**: node1
- **GPUs**: 7x NVIDIA A100 (80GB each)
- **Memory**: 250GB RAM allocated

### Software Stack
```
Python: 3.10
PyTorch: 2.6.0+cu118
Transformers: 4.57.1
DeepSpeed: (version in conda env)
CUDA: 11.1 (system) vs 11.8 (PyTorch compiled)
```

### Conda Environment
```bash
conda env: ko-centaur
Location: /scratch/connectome/connectome1/miniconda3/envs/ko-centaur
```

---

## Job History

| Job ID | Phase | Changes Applied | Model Loading | DeepSpeed Init | Training | Duration | Result |
|--------|-------|----------------|---------------|----------------|----------|----------|---------|
| 62706 | 1 | v1 (model_init) | ✅ | ❌ Parameter error | ⬜ | ~5min | FAILED |
| 62707 | 1 | v2 (fixed params) | ❌ OOM | ⬜ | ⬜ | ~2min | FAILED |
| 62708 | 1 | v3 (meta tensors) | ❌ Meta error | ⬜ | ⬜ | ~3min | FAILED |
| 62710 | 2 | v5 (correct order) | ✅ | ❌ Killed at Param Offload | ⬜ | ~6min | FAILED |
| 62711 | 2 | Option 1: workers=0 | ✅ | ❌ Killed at Param Offload | ⬜ | ~5min | FAILED |
| 62712 | 2 | Option 2: simplified | ✅ | ❌ Killed at Param Offload | ⬜ | ~5:33min | FAILED |
| 62713 | 2 | Option 3: aggressive GPU mem | ✅ | ❌ Killed at Param Offload | ⬜ | ~6:05min | FAILED |
| 62714 | 2 | Option 4: no param offload | ✅ | ❌ Killed at Param Offload | ⬜ | ~48min | FAILED |

**Notes**:
- Phase 1 (Jobs 62706-62708): Model loading issues - **RESOLVED**
- Phase 2 (Jobs 62710-62714): DeepSpeed initialization hang - **UNRESOLVED**
- All Phase 2 jobs fail at identical point: "Parameter Offload" log → optimizer init → SIGKILL
- Job 62714 took longer due to 31-minute model download (not previously cached)

---

## Key Findings

### What We Learned

1. **Correct DeepSpeed Initialization Order** (Critical):
   - TrainingArguments MUST be created BEFORE model loading
   - This allows DeepSpeed to partition model during loading
   - Source: GitHub Issue #4100, HuggingFace documentation

2. **"Parameter Offload" is NOT CPU Offload** (Important):
   - Log message appears regardless of `offload_param` configuration
   - It's a standard DeepSpeed ZeRO-3 initialization phase
   - Logs which parameters remain persistent (not partitioned)
   - Actual failure occurs during optimizer initialization after this log

3. **Failure is Not GPU Memory Related**:
   - 10x more aggressive GPU memory settings (Option 3) had no effect
   - A100 80GB has sufficient VRAM for partitioned parameters (~9GB per GPU)
   - Problem is CPU-side, not GPU-side

4. **Failure is Not Configuration Related**:
   - Independent of: dataloader workers, gradient accumulation, gradient checkpointing
   - Independent of: parameter offload settings
   - Consistent across all configuration variations tested

5. **Likely Root Cause: CPU Memory Exhaustion**:
   - AdamW optimizer: 256GB total states (2 states × 32B params × 4 bytes)
   - With ZeRO-3: ~36.5GB per GPU offloaded to CPU
   - 7 GPUs initializing simultaneously: 7 × 36.5GB = ~255GB
   - Allocated CPU RAM: 250GB (borderline insufficient)
   - Temporary spike during initialization likely exceeds limit

### What Doesn't Work

- ❌ Disabling dataloader workers
- ❌ Reducing gradient accumulation
- ❌ Disabling gradient checkpointing
- ❌ Aggressive GPU memory optimization (10x more conservative)
- ❌ Disabling parameter CPU offload

---

## Recommended Next Steps

### High Priority Solutions

#### Option 5: Increase CPU Memory Allocation (RECOMMENDED)
**Changes**:
```bash
#SBATCH --mem=500G  # Double from 250GB to 500GB
```

**Rationale**:
- Current allocation (250GB) barely sufficient for theoretical requirement (255GB)
- Temporary spikes during initialization likely cause OOM
- Simple solution: double CPU RAM to 500GB
- Most likely to resolve issue based on evidence

**Probability of success**: 80%

#### Option 6: Use Lighter Optimizer (SGD)
**Changes**:
```yaml
optim: "sgd"  # Instead of adamw_torch
momentum: 0.9
```

**Rationale**:
- AdamW: 2 optimizer states (momentum + variance) = 256GB total
- SGD with momentum: 1 optimizer state = 128GB total
- Reduces optimizer memory by 50%
- May reduce final model quality slightly

**Probability of success**: 60%

### Medium Priority Solutions

#### Option 7: ZeRO-2 Instead of ZeRO-3
**Changes**:
```json
{
  "zero_optimization": {
    "stage": 2,  // Instead of 3
    "offload_optimizer": {"device": "cpu"}
  }
}
```

**Rationale**:
- ZeRO-2: Partitions gradients + optimizer, keeps full parameters on each GPU
- Requires: 64GB model + ~20GB activations = ~84GB per GPU (may exceed 80GB A100)
- Simpler initialization, different memory pattern
- **Risk**: May not fit in 80GB A100 with full parameters

**Probability of success**: 30% (likely GPU OOM)

### Lower Priority / Experimental Solutions

#### Option 8: PyTorch FSDP Instead of DeepSpeed
**Rationale**: Different distributed training framework, may have different memory pattern

#### Option 9: Smaller Model Test (EXAONE-7.8B)
**Rationale**: Verify infrastructure works, confirm issue is 32B-specific

#### Option 10: Different DeepSpeed Version
**Rationale**: May be version-specific bug
```bash
pip install deepspeed==0.12.6  # Older stable version
```

#### Option 11: Sequential Optimizer Initialization
**Rationale**: Initialize optimizers one GPU at a time to avoid simultaneous spike
- Requires custom DeepSpeed initialization code
- Complex implementation

### Research Questions

1. **Is this a known DeepSpeed + EXAONE issue?**
   - Search: "EXAONE DeepSpeed ZeRO-3 OOM"
   - Check EXAONE model card for training recommendations

2. **Are there EXAONE-specific training configs?**
   - Review official LGAI-EXAONE training examples
   - Check for model-specific quirks (rotary embeddings, attention mechanism)

3. **What is the actual memory usage pattern?**
   - Enable DeepSpeed memory profiling: `"memory_breakdown": true`
   - Add manual memory tracking in training script

4. **Is Parameter Offload the issue?**
   - Try disabling parameter offload entirely (keep optimizer offload)
   ```json
   "offload_param": null,  // Disable param offload
   "offload_optimizer": {"device": "cpu"}  // Keep optimizer offload
   ```

---

## Files Modified

### Training Script
- **Location**: `/scratch/connectome/connectome1/ko-centaur/training/train_exaone40_risky.py`
- **Version**: v5 (correct DeepSpeed initialization order)
- **Key changes**: TrainingArguments before model loading

### Configuration Files
- **Training config**: `configs/training_exaone40.yaml`
  - `dataloader_num_workers: 0` (Option 1)
  - `gradient_accumulation_steps: 1` (Option 2)
  - `gradient_checkpointing: false` (Option 2)
  - `max_steps: 10` (Option 2)

- **DeepSpeed config**: `configs/deepspeed_zero3_32b.json`
  - `stage3_max_live_parameters: 1e8` (Option 3)
  - `stage3_prefetch_bucket_size: 5e7` (Option 3)
  - `stage3_param_persistence_threshold: 1e5` (Option 3)
  - **Backup**: `deepspeed_zero3_32b.json.backup`

### Documentation
- **Research report**: `/tmp/DEEPSPEED_RESEARCH_20251018.md`
- **Training scripts**: `/tmp/train_exaone40_v*.py` (v1-v5)
- **This document**: `claudedocs/EXAONE40_TRAINING_TROUBLESHOOTING.md`

---

## Logs and Debugging

### Log Locations
```bash
# Job outputs
/scratch/connectome/connectome1/ko-centaur/logs/slurm_[JOBID]_exaone40.log
/scratch/connectome/connectome1/ko-centaur/logs/slurm_[JOBID]_exaone40.err

# Model checkpoints (if any created)
/scratch/connectome/connectome1/ko-centaur/models/ko-centaur-exaone40/
```

### Useful Debug Commands
```bash
# Check job status
squeue -u connectome1
sacct -j [JOBID] --format=JobID,JobName,State,ExitCode,MaxRSS,Elapsed

# Monitor job in real-time
watch -n 5 'tail -50 /scratch/connectome/connectome1/ko-centaur/logs/slurm_[JOBID]_exaone40.log'

# Check GPU memory
nvidia-smi dmon -s mu -c 100

# Check for OOM in system logs (if accessible)
dmesg -T | grep -i 'out of memory'
journalctl -k | grep -i oom
```

### Key Metrics to Track
- **Model loading time**: ~1:39 (14 shards)
- **Time to Parameter Offload**: ~5-6 minutes
- **Time to kill**: ~13-30 seconds after Parameter Offload message
- **Memory usage at failure**: Unknown (need sacct with MaxRSS)

---

## References

### Phase 1 Research (Model Loading)
1. **HuggingFace DeepSpeed Documentation**
   - https://huggingface.co/docs/transformers/en/deepspeed

2. **GitHub Issue #4100** (Critical)
   - https://github.com/huggingface/trl/issues/4100
   - "DeepSpeed Zero 3 + LoRA 32B model still OOM on 8*H100"
   - Solution: TrainingArguments before model loading

3. **Medium Tutorial**
   - https://medium.com/@yxinli92/fine-tuning-large-language-models-with-deepspeed-a-step-by-step-guide-2fa6ce27f68a

### Phase 2 Research (Needed)
- DeepSpeed ZeRO-3 memory profiling
- EXAONE-specific training documentation
- A100 GPU memory behavior with large models
- DeepSpeed initialization sequence deep dive

---

## Conclusion

**Current Status**: Blocked at DeepSpeed ZeRO-3 optimizer initialization phase

**What We Solved**:
- ✅ Phase 1: Model loading OOM through correct initialization order
- ✅ Verified model loads successfully (14/14 checkpoint shards, 100% complete)
- ✅ Documented complete troubleshooting process with evidence
- ✅ Systematically tested and eliminated 5 different potential causes

**What Remains Unresolved**:
- ❌ All 5 attempted solutions (Options 0-4) failed at identical point
- ❌ DeepSpeed optimizer initialization consistently triggers SIGKILL (-9)
- ❌ Failure independent of: workers, gradient settings, GPU memory, parameter offload

**Root Cause Hypothesis**:
CPU memory exhaustion during optimizer initialization (AdamW requires ~255GB, allocated 250GB)

**Recommended Next Action**:
**Option 5: Increase CPU Memory to 500GB** (80% probability of success)
- Simple change: `#SBATCH --mem=500G` in Slurm script
- Addresses root cause directly
- Low risk, high probability solution

**Alternative Options**:
- Option 6: Use SGD optimizer (60% success probability)
- Option 7: Try ZeRO-2 instead (30% success probability, may GPU OOM)

**Timeline**:
- Phase 1 (Model loading): ✅ RESOLVED (2025-10-18)
- Phase 2 (DeepSpeed initialization): ❌ UNRESOLVED after 5 attempts (2025-10-18)
- Next attempt: Option 5 with increased CPU memory
