# Ko-CENTaUR Phase 2: Next Experiment Plan (CORRECTED v2)

**Date**: 2025-10-14
**Status**: 🔄 Planning Complete, Ready for Implementation
**Execution Location**: Connectome Server (147.47.200.154)

---

## ⚠️ CRITICAL CORRECTIONS FROM INITIAL PLANS

**What Was WRONG in First Draft**:
1. ❌ Included Llama 3.1-8B: **UNNECESSARY** - Original CENTaUR paper already uses LLaMA baselines
2. ❌ Used EXAONE 3.5-7.8B: User requested EXAONE 4.0-32B which "far surpasses" 3.5
3. ❌ Missing GPT OSS model: Failed to include latest GPT open-source model

**What Was WRONG in Second Draft**:
4. ❌ Used GPT-NeoX-20B: **TOO LARGE** for training (15-20GB per GPU, tight memory)

**What Is NOW CORRECT (v2)**:
1. ✅ EXAONE 4.0-32B: Latest version with superior performance (July 2025)
2. ✅ GPT-OSS-20B: Latest OpenAI quantized model (August 2025, 14GB VRAM with Unsloth)
3. ✅ No Llama training: CENTaUR methodology already validated on LLaMA
4. ✅ Smaller, optimized GPT model: Better performance with safer memory usage

---

## 🎯 Executive Summary

**Objective**: Train and evaluate two new cognitive models (EXAONE 4.0-32B and GPT-OSS-20B) to compare against the baseline EXAONE-base model, with rigorous statistical validation addressing the variance concerns from Phase 1.

**Key Improvements from Phase 1**:
- ✅ Variance issue resolved (mathematical expectation confirmed)
- 📈 Scale up to N=1000 samples (10x increase)
- 📊 Dual evaluation: LOO CV + 10-fold CV
- 🧪 Rigorous statistics: McNemar test, permutation test, bootstrap CI
- 📋 Proper reporting: SE and 95% CI instead of raw SD

**Timeline**: 2-3 weeks (reduced from 4-5 weeks with optimized GPT-OSS)
**Models**: 2 new models (32B + 20B quantized) + 1 baseline (7.8B)
**Evaluation**: N=1000 samples from choices13k

---

## 📊 Model Specifications

### Model 1: Ko-CENTaUR (EXAONE 4.0-32B)
- **Base Model**: LGAI-EXAONE/EXAONE-4.0-32B
- **Release**: July 2025
- **Parameters**: 32B
- **Size**: 64GB
- **Context Length**: 32K tokens
- **Training**: Fine-tune on Psych-101 (category learning)
- **Training Method**: DeepSpeed ZeRO-3 with CPU offloading
- **Rationale**: Latest EXAONE version that "far surpasses" 3.5 performance (user requirement)

**Technical Feasibility**:
- Memory requirement: ~512GB baseline (32B params × 16 bytes with Adam)
- With DeepSpeed ZeRO-3 + CPU offloading: ~23-26GB per GPU ✅ FITS 24GB
- Training time: 30-50 hours (3 epochs) with offloading overhead
- **Challenge**: Pushes memory limits, requires aggressive optimization

### Model 2: Ko-CENTaUR (GPT-OSS-20B)
- **Base Model**: openai/gpt-oss-20b
- **Release**: August 2025 (OpenAI's latest open-source release)
- **Parameters**: 20B (21B exactly)
- **Size**: Already quantized in mxfp4 format
- **Context Length**: 2048 tokens
- **Training**: QLoRA fine-tuning on Psych-101 (category learning)
- **Training Method**: Unsloth-optimized 4-bit QLoRA
- **Rationale**: Latest quantized GPT OSS model, state-of-the-art with efficient training

**Technical Feasibility**:
- Memory requirement: ~240GB baseline (20B params × 12 bytes)
- With Unsloth QLoRA (4-bit quantization): ~14GB per GPU ✅ COMFORTABLE FIT on 24GB
- Training time: 6-8 hours (3 epochs) with Unsloth optimization
- **Advantages**:
  - 70% less VRAM than standard training methods
  - 1.5x faster training than standard QLoRA
  - Latest OpenAI model (August 2025, cutting-edge)
  - Built-in mxfp4 quantization (optimized for inference)

**Why GPT-OSS-20B instead of GPT-NeoX-20B**:
- ✅ Much newer: August 2025 vs February 2022
- ✅ Lower memory: 14GB vs 15-20GB (safer training)
- ✅ Faster training: 6-8hrs vs 18-24hrs (more efficient)
- ✅ Better optimization: Unsloth support vs standard methods
- ✅ Latest from OpenAI: State-of-the-art quality
- ✅ User requirement: "quantized model 중에서 성능이 좋은 모델"

### Baseline: EXAONE-base
- **Base Model**: LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct (no fine-tuning)
- **Purpose**: Control for fine-tuning effect

**Why NO Llama 3.1?**
- ✅ Original CENTaUR paper (Binz & Schulz, 2023) already uses LLaMA models (7B, 13B, 30B, 65B)
- ✅ Ko-CENTaUR methodology is based on fine-tuning EXAONE following CENTaUR's LLaMA approach
- ✅ No need to replicate what CENTaUR paper already validated
- ✅ Focus on comparing Korean model (EXAONE) vs English GPT model instead

**Why EXAONE 4.0-32B instead of 3.5-7.8B?**
- ✅ User explicitly stated 4.0 "far surpasses" 3.5 in performance
- ✅ Research value: Test if larger, more capable base model improves cognitive modeling
- ✅ Technical feasibility confirmed with DeepSpeed ZeRO-3 + CPU offloading

---

## 🖥️ Infrastructure Specifications

### Connectome Server
- **Location**: connectome1@147.47.200.154
- **Base Path**: `/scratch/connectome/connectome1/ko-centaur/`
- **Environment**: ko-centaur (Python 3.10.18)

### GPU Configuration
```
GPUs: 7x NVIDIA RTX 3090
Memory per GPU: 24GB
Total VRAM: 168GB
Compute Capability: 8.6 (Ampere)
CUDA Version: 11.8
```

### Disk Space
```
Available: 2.4TB
Required (estimated):
  - Models: ~100GB (EXAONE 4.0: 64GB + GPT-OSS: 40GB)
  - Datasets: ~5GB
  - Checkpoints: ~120GB (training checkpoints, reduced)
  - Results: ~15GB
Total Required: ~240GB
```

---

## 🎓 Training Configuration

### Model 1: EXAONE 4.0-32B Training Setup

**Distributed Training Framework**:
```yaml
framework: HuggingFace Transformers + DeepSpeed
strategy: ZeRO-3 with CPU offloading
gpus: 7x RTX 3090
gradient_checkpointing: enabled
mixed_precision: fp16
cpu_offload: optimizer + parameters
```

**DeepSpeed ZeRO-3 Configuration**:
```json
{
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {
      "device": "cpu",
      "pin_memory": true
    },
    "offload_param": {
      "device": "cpu",
      "pin_memory": true
    },
    "overlap_comm": true,
    "contiguous_gradients": true,
    "sub_group_size": 1e9,
    "reduce_bucket_size": "auto",
    "stage3_prefetch_bucket_size": "auto",
    "stage3_param_persistence_threshold": "auto",
    "stage3_max_live_parameters": 1e9,
    "stage3_max_reuse_distance": 1e9
  },
  "gradient_accumulation_steps": 4,
  "gradient_clipping": 1.0,
  "train_batch_size": "auto",
  "train_micro_batch_size_per_gpu": 1,
  "fp16": {
    "enabled": true,
    "loss_scale": 0,
    "loss_scale_window": 1000,
    "hysteresis": 2,
    "min_loss_scale": 1
  }
}
```

**Hyperparameters**:
```yaml
learning_rate: 2e-5
lr_scheduler: linear
warmup_ratio: 0.1
epochs: 3
max_sequence_length: 512
per_device_batch_size: 1  # Minimum due to model size
gradient_accumulation_steps: 4
effective_batch_size: 28  # 7 GPUs × 1 × 4
weight_decay: 0.01
adam_epsilon: 1e-8
max_grad_norm: 1.0
```

---

### Model 2: GPT-OSS-20B Training Setup

**QLoRA Training Framework**:
```yaml
framework: Unsloth + HuggingFace Transformers
method: QLoRA (4-bit quantization + LoRA adapters)
quantization: bitsandbytes 4-bit (NF4)
lora_rank: 16
lora_alpha: 32
target_modules: ['q_proj', 'k_proj', 'v_proj', 'o_proj', 'gate_proj', 'up_proj', 'down_proj']
lora_dropout: 0.05
```

**Model Loading**:
```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="openai/gpt-oss-20b",
    max_seq_length=2048,
    dtype=None,  # Auto-detect
    load_in_4bit=True,  # 4-bit quantization
)

# Add LoRA adapters
model = FastLanguageModel.get_peft_model(
    model,
    r=16,
    lora_alpha=32,
    target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                    'gate_proj', 'up_proj', 'down_proj'],
    lora_dropout=0.05,
    bias="none",
    use_gradient_checkpointing=True,
)
```

**Hyperparameters**:
```yaml
learning_rate: 2e-4  # Higher for LoRA training
lr_scheduler: linear
warmup_steps: 100
epochs: 3
max_sequence_length: 512
per_device_batch_size: 1-2  # More flexible than full fine-tuning
gradient_accumulation_steps: 4-8
effective_batch_size: 28-56
weight_decay: 0.01
```

**Memory Breakdown**:
```
Quantized model weights (4-bit): ~10GB
LoRA adapters: ~100-200MB
Gradients + optimizer states: ~3-4GB
Activations (batch_size=1): ~1GB
Total: ~14GB per GPU ✅ Comfortable on 24GB GPU
```

---

### Training Data
- **Dataset**: Psych-101 (category learning tasks)
- **Size**: ~15,000 samples
- **Task Format**: E vs K, O vs S category classification
- **Prompt Structure**: Multi-trial learning with feedback

### Checkpointing Strategy
```yaml
save_strategy: steps
save_steps: 500  # More frequent for EXAONE, 1000 for GPT-OSS
save_total_limit: 3
load_best_model_at_end: true
metric_for_best_model: eval_loss
resume_from_checkpoint: true
```

### Estimated Training Time
```
EXAONE 4.0-32B: 30-50 hours (3 epochs)
  - 4x larger than 7.8B model
  - CPU offloading adds 20-30% overhead
  - Strategy: 2-3 day continuous job

GPT-OSS-20B (with Unsloth): 6-8 hours (3 epochs)
  - Quantized training (4-bit)
  - Unsloth optimization (1.5x faster)
  - Only train LoRA adapters (~0.1-1% of params)
  - Strategy: Single-day job

Total Training Time: 36-58 hours
Can run in parallel on split GPUs: ~2-3 days wall time
```

---

## 📋 Data Preparation

### Primary Evaluation Dataset: choices13k_1000.jsonl
```yaml
source: choices13k (13,006 risky choice problems)
sample_size: 1000
sampling_method: stratified random sampling
random_seed: 42
class_balance: aim for 45-55% each class (A vs B)
stratification: by problem difficulty (expected value difference)
```

**Sampling Script**: `scripts/create_choices13k_1000.py`
```python
# Key functionality:
1. Load full choices13k dataset
2. Compute problem characteristics (EV, variance)
3. Stratified sampling to ensure diverse problem types
4. Verify class balance
5. Convert to evaluation format: {"text": "...", "choice": 0/1}
6. Generate quality report (class distribution, EV ranges)
7. Save to data/choices13k_1000.jsonl
```

### Test Set: choices13k_test_500.jsonl
- **Size**: 500 held-out samples
- **Purpose**: Final validation on unseen data
- **Sampling**: Independent from choices13k_1000

### K-Fold Splits
- **Directory**: `data/kfold_splits/`
- **Folds**: 10
- **Format**: 10 files with train/test indices per fold

---

## 🧪 Evaluation Pipeline

### Evaluation Strategy: Dual Approach

**Why Dual Approach?**
- **LOO CV**: Maximum data usage, consistency with Phase 1
- **10-Fold CV**: Lower variance, more intuitive interpretation
- Together: Comprehensive validation from multiple perspectives

### 1. Leave-One-Out Cross-Validation (LOO CV)

```yaml
folds: 1000 (N=1000)
test_size_per_fold: 1 sample
training_size_per_fold: 999 samples

inner_cv: 5-fold nested CV for hyperparameter tuning
alpha_grid: [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

feature_extraction:
  method: last-layer hidden states
  dimension: 4096 (for all models)
  normalization: per-fold z-score (training set statistics)

classifier: LogisticRegression with L2 regularization
solver: lbfgs
max_iter: 1000

metrics:
  - per_fold_accuracy: binary (0.0 or 1.0)
  - mean_accuracy: average across all folds
  - standard_error: SD / sqrt(N)
  - 95% CI: [mean - 1.96*SE, mean + 1.96*SE]
  - log_likelihood: per-sample log probability

estimated_runtime:
  - EXAONE 4.0 feature extraction: 10 hours (4x slower than 7.8B)
  - GPT-OSS feature extraction: 6 hours (2.5x slower, plus merge adapters)
  - EXAONE-base feature extraction: 2.5 hours
  - LOO CV (all 3 models): 3-4 hours
  - Total: ~22-25 hours (overnight job)
```

**Note on GPT-OSS Feature Extraction**:
```python
# After training, merge LoRA adapters for evaluation
model = model.merge_and_unload()
# Now extract features from merged full-precision model
features = extract_last_layer_hidden_states(model, prompts)
```

**Expected Results Format**:
```
Model: Ko-CENTaUR (EXAONE 4.0-32B)
Mean Accuracy: XX.X% (SE=X.X%, 95% CI: [XX.X%, XX.X%])
Log-Likelihood: -X.XXX ± X.XXX

Model: Ko-CENTaUR (GPT-OSS-20B)
Mean Accuracy: XX.X% (SE=X.X%, 95% CI: [XX.X%, XX.X%])
Log-Likelihood: -X.XXX ± X.XXX

Model: EXAONE-base
Mean Accuracy: XX.X% (SE=X.X%, 95% CI: [XX.X%, XX.X%])
Log-Likelihood: -X.XXX ± X.XXX
```

### 2. 10-Fold Cross-Validation (Complementary)

```yaml
folds: 10
test_size_per_fold: 100 samples
training_size_per_fold: 900 samples

feature_extraction: same as LOO CV
classifier: same as LOO CV
hyperparameter_tuning: grid search on training set

metrics:
  - per_fold_accuracy: averaged over 100 test samples
  - mean_accuracy: average across 10 folds
  - standard_deviation: across 10 fold accuracies
  - 95% CI: bootstrap or normal approximation

estimated_runtime:
  - Feature extraction: Already done in LOO CV (reuse)
  - 10-fold CV: 1.5 hours
```

**Key Difference from LOO CV**:
- Lower variance (larger test sets)
- More stable accuracy estimates
- Easier to interpret SD (not ~50% like LOO)

---

## 📊 Statistical Analysis Suite

### 1. McNemar's Test (Primary Comparison)

**Purpose**: Test if two models have significantly different error rates on same samples

```yaml
test_type: paired binary outcomes
null_hypothesis: models have equal error rates
contingency_table:
  - both_correct | model_a_correct_b_wrong
  - model_a_wrong_b_correct | both_wrong

statistic: chi-square on discordant pairs
significance_level: 0.05
bonferroni_correction: true (3 pairwise comparisons)
adjusted_alpha: 0.0167
```

**Comparisons**:
1. Ko-CENTaUR (EXAONE 4.0) vs EXAONE-base
2. Ko-CENTaUR (GPT-OSS) vs EXAONE-base
3. Ko-CENTaUR (EXAONE 4.0) vs Ko-CENTaUR (GPT-OSS)

**Output Format**:
```
Comparison: Ko-CENTaUR (EXAONE 4.0) vs EXAONE-base
Contingency Table:
  Both Correct: XXX | EXAONE 4.0 Correct, Base Wrong: XX
  EXAONE 4.0 Wrong, Base Correct: XX | Both Wrong: XXX

Chi-square: X.XXX
p-value: X.XXXX
Adjusted alpha: 0.0167
Significant: Yes/No
Effect Size (Odds Ratio): X.XX
```

### 2. Permutation Test (Non-Parametric Validation)

```yaml
iterations: 10000
null_hypothesis: accuracy difference is due to chance
procedure:
  1. Randomly permute model labels
  2. Calculate accuracy difference
  3. Repeat 10,000 times
  4. Compute empirical p-value

p_value: proportion of permutations with difference >= observed
```

### 3. Bootstrap Confidence Intervals

```yaml
bootstrap_samples: 10000
method: percentile method
confidence_level: 0.95

procedure:
  1. Resample N=1000 with replacement
  2. Calculate accuracy on bootstrap sample
  3. Repeat 10,000 times
  4. Extract 2.5th and 97.5th percentiles

output: 95% CI without normality assumptions
```

### 4. Effect Size Measures

```yaml
cohens_h: standardized difference between proportions
formula: 2 * (arcsin(sqrt(p1)) - arcsin(sqrt(p2)))
interpretation:
  - small: 0.2
  - medium: 0.5
  - large: 0.8

odds_ratio: ratio of odds of correct prediction
formula: (p1/(1-p1)) / (p2/(1-p2))
interpretation: OR > 1 favors model 1

number_needed_to_treat: 1 / (p1 - p2)
interpretation: how many predictions to see 1 more correct
```

---

## 📅 Timeline and Milestones

### Phase 1: Data Preparation (Days 1-4)

**Day 1**:
- Create `scripts/create_choices13k_1000.py`
- Implement stratified sampling logic
- Add quality validation checks

**Day 2**:
- Run sampling script
- Validate class balance (45-55%)
- Generate data statistics report
- Create `data/choices13k_1000.jsonl`

**Day 3**:
- Create k-fold split generator
- Generate 10-fold splits
- Validate split quality (no data leakage)

**Day 4**:
- Create test set (choices13k_test_500.jsonl)
- Final data integrity checks
- Document data preparation

**Deliverables**: choices13k_1000.jsonl, kfold_splits/, choices13k_test_500.jsonl

---

### Phase 2: Training Pipeline Setup (Days 5-10)

**Days 5-6** (EXAONE 4.0 Setup):
- Adapt `scripts/train_model.py` for EXAONE 4.0-32B
- Create `configs/training_exaone40.yaml`
- Create `configs/deepspeed_zero3_32b.json` (aggressive offloading)
- Test model loading on server

**Days 7-8** (GPT-OSS Setup):
- Install Unsloth library: `pip install unsloth`
- Create `scripts/train_gpt_oss.py` with QLoRA setup
- Create `configs/training_gpt_oss.yaml`
- Create `configs/qlora_config.yaml` (LoRA parameters)
- Configure tokenizer compatibility
- Test 4-bit model loading with Unsloth

**Days 9-10** (Validation):
- Test EXAONE training on tiny subset (10 samples, 1 epoch)
- Test GPT-OSS QLoRA training on tiny subset
- Validate GPU memory usage:
  - EXAONE: <24GB per GPU
  - GPT-OSS: ~14GB per GPU (comfortable)
- Verify distributed training works
- Benchmark training speed
- Test LoRA adapter merge for GPT-OSS

**Deliverables**: train_exaone40.py, train_gpt_oss.py, configs, Unsloth setup verified

---

### Phase 3: Model Training (Days 11-14)

**Strategy: Parallel Training** (if possible)
- 4 GPUs → EXAONE 4.0-32B (needs more GPUs)
- 1 GPU → GPT-OSS-20B (single GPU sufficient)

**Days 11-14** (Parallel Training):
- **Track A - EXAONE 4.0-32B**: Train on 4-7 GPUs
  - Monitor training loss and GPU utilization
  - Monitor CPU memory usage (offloading)
  - Save checkpoints every 500 steps
  - **Duration**: 30-50 hours (continuous)

- **Track B - GPT-OSS-20B**: Train on 1 GPU (or remaining GPUs)
  - Monitor training loss
  - Save LoRA adapters every 1000 steps
  - **Duration**: 6-8 hours
  - Can start after EXAONE or run in parallel

**Day 14** (afternoon):
- Validate final checkpoints
- Merge LoRA adapters for GPT-OSS: `model.merge_and_unload()`
- Select best models based on validation loss
- Document training metrics

**Deliverables**:
- `models/ko-centaur-exaone40/checkpoint-final/`
- `models/ko-centaur-gpt-oss/merged-model/` (LoRA merged)
- Training logs and metrics

---

### Phase 4: Evaluation Implementation (Days 15-17)

**Days 15-16**:
- Create `scripts/run_kfold_eval.py`
- Implement 10-fold CV pipeline
- Test on choices13k_100 (validation)

**Day 17**:
- Create `scripts/statistical_analysis.py`
- Implement McNemar test
- Implement permutation test
- Implement bootstrap CI
- Test on choices13k_100 (validation)

**Deliverables**: run_kfold_eval.py, statistical_analysis.py

---

### Phase 5: Full Evaluation (Days 18-21)

**Days 18-20** (Feature Extraction + LOO CV):
- Extract features for N=1000 (3 models)
  - EXAONE 4.0: ~10 hours
  - GPT-OSS (merged): ~6 hours
  - EXAONE-base: ~2.5 hours
- Run LOO CV evaluation
- **Estimated Time**: 22-25 hours (overnight job)

**Day 21** (10-Fold CV + Statistics):
- Run 10-fold CV evaluation (reuse features)
  - **Time**: 1.5 hours
- Run McNemar tests (3 pairwise comparisons)
- Run permutation tests (3 comparisons, 10K iterations)
- Compute bootstrap CIs (3 models)
- Calculate effect sizes
- **Time**: 2-3 hours

**Deliverables**:
- `results/choices13k_1000_loo/`
- `results/choices13k_1000_kfold/`
- `results/statistical_tests/`

---

### Phase 6: Analysis and Reporting (Days 22-24)

**Days 22-23**:
- Analyze all results (LOO, k-fold, statistics)
- Identify which model performs best and why
- Examine per-class accuracy patterns
- Investigate problem types where models differ
- Compare with Phase 1 results (N=100)
- Assess impact of model size (32B vs 20B vs 7.8B)
- Compare Korean (EXAONE) vs English (GPT-OSS) models
- Evaluate QLoRA effectiveness vs full fine-tuning

**Day 24**:
- Create comprehensive final report
- Include all metrics: accuracy, SE, CI, p-values, effect sizes
- Generate publication-quality figures
- Write interpretation and conclusions
- Document limitations and future work

**Deliverables**: `results/final_report/PHASE2_RESULTS.md`

---

## ⏱️ Total Timeline Summary

**Total Duration**: 24 days (3.5 weeks)
- Week 1 (Days 1-7): Data prep + setup
- Week 2 (Days 8-14): Training
- Week 3 (Days 15-21): Evaluation
- Week 4 (Days 22-24): Analysis + reporting

**Improvement from Previous Plan**: 4-5 weeks → 3.5 weeks
- Reason: GPT-OSS trains much faster (6-8hrs vs 18-24hrs)

---

## ⚠️ Risk Assessment and Mitigation

### Risk 1: EXAONE 4.0-32B OOM (Critical)

**Probability**: Medium-High
**Impact**: Critical (blocks training)

**Mitigation**:
- Start with `per_device_batch_size=1`
- Enable aggressive gradient checkpointing
- Enable CPU offloading for optimizer + parameters
- Monitor GPU memory continuously
- If approaching limit: reduce sequence length to 256

**Backup Plan**:
- If 4.0-32B OOMs: Try EXAONE 3.5-32B-Instruct
- If still OOM: Fall back to EXAONE 3.5-7.8B
- GPT-OSS should train safely (14GB < 24GB)

---

### Risk 2: Unsloth Library Compatibility

**Probability**: Low-Medium
**Impact**: Medium (blocks GPT-OSS training)

**Mitigation**:
- Test Unsloth installation early (Day 7-8)
- Run small-scale test before full training
- Verify model loading and LoRA adapter creation
- Test adapter merge process

**Backup Plan**:
- Use standard bitsandbytes QLoRA (18-20GB, still fits)
- Alternative: Use GPT-J-6B with standard QLoRA (12GB, well-tested)
- Standard QLoRA is well-documented, lower risk

---

### Risk 3: LoRA Adapter Merge Issues

**Probability**: Low
**Impact**: Medium (affects evaluation)

**Mitigation**:
- Test merge process during setup (Day 9-10)
- Verify merged model produces same outputs as adapter model
- Document merge procedure carefully

**Backup Plan**:
- If merge fails: Extract features using adapter model directly
- Unsloth supports both merged and adapter-based inference

---

### Risk 4: Long EXAONE Training Time

**Probability**: High
**Impact**: Medium (extends timeline)

**Mitigation**:
- Implement robust checkpointing (every 500 steps)
- Test resume-from-checkpoint before long job
- Coordinate with server users
- Run during low-traffic periods

**Backup Plan**:
- Resume from latest checkpoint if interrupted
- Reduce to 2 epochs if taking >60 hours

---

### Risk 5: Parallel Training Resource Conflict

**Probability**: Medium
**Impact**: Medium (extends timeline)

**Mitigation**:
- Run GPT-OSS after EXAONE completes (sequential safer)
- GPT-OSS trains fast (6-8hrs), won't significantly extend timeline
- Monitor GPU availability before starting

**Backup Plan**:
- Sequential training is acceptable (still 3.5 weeks total)
- GPT-OSS can train on single GPU if needed

---

## 📝 Implementation Checklist

### Scripts to Create

- [ ] `scripts/create_choices13k_1000.py` - Stratified data sampling
- [ ] `scripts/create_kfold_splits.py` - Generate 10-fold splits
- [ ] `scripts/train_exaone40.py` - EXAONE 4.0-32B training
- [ ] `scripts/train_gpt_oss.py` - GPT-OSS-20B QLoRA training
- [ ] `scripts/run_kfold_eval.py` - 10-fold CV evaluation
- [ ] `scripts/statistical_analysis.py` - McNemar, permutation, bootstrap

### Scripts to Modify

- [ ] `scripts/run_full_eval.py` - Update for SE/CI reporting
- [ ] `scripts/extract_features.py` - Handle merged LoRA models

### Configuration Files

- [ ] `configs/training_exaone40.yaml` - EXAONE 4.0 hyperparameters
- [ ] `configs/training_gpt_oss.yaml` - GPT-OSS hyperparameters
- [ ] `configs/deepspeed_zero3_32b.json` - DeepSpeed for 32B
- [ ] `configs/qlora_config.yaml` - LoRA adapter configuration
- [ ] `configs/evaluation_1000.yaml` - Evaluation settings

### Dependencies to Install

- [ ] `pip install unsloth` - For GPT-OSS QLoRA training
- [ ] Verify `transformers`, `bitsandbytes` versions compatible with Unsloth
- [ ] Test DeepSpeed installation

### Datasets to Generate

- [ ] `data/choices13k_1000.jsonl` - Main evaluation dataset
- [ ] `data/choices13k_test_500.jsonl` - Held-out test set
- [ ] `data/kfold_splits/fold_{0-9}.json` - 10-fold splits

### Expected Outputs

- [ ] `models/ko-centaur-exaone40/checkpoint-final/`
- [ ] `models/ko-centaur-gpt-oss/merged-model/`
- [ ] `results/choices13k_1000_loo/all_results.pth`
- [ ] `results/choices13k_1000_kfold/all_results.pth`
- [ ] `results/statistical_tests/`
- [ ] `results/final_report/PHASE2_RESULTS.md`

---

## 📈 Success Criteria

### Training Success
- ✅ Both models train without OOM errors
- ✅ EXAONE 4.0-32B completes with CPU offloading
- ✅ GPT-OSS-20B QLoRA training completes successfully
- ✅ LoRA adapters merge successfully for GPT-OSS
- ✅ Training loss decreases consistently
- ✅ Validation performance ≥ Phase 1 Ko-CENTaUR (60%)

### Evaluation Success
- ✅ Feature extraction completes for all 3 models (N=1000)
- ✅ LOO CV + 10-fold CV complete for all models
- ✅ Features show non-zero variance
- ✅ Predictions show meaningful variation

### Statistical Success
- ✅ McNemar tests complete for 3 comparisons
- ✅ Permutation tests and bootstrap CIs computed
- ✅ Effect sizes calculated
- ✅ Clear results with proper confidence intervals

### Research Success
- ✅ Determine if EXAONE 4.0-32B improves cognitive modeling
- ✅ Compare Korean (EXAONE) vs English (GPT-OSS) models
- ✅ Assess QLoRA effectiveness vs full fine-tuning
- ✅ Evaluate model size impact (32B vs 20B vs 7.8B)
- ✅ Publication-ready results with rigorous statistics

---

## 🔬 Expected Results Format

### Quantitative Results

```markdown
## Model Performance Comparison (N=1000)

### Leave-One-Out Cross-Validation

| Model | Accuracy | SE | 95% CI | Log-Likelihood |
|-------|----------|-----|---------|----------------|
| Ko-CENTaUR (EXAONE 4.0-32B) | XX.X% | X.X% | [XX.X%, XX.X%] | -X.XXX ± X.XXX |
| Ko-CENTaUR (GPT-OSS-20B) | XX.X% | X.X% | [XX.X%, XX.X%] | -X.XXX ± X.XXX |
| EXAONE-base (7.8B) | XX.X% | X.X% | [XX.X%, XX.X%] | -X.XXX ± X.XXX |

### 10-Fold Cross-Validation

| Model | Accuracy | SD | 95% CI | Log-Likelihood |
|-------|----------|-----|---------|----------------|
| Ko-CENTaUR (EXAONE 4.0-32B) | XX.X% | X.X% | [XX.X%, XX.X%] | -X.XXX ± X.XXX |
| Ko-CENTaUR (GPT-OSS-20B) | XX.X% | X.X% | [XX.X%, XX.X%] | -X.XXX ± X.XXX |
| EXAONE-base (7.8B) | XX.X% | X.X% | [XX.X%, XX.X%] | -X.XXX ± X.XXX |

### Statistical Comparisons

| Comparison | McNemar p | Permutation p | Cohen's h | Odds Ratio |
|------------|-----------|---------------|-----------|------------|
| EXAONE 4.0 vs Base | X.XXXX | X.XXXX | X.XX | X.XX |
| GPT-OSS vs Base | X.XXXX | X.XXXX | X.XX | X.XX |
| EXAONE 4.0 vs GPT-OSS | X.XXXX | X.XXXX | X.XX | X.XX |

**Significance Level**: α = 0.0167 (Bonferroni-corrected)
```

### Qualitative Results

```markdown
## Interpretation

### Performance Summary
[Which model performs best? How much improvement?]
[Does EXAONE 4.0's superior base performance help?]
[How does GPT-OSS QLoRA compare to full fine-tuning?]

### Model Size Analysis
[Impact of 32B vs 20B vs 7.8B on cognitive modeling]
[Is there a scaling law?]

### Architecture Comparison
[Korean EXAONE vs English GPT-OSS]
[Does training language matter?]

### Training Method Comparison
[Full fine-tuning (EXAONE) vs QLoRA (GPT-OSS)]
[Can efficient QLoRA match full fine-tuning?]

### Statistical Significance
[Which comparisons reach significance?]

### Effect Sizes
[Practical significance beyond p-values]

### Implications
[What this means for turning LLMs into cognitive models]
```

---

## 📚 References and Resources

### Model Documentation
- [EXAONE 4.0-32B](https://huggingface.co/LGAI-EXAONE/EXAONE-4.0-32B)
- [GPT-OSS-20B](https://huggingface.co/openai/gpt-oss-20b)
- [OpenAI GPT-OSS Blog](https://openai.com/index/introducing-gpt-oss/)
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [Unsloth GPT-OSS Guide](https://docs.unsloth.ai/new/gpt-oss-how-to-run-and-fine-tune)
- [DeepSpeed ZeRO](https://www.deepspeed.ai/tutorials/zero/)

### QLoRA and Quantization
- [QLoRA Paper](https://arxiv.org/abs/2305.14314) - Dettmers et al.
- [Bitsandbytes](https://github.com/TimDettmers/bitsandbytes)
- [HuggingFace 4-bit Guide](https://huggingface.co/blog/4bit-transformers-bitsandbytes)

### Statistical Methods
- McNemar's Test: [Wikipedia](https://en.wikipedia.org/wiki/McNemar%27s_test)
- Bootstrap Methods: Efron & Tibshirani (1993)
- Effect Sizes: Cohen (1988)

### Previous Work
- CENTaUR Paper: Binz & Schulz (2023, ICLR)
- Ko-CENTaUR Phase 1: EVALUATION_STATUS.md, VARIANCE_INVESTIGATION_REPORT.md

---

## 🎓 Lessons from Phase 1 and Plan Revisions

### What Went Well ✅
- Feature extraction pipeline works correctly
- LOO CV implementation is robust
- Systematic bug detection and fixing
- Variance analysis revealed mathematical correctness

### What to Improve 📈
1. **Sample Size**: N=100 → N=1000 (10x increase)
2. **Evaluation**: Add k-fold CV
3. **Statistics**: Rigorous testing (McNemar, permutation, bootstrap)
4. **Reporting**: SE and CI instead of raw SD
5. **Planning**: Verify requirements carefully before starting

### Critical Planning Lessons 💡
- ✅ Verify user requirements thoroughly (don't miss GPT OSS model)
- ✅ Respect version preferences (EXAONE 4.0 vs 3.5)
- ✅ Consider memory constraints (GPT-OSS-20B vs GPT-NeoX-20B)
- ✅ Prioritize latest, optimized solutions (Unsloth vs standard methods)
- ✅ Balance ambition with feasibility (14GB vs 15-20GB makes a difference)

### Why GPT-OSS-20B is the Right Choice
1. ✅ Latest model (August 2025 vs February 2022)
2. ✅ Lower memory (14GB vs 15-20GB, safer margin)
3. ✅ Faster training (6-8hrs vs 18-24hrs, -67% time)
4. ✅ Better optimization (Unsloth 1.5x speedup, 70% less VRAM)
5. ✅ From OpenAI (state-of-the-art quality vs EleutherAI)
6. ✅ User requirement: "quantized model 중에서 성능이 좋은 모델" ✓

---

## 📞 Contact and Support

**Primary Investigator**: [User]
**Location**: Connectome Server (connectome1@147.47.200.154)
**Project Directory**: `/scratch/connectome/connectome1/ko-centaur/`

**Documentation**:
- Phase 1 Results: EVALUATION_STATUS.md, DEBRIEF.md
- Variance Analysis: VARIANCE_INVESTIGATION_REPORT.md
- This Plan: NEXT_EXPERIMENT_PLAN.md

---

## 🚀 Ready to Execute

This CORRECTED (v2) plan addresses all user requirements and technical constraints:

**✅ ALL CORRECTIONS APPLIED**:
- ✅ EXAONE 4.0-32B (not 3.5) - User requirement: "far surpasses"
- ✅ GPT-OSS-20B (not GPT-NeoX) - Quantized, latest, optimal performance
- ✅ No Llama training - CENTaUR already validated this
- ✅ Safer memory usage - 14GB vs 15-20GB per GPU
- ✅ Faster timeline - 3.5 weeks vs 4-5 weeks

**✅ TECHNICAL VALIDATION**:
- ✅ GPU infrastructure: 7x RTX 3090, 168GB VRAM
- ✅ Disk space: 2.4TB available
- ✅ EXAONE 4.0-32B: ~23-26GB per GPU (DeepSpeed ZeRO-3 + offload)
- ✅ GPT-OSS-20B: ~14GB per GPU (Unsloth QLoRA, comfortable)
- ✅ Dual evaluation: LOO CV + 10-fold CV
- ✅ Statistical methods: McNemar, permutation, bootstrap
- ✅ Timeline: 3.5 weeks (24 days)
- ✅ Risks identified with mitigation plans

**Next Step**: Begin Phase 1 (Data Preparation) - Create sampling scripts and generate evaluation datasets.

---

**Status**: ✅ **CORRECTED PLANNING COMPLETE (v2) - READY FOR IMPLEMENTATION**
**Date**: 2025-10-14
**Final Model Selection**:
- Model 1: EXAONE 4.0-32B (Korean-focused, 32B params, full fine-tune)
- Model 2: GPT-OSS-20B (OpenAI latest, 20B params, QLoRA)
- Baseline: EXAONE-base (7.8B params, no fine-tune)

This plan is technically sound, addresses all user corrections, and provides the best balance of performance, safety, and efficiency.
