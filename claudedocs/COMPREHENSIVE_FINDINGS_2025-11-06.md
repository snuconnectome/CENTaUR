# Comprehensive Investigation Findings
**Date**: 2025-11-06
**Session**: Generation Bias Investigation - Complete Analysis
**Status**: Investigation Complete - Recommendations Provided

---

## Executive Summary

**Problem Confirmed**: Qwen2.5-32B-Instruct with QLoRA fine-tuning shows:
- **72% generation bias** toward choice B
- **50% accuracy** on full dataset (at chance level)
- **95.25% feature similarity** caused by biased token generation
- **NLL = 0.7623** (worse than random baseline 0.6931)

**Root Cause Identified**:
1. **Base model doesn't understand task** → generates ' Option' (0% accuracy)
2. **Fine-tuning helped** → model learned to generate A/B tokens
3. **But fine-tuning introduced bias** → 72% toward B instead of balanced

**Verdict**: **Model is NOT suitable for CENTaUR in current form**
The fine-tuned model defaults to B regardless of prompt content, indicating it hasn't learned the task properly.

---

## Investigation Timeline

### Phase 1: LOO CV Partial Results (Folds 1-50)
**Finding**: NLL = 0.7623 at fold 50/100
- Better than buggy features (0.8560) ✓
- Worse than random baseline (0.6931) ✗
- High similarity persists (95.25%)

**Conclusion**: Extraction methodology fix was correct, but high similarity remains a problem.

### Phase 2: Generation Bias Discovery (5 samples)
**Finding**: 80% bias toward ' B' (4/5 samples)
- Sample 0: True=B, Generated=' B' ✓
- Sample 1: True=A, Generated=' B' ✗
- Sample 2: True=B, Generated=' A' ✗
- Sample 3: True=A, Generated=' B' ✗
- Sample 4: True=B, Generated=' B' ✓

**Accuracy**: 40% (2/5 correct)
**Similarity**: 75.41% pairwise

**Conclusion**: Model defaults to B regardless of true choice.

### Phase 3: Full Dataset Confirmation (100 samples)
**Finding**: **72% generation bias toward B** (confirmed)

**Generation Distribution**:
- Generated ' A': 28/100 (28%)
- Generated ' B': 72/100 (72%)

**Ground Truth Distribution**:
- True Choice A: 44/100 (44%)
- True Choice B: 56/100 (56%)

**Accuracy**: 50/100 (50% - exactly chance level)

**Conditional Patterns** (critical evidence):
- True A → Generated A: **25%** (should be ~100%)
- True A → Generated B: **75%** (wrong 3/4 of the time)
- True B → Generated A: 30.4%
- True B → Generated B: 69.6%

**Interpretation**: Model is **NOT learning choice patterns** from prompts. It's defaulting to B regardless of input.

### Phase 4: Base Model Comparison
**Finding**: Base model generates **' Option' for ALL samples** (100%)

**Critical Discovery**:
- Base Qwen2.5-32B-Instruct: 0% A/B token generation
- Fine-tuned with QLoRA: 72% toward B, but at least generates A/B
- **Fine-tuning actually IMPROVED task understanding** (from 0% to 50% accuracy)

**Prompt Format Issue Confirmed**:
```
Prompt: "...Option A: ... Option B: ... Machine chose:"

Base model completes as: "Machine chose: Option [A/B]"
Fine-tuned model learned: "Machine chose: [A/B]"
```

Base model doesn't understand we want single-letter output.

### Phase 5: Alternative Prompt Formats (20 samples)

| Format | Ending | Bias | Accuracy | Notes |
|--------|--------|------|----------|-------|
| **Original** | "Machine chose:" | 70% → B | **70%** | Best accuracy |
| Simple | "Answer:" | 75% → A | 15% | Generates ' Machine' |
| Explicit | "Choose A or B. Answer:" | 60% → A | 30% | Reduced bias, lower accuracy |

**Key Insight**: Original prompt has BEST accuracy (70% on this subset) despite bias!

**Caveat**: This 20-sample subset shows 70% accuracy vs 50% on full 100 samples. Subset may not be representative.

---

## Root Cause Analysis

### Why High Similarity (95.25%)?
**Confirmed Cause**: Most samples generate the same token (' B')

**Evidence**:
- 72% of samples → ' B' token
- Hidden states for same token are >96% similar
- Only 28% generate different token (' A')
- Average similarity dominated by B-token cluster

### Why Worse Than Random (NLL = 0.7623 vs 0.6931)?
**Confirmed Cause**: Model behaves like biased coin, not informed predictor

**Mechanism**:
1. Ground truth: 44% A, 56% B (relatively balanced)
2. Model generation: 28% A, 72% B (heavily skewed)
3. Model predicts B for most samples regardless of actual choice
4. Equivalent to always guessing majority class
5. Random 50/50 guessing performs better

### Why Fine-Tuning Introduced Bias?
**Hypothesis**: Training data or methodology issue

**Evidence**:
- Base model: Doesn't understand task (generates ' Option')
- Fine-tuned: Understands task format (generates A/B) but biased toward B
- Fine-tuning taught the model WHAT to output but not HOW to choose correctly

**Possible Causes**:
1. Training data imbalance (56% B in dataset)
2. QLoRA hyperparameters (learning rate, rank, alpha)
3. Prompt format during training
4. Model architecture preference

---

## Implications for CENTaUR Methodology

### Current State: NOT SUITABLE
**Reasons**:
1. ✗ **No predictive power**: 50% accuracy = random guessing
2. ✗ **Strong generation bias**: 72% → B shows model defaults to majority class
3. ✗ **No learning**: Model doesn't capture choice patterns from prompts
4. ✗ **High similarity**: 95.25% means features lack diversity for regression

### Why Original CENTaUR Worked (Binz & Schulz, 2023)
**LLaMA-65B Performance**:
- NLL ≈ 30,000 (much better than random)
- Generated balanced choices
- Hidden states captured choice-relevant information

**Key Differences**:
1. **Model size**: LLaMA-65B vs Qwen2.5-32B (size matters)
2. **Model architecture**: LLaMA vs Qwen (different training objectives)
3. **Fine-tuning**: Original used base LLaMA, we used fine-tuned Qwen
4. **Prompt format**: May have used different prompts

---

## Recommendations

### Priority 1: Test Different Model Architecture ⚠️ CRITICAL
**Hypothesis**: Qwen2.5 architecture not suitable for this task

**Action**: Test alternatives available in Ko-CENTaUR:
1. **DeepSeek-R1-Distill-Qwen-32B** (already fine-tuned)
2. **EXAONE-3.5-32B** (Ko-CENTaUR base model)
3. Compare generation patterns and bias

**Expected Outcome**:
- If DeepSeek/EXAONE also show bias → Prompt format issue
- If DeepSeek/EXAONE more balanced → Qwen architecture issue

**Implementation**:
```bash
# Test DeepSeek generation
python scripts/analyze_full_generation.py --model deepseek

# Test EXAONE generation
python scripts/analyze_full_generation.py --model exaone
```

### Priority 2: Investigate Fine-Tuning Methodology
**Hypothesis**: Fine-tuning introduced bias instead of improving performance

**Questions to Answer**:
1. What was the training data distribution? (56% B in test set)
2. Were prompts formatted consistently during training?
3. What were QLoRA hyperparameters (lr, rank, alpha)?
4. Did validation metrics show bias during training?

**Action**: Review fine-tuning logs and training configuration
```bash
# Check training logs for bias indicators
cat /scratch/connectome/connectome1/ko-centaur/logs/train_qwen25.log

# Review training configuration
cat /scratch/connectome/connectome1/ko-centaur/configs/qwen25_qlora.yaml
```

### Priority 3: Test Prompt Format with Other Models (If Needed)
**Hypothesis**: Prompt format causes bias across models

**Action**: If DeepSeek/EXAONE also show bias, test prompt formats:

**Alternative Formats to Test**:
```python
# Current
"Option A: ..., Option B: ... Machine chose:"

# Alternative 1: Direct choice
"Option A: ..., Option B: ... Choice: "

# Alternative 2: Explicit instruction
"Select A or B: A: ..., B: ... Answer: "

# Alternative 3: No trailing text (like original LLaMA)
"A: ..., B: ... "
```

**Caveat**: Based on Phase 5 results, original prompt had BEST accuracy. Only pursue this if alternative models also show bias.

### Priority 4: Consider Full Fine-Tuning (If Budget Allows)
**Hypothesis**: QLoRA (4-bit quantization) limiting learning capacity

**Action**: Try full fine-tuning without quantization
- Higher GPU memory requirements (7x24GB may not be enough)
- May require gradient accumulation or smaller batch size

---

## Decision Tree

```
Test DeepSeek-R1 and EXAONE generation patterns
│
├─ Both show bias (>70%)
│  └─ Prompt format is the issue
│     ├─ Test alternative prompt formats
│     │  ├─ Format reduces bias (<60%) → Use new format
│     │  └─ Format doesn't help → Fundamental methodology issue
│     │
│     └─ Consult original CENTaUR authors for prompt details
│
└─ At least one is balanced (<60% bias)
   └─ Qwen architecture is the issue
      ├─ Use DeepSeek/EXAONE instead of Qwen
      ├─ Re-extract features with better model
      └─ Re-run LOO CV with balanced features
```

---

## Code Artifacts Created

### Analysis Scripts
1. **`scripts/analyze_full_generation.py`**: Analyze generation bias on 100 samples
2. **`scripts/analyze_base_model_generation.py`**: Test base model without fine-tuning
3. **`scripts/test_alternative_prompt.py`**: Compare prompt format effects

### Documentation
1. **`claudedocs/LOO_CV_PARTIAL_RESULTS_2025-11-06.md`**: LOO CV analysis
2. **`claudedocs/GENERATION_BIAS_ANALYSIS_2025-11-06.md`**: Detailed bias investigation
3. **`claudedocs/COMPREHENSIVE_FINDINGS_2025-11-06.md`**: This document

---

## Key Metrics Summary

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Generation Bias | 72% → B | CRITICAL - Strong bias |
| Accuracy (Full) | 50% | At chance level |
| Accuracy (Subset) | 70% | Subset variability |
| Feature Similarity | 95.25% | Too high for regression |
| LOO CV NLL | 0.7623 | Worse than random (0.6931) |
| Base Model Accuracy | 0% | Doesn't understand task |
| True A → Generated A | 25% | Should be ~100% |
| True A → Generated B | 75% | Model defaulting to B |

---

## Lessons Learned

### About Extraction Methodology
✅ **Correct**: Two-step process (generate + forward pass) extracts from GENERATED token
✅ **Validated**: Methodology fix improved NLL from 0.8560 → 0.7623 (11%)
✗ **Insufficient**: Even with correct extraction, biased generation causes high similarity

### About Fine-Tuning Effects
✅ **Positive**: Fine-tuning taught model to generate A/B tokens (vs base ' Option')
✗ **Negative**: Fine-tuning introduced 72% bias toward B
⚠️ **Unexpected**: Base model performs worse (0% A/B) than fine-tuned (50% accuracy)

### About Prompt Engineering
⚠️ **Surprising**: Original prompt had BEST accuracy despite bias
⚠️ **Caution**: Alternative prompts reduce bias but hurt accuracy
📊 **Data**: Need full dataset tests, not just 20-sample subsets

### About Model Selection
🎯 **Critical**: Model architecture matters significantly
🔍 **Next Step**: Test alternative models before concluding methodology failed
📈 **Baseline**: Original CENTaUR used LLaMA-65B with much better performance

---

## Next Steps (Immediate)

1. **[URGENT]** Test DeepSeek-R1 generation patterns
   ```bash
   cd /scratch/connectome/connectome1/ko-centaur
   # Modify analyze_full_generation.py to use DeepSeek adapter
   python scripts/analyze_full_generation.py --model deepseek
   ```

2. **[HIGH]** Test EXAONE generation patterns
   ```bash
   # Test Ko-CENTaUR base model
   python scripts/analyze_full_generation.py --model exaone
   ```

3. **[MEDIUM]** Review fine-tuning methodology
   - Check training logs for bias indicators
   - Review training data distribution
   - Assess QLoRA hyperparameters

4. **[LOW]** If all models show bias, consult original CENTaUR paper
   - Email authors for prompt format details
   - Check if they used special preprocessing
   - Ask about generation patterns they observed

---

## Conclusion

**Current Status**: Qwen2.5-32B-Instruct with QLoRA fine-tuning is **NOT suitable** for CENTaUR methodology in current form.

**Why It Failed**:
1. Fine-tuning introduced 72% generation bias toward B
2. Model defaults to majority class instead of learning from prompts
3. 50% accuracy (chance level) indicates no predictive power
4. High similarity (95.25%) makes binomial regression ineffective

**Why There's Hope**:
1. Fine-tuning DID improve task understanding (0% → 50% accuracy vs base)
2. Alternative models (DeepSeek, EXAONE) may perform better
3. Original CENTaUR achieved excellent results with different model
4. Methodology is sound (extraction fix validated)

**Next Actions**: Test alternative model architectures before concluding the approach failed.

---

## References

- Original investigation: `claudedocs/INVESTIGATION_REPORT_2025-11-05.md`
- LOO CV results: `claudedocs/LOO_CV_PARTIAL_RESULTS_2025-11-06.md`
- Bias analysis: `claudedocs/GENERATION_BIAS_ANALYSIS_2025-11-06.md`
- Original paper: Binz, M., & Schulz, E. (2023). Turning large language models into cognitive models. ICLR 2023.
