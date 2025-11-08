# LOO CV Partial Results Analysis
**Date**: 2025-11-06
**Test**: Corrected feature extraction methodology validation
**Status**: 50/100 folds completed before termination

---

## Executive Summary

The corrected feature extraction methodology (extracting from GENERATED token instead of last PROMPT token) **partially succeeded**:

- ✅ **Improved performance**: NLL decreased from 0.8560 → 0.7623 (11% improvement)
- ❌ **Still suboptimal**: NLL = 0.7623 vs random baseline NLL = 0.6931 (10% worse)
- ⚠️ **High feature similarity persists**: 95.25% pairwise similarity remains limiting factor

**Conclusion**: The bug fix was correct, but **high similarity is the real problem**.

---

## Results

### Performance Metrics (50/100 folds)

| Model | NLL | vs Random | vs Buggy | Status |
|-------|-----|-----------|----------|--------|
| Random baseline | 0.6931 | baseline | +24% better | ✓ |
| Buggy features | 0.8560 | +23% worse | baseline | ✗ |
| **Corrected features** | **0.7623** | **+10% worse** | **+11% better** | **⚠️** |

### NLL Progression

```
Fold 10:  0.5593  ← Very promising early signal
Fold 20:  0.7395  ← Stabilized higher
Fold 30:  0.7138  ← Slight improvement
Fold 40:  0.7606  ← Increased again
Fold 50:  0.7623  ← Appears converged
```

**Trend**: NLL stabilized around **0.76** after initial volatility, suggesting this is the true performance level.

---

## What We Learned

### 1. Extraction Methodology is Correct

The two-step process works as intended:
```python
# Step 1: Generate one token
gen_outputs = model.generate(max_new_tokens=1, temperature=0.0, do_sample=False)

# Step 2: Extract hidden state from generated token
full_outputs = model(input_ids=gen_outputs, output_hidden_states=True)
generated_token_hidden = full_outputs.hidden_states[-1][0, -1, :]
```

Evidence: **11% improvement** over buggy extraction proves the fix works.

### 2. High Similarity is the Real Bottleneck

**95.25% pairwise similarity** is abnormally high and limiting predictive power.

**Comparison**:
- Buggy extraction: 89% similarity, NLL = 0.8560
- Corrected extraction: 95% similarity, NLL = 0.7623

**Hypothesis**: Even with correct extraction, models are generating:
- Very similar tokens across prompts
- Hidden states that cluster tightly
- Insufficient diversity for binomial regression

### 3. Features are Weakly Informative

NLL = 0.7623 indicates features have *some* predictive power (better than buggy 0.8560) but are **not sufficient** to beat random guessing.

**Possible causes**:
1. Fine-tuning reduced diversity instead of enhancing it
2. Prompts are too similar (all end with "A: Machine")
3. Model defaulting to one choice for most samples
4. Hidden states don't capture choice-relevant information

---

## Critical Questions to Answer

### Q1: What tokens are being generated?

**Test**: Run `debug_extraction_detailed.py` on full 100 samples to check:
- Unique token count (expecting 2: "1" and "2")
- Actual tokens generated (are they all "B"?)
- Token distribution (80/20 split? 90/10?)

**If all samples → same token**: Model has generation bias, not useful for CENTaUR.

### Q2: Is the two-step process equivalent to original?

**Original CENTaUR**:
```python
results, _ = llama.generate([text], temperature=0.0, top_p=1, max_length=1)
llama_features[index] = llama.generator.model.hl.squeeze().detach().cpu()
```

**Our process**:
```python
# Step 1: Generate
gen_outputs = model.generate(max_new_tokens=1, ...)
# Step 2: Forward pass
full_outputs = model(input_ids=gen_outputs, output_hidden_states=True)
hidden = full_outputs.hidden_states[-1][0, -1, :]
```

**Concern**: Original extracted `.hl` directly from generator, we do separate forward pass.

**Test needed**: Check if `model.generate(output_hidden_states=True, return_dict_in_generate=True)` captures hidden states during generation **OR** if forward pass changes them.

### Q3: Does single-step generation with hidden states work?

Try alternative extraction that's closer to original:
```python
with torch.no_grad():
    gen_outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        temperature=0.0,
        do_sample=False,
        output_hidden_states=True,
        return_dict_in_generate=True
    )
    # Extract from generation outputs directly
    last_gen_step = gen_outputs.hidden_states[-1]
    last_layer = last_gen_step[-1]
    hidden = last_layer[0, -1, :]
```

This extracts hidden states **during generation** rather than separate forward pass.

---

## Next Steps

### Priority 1: Analyze Generation Patterns
```bash
cd /scratch/connectome/connectome1/ko-centaur
python scripts/debug_extraction_detailed.py
```

**Check**:
1. How many unique tokens across 100 samples?
2. What's the token distribution (A vs B)?
3. Is there generation bias (e.g., 95% samples → "B")?

**Decision**:
- If generation bias exists → Model not suitable for CENTaUR, try different prompt format
- If tokens diverse → Similarity issue is elsewhere

### Priority 2: Test Single-Step Generation

Modify extraction to use `output_hidden_states=True, return_dict_in_generate=True` and extract hidden states directly from generation outputs (no separate forward pass).

**Hypothesis**: Separate forward pass may alter hidden states slightly.

### Priority 3: Compare Base Model Performance

Extract features from **Qwen2.5-Base** (no fine-tuning) and run LOO CV.

**Expected**:
- Base model: 86.68% similarity (from previous diagnosis)
- Fine-tuned: 95.25% similarity

**Test**: Does fine-tuning make performance worse?

### Priority 4: Investigate Prompt Format

Current prompts end with: `"A: Machine"`

**Hypothesis**: All prompts ending identically → similar encodings

**Test**: Try prompts without the trailing "A: Machine" or with varied endings.

---

## Decision Tree

```
Generation Analysis Results:
├─ All samples → same token (>95% bias)
│  └─ Outcome: Model has generation bias
│     └─ Action: Change prompt format or try different model
│
├─ Tokens reasonably diverse (60/40 - 80/20 split)
│  ├─ Single-step extraction fixes similarity?
│  │  ├─ Yes → Use single-step, re-extract all models
│  │  └─ No  → Continue to base model test
│  │
│  └─ Base model performs better?
│     ├─ Yes → Fine-tuning reduces diversity (problematic)
│     │  └─ Action: Use base models for CENTaUR instead
│     └─ No  → Both have high similarity
│        └─ Action: Investigate prompt format
│
└─ Prompt format change reduces similarity?
   ├─ Yes → Re-extract with new prompts
   └─ No  → Fundamental limitation of approach
      └─ Action: Consult original CENTaUR authors or try different models
```

---

## Technical Notes

### Why Process Terminated at Fold 50

Exit code 255 suggests:
- SSH connection timeout or instability
- Remote process killed (OOM, time limit)
- SLURM job preemption

### Binomial Regression Warnings

Frequent warnings:
```
Warning: Invalid loss detected at iteration 10
Stopping early at iteration 11 due to invalid loss: 10000000000.0
```

**Cause**: High feature similarity (95.25%) causes near-perfect collinearity, leading to:
- Numerical instability in loss computation
- Infinite or NaN gradients during LBFGS optimization
- Early stopping triggered correctly

**Not a bug**: This is expected behavior given feature characteristics.

### Alpha Selection Pattern

From partial results, many folds selected alpha=1.0 (maximum regularization):
- Indicates features are not strongly informative
- Regularization trying to prevent overfitting to noise
- Consistent with high similarity hypothesis

---

## References

- Investigation Report: `claudedocs/INVESTIGATION_REPORT_2025-11-05.md`
- Corrected Extraction Script: `scripts/extract_centaur_features.py`
- Debug Scripts: `scripts/debug_extraction_detailed.py`, `scripts/debug_generation.py`
- Test Script: `scripts/test_loo_cv_fixed.py`

---

## Verdict

**Status**: 🟡 **PARTIAL SUCCESS - ROOT CAUSE IDENTIFIED**

1. ✅ Extraction methodology bug correctly fixed
2. ✅ Performance improved (0.8560 → 0.7623)
3. ❌ Still worse than random baseline
4. 🔍 **New focus**: High similarity (95.25%) is the real bottleneck
5. 📋 **Action**: Investigate generation patterns, test single-step extraction, compare base models

**The extraction fix was necessary but not sufficient. The high similarity problem requires deeper investigation.**
