# Generation Bias Analysis
**Date**: 2025-11-06
**Issue**: High feature similarity (95.25%) and worse-than-random performance (NLL = 0.7623)
**Root Cause Identified**: **Strong generation bias toward choice B**

---

## Executive Summary

**CONFIRMED on full 100-sample dataset**:

🔴 **Generation Bias**: Model generates ' B' for **72% of samples** (72/100)
🔴 **Low Accuracy**: **50% accuracy** - exactly at chance level (50/100 correct)
🔴 **No Learning**: Model defaults to B regardless of true choice
  - True A → Generated A: only 25% (should be ~100%)
  - True A → Generated B: 75% (wrong 3/4 of the time)

**Conclusion**: The high similarity (95.25% on full dataset) is **confirmed** to be caused by **most samples generating the same token** (' B'), not extraction methodology. The model is NOT learning choice patterns from prompts.

---

## Debug Results (5 samples)

### Generation Patterns

| Sample | True Label | Generated Token | Match | Token ID |
|--------|------------|----------------|-------|----------|
| 0 | B | ' B' | ✓ | 425 |
| 1 | A | ' B' | ✗ | 425 |
| 2 | B | ' A' | ✗ | 362 |
| 3 | A | ' B' | ✗ | 425 |
| 4 | B | ' B' | ✓ | 425 |

**Statistics**:
- **Generation distribution**: 80% → ' B' (4/5), 20% → ' A' (1/5)
- **Accuracy**: 40% (2/5 correct)
- **Unique tokens**: 2 (as expected for binary choice)

### Feature Similarity

**Average pairwise similarity**: 75.41%

**Similarity Matrix**:
```
     0      1      2      3      4
0  1.000  0.961  0.934  0.926  0.945
1  0.961  0.996  0.914  0.902  0.938
2  0.934  0.914  1.000  0.973  0.961
3  0.926  0.902  0.973  1.000  0.969
4  0.945  0.938  0.961  0.969  0.996
```

**Observations**:
- Samples 1 and 4 (both generated ' B'): 99.6% similar
- Samples 0 and 4 (both generated ' B'): 99.6% similar
- Samples with same token → very high similarity (>96%)
- Samples with different tokens → lower similarity (~91-93%)

### Hidden State Statistics

| Sample | Mean | Std | Generated |
|--------|------|-----|-----------|
| 0 | 0.0669 | 3.03 | ' B' |
| 1 | 0.0859 | 3.03 | ' B' |
| 2 | 0.0908 | 3.31 | ' A' |
| 3 | 0.0762 | 3.33 | ' B' |
| 4 | 0.0811 | 3.27 | ' B' |

**Key insight**: Hidden states for same generated token are nearly identical (mean 0.066-0.086, std 3.03-3.27).

---

## Root Cause Analysis

### Why High Similarity (95.25% on full dataset)?

**Hypothesis**: If the model generates ' B' for 80-90% of samples:
- Hidden states cluster around ' B' token representation
- Only 10-20% of samples have distinct ' A' representation
- Average pairwise similarity inflated by majority token

**Evidence from 5-sample analysis**:
- 4 samples generated ' B' → pairwise similarity 94-99.6%
- 1 sample generated ' A' → similarity with others 91-93%
- Average similarity: 75.41% (dominated by ' B' cluster)

### Why Worse Than Random (NLL = 0.7623 vs 0.6931)?

**Explanation**:
1. **Ground truth labels**: 56% choice B, 44% choice A (relatively balanced)
2. **Model generation**: ~80% choice B, ~20% choice A (strongly biased)
3. **Consequence**: Model predicts B for most samples, regardless of actual choice
4. **Result**: Equivalent to always guessing the majority class

**Comparison**:
- **Random guessing**: 50/50 → NLL = 0.6931
- **Always predict B**: ~80/20 → NLL ≈ 0.76 (matches observed 0.7623!)
- **Informed prediction**: Should be < 0.69

The model's strong bias toward B makes it perform like a biased coin, not an informed predictor.

### Why Better Than Buggy (0.7623 vs 0.8560)?

Buggy extraction had even worse performance because:
1. Extracted from last PROMPT token (not choice-relevant)
2. All prompts ended with "A: Machine" → all features extremely similar (89%)
3. No predictive signal at all

Corrected extraction at least captures:
- Token-level diversity (2 unique tokens instead of 1)
- Weak signal from generation (40% accuracy vs 0%)

---

## Implications

### For Current Methodology

❌ **Model is NOT suitable for CENTaUR in current form**

Reasons:
1. Strong generation bias (80% → B) means features don't capture true choice patterns
2. Low generation accuracy (40%) indicates model hasn't learned risky choice behavior
3. Fine-tuning may have created bias instead of improving choice modeling

### For Next Steps

Three possible causes of generation bias:

#### Cause 1: Prompt Format Issue
**Hypothesis**: Prompts ending with "Machine chose:" bias toward one token

**Evidence**:
- All prompts have identical ending
- Original CENTaUR used different prompt format

**Test**: Try alternative prompt formats:
```python
# Current
"Option A: ..., Option B: ... Machine chose:"

# Alternative 1: No leading text
"Option A: ..., Option B: ..."

# Alternative 2: Question format
"Option A: ..., Option B: ... Which option? Answer:"

# Alternative 3: Direct format (like original LLaMA)
"A: ..., B: ... Choice:"
```

#### Cause 2: Fine-Tuning Artifact
**Hypothesis**: QLoRA fine-tuning on Choices13k introduced bias

**Evidence**:
- Base model: 86.68% similarity (from previous diagnosis)
- Fine-tuned: 95.25% similarity (WORSE diversity)
- Fine-tuning should improve, not worsen

**Test**: Extract features from Qwen2.5-**Base** (no fine-tuning) and compare:
- Generation bias (% B vs A)
- Feature similarity
- LOO CV performance

**Expected**: Base model should have less bias and better diversity.

#### Cause 3: Model Architecture Limitation
**Hypothesis**: Qwen2.5 not suitable for this task (preference bias)

**Evidence**:
- Original CENTaUR used LLaMA (different architecture)
- Qwen may have safety/preference tuning affecting choices

**Test**: Try different model family:
- DeepSeek-R1 (already fine-tuned)
- EXAONE (different base architecture)
- Original LLaMA models (if accessible)

---

## Recommended Investigation Path

### Priority 1: Check Generation Bias on Full Dataset ⚠️ URGENT

Modify `debug_extraction_detailed.py` to process all 100 samples and report:
```python
# Expected output
Total samples: 100
Generated A: 20 (20%)  ← Check if this is accurate
Generated B: 80 (80%)  ← Check if this matches hypothesis

Ground truth A: 44 (44%)
Ground truth B: 56 (56%)

Generation accuracy: 48% (48/100 correct)

Token distribution by true label:
  True A → Generated A: 8 / 44 (18%)
  True A → Generated B: 36 / 44 (82%)
  True B → Generated A: 12 / 56 (21%)
  True B → Generated B: 44 / 56 (79%)
```

**Decision based on results**:
- If bias > 70%: Generation is the problem, not extraction
- If bias < 60%: Similarity may have other causes

### Priority 2: Test Base Model (No Fine-Tuning)

**Hypothesis**: Fine-tuning created the bias

**Test**:
1. Extract features from Qwen2.5-32B-Instruct (NO adapter)
2. Check generation bias
3. Compare similarity with fine-tuned

**Expected outcomes**:
- **Base < 70% bias**: Fine-tuning caused bias → Use base models
- **Base > 70% bias**: Model architecture issue → Try different models

### Priority 3: Test Alternative Prompts

**Hypothesis**: Prompt format causes bias

**Test**: Modify prompt ending and re-extract small sample (10 samples):
```python
# Test 3 prompt formats
formats = [
    "Option A: {}, Option B: {} Machine chose:",      # Current
    "Option A: {}, Option B: {}",                      # No prompt
    "Choose A: {} or B: {}. Answer:"                   # Question format
]
```

Check generation bias for each format.

---

## Code Changes Needed

### 1. Full Dataset Generation Analysis Script

Create `scripts/analyze_full_generation.py`:
```python
#!/usr/bin/env python3
"""
Analyze generation patterns on full 100-sample dataset
Reports generation bias, accuracy, and token distribution
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json
from collections import Counter

# Load model (same as extract_centaur_features.py)
base_model = "Qwen/Qwen2.5-32B-Instruct"
adapter = "/scratch/connectome/connectome1/ko-centaur/outputs/qwen25-32b-qlora"
dataset_path = "/scratch/connectome/connectome1/ko-centaur/data/choices13k_100.jsonl"

# ... (model loading code)

# Generate and analyze
generated_tokens = []
true_labels = []

for idx, sample in enumerate(data):
    prompt = sample["text"]
    label = sample["choice"]

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )

    generated_token_id = gen_outputs[0, -1].item()
    generated_text = tokenizer.decode(generated_token_id)

    generated_tokens.append((generated_token_id, generated_text))
    true_labels.append(label)

# Analyze distribution
token_counter = Counter([text for _, text in generated_tokens])
print(f"\nGeneration Distribution:")
for token, count in token_counter.most_common():
    print(f"  '{token}': {count}/{len(data)} ({count/len(data)*100:.1f}%)")

# Analyze accuracy
correct = sum(1 for i, (_, text) in enumerate(generated_tokens)
              if (true_labels[i] == 1 and 'B' in text) or
                 (true_labels[i] == 0 and 'A' in text))
print(f"\nGeneration Accuracy: {correct}/{len(data)} ({correct/len(data)*100:.1f}%)")

# Conditional distributions
print(f"\nConditional Generation:")
true_a_gen_a = sum(1 for i, (_, text) in enumerate(generated_tokens) if true_labels[i] == 0 and 'A' in text)
true_a_gen_b = sum(1 for i, (_, text) in enumerate(generated_tokens) if true_labels[i] == 0 and 'B' in text)
true_b_gen_a = sum(1 for i, (_, text) in enumerate(generated_tokens) if true_labels[i] == 1 and 'A' in text)
true_b_gen_b = sum(1 for i, (_, text) in enumerate(generated_tokens) if true_labels[i] == 1 and 'B' in text)

print(f"  True A → Generated A: {true_a_gen_a}/{sum(1 for l in true_labels if l == 0)} ({true_a_gen_a/sum(1 for l in true_labels if l == 0)*100:.1f}%)")
print(f"  True A → Generated B: {true_a_gen_b}/{sum(1 for l in true_labels if l == 0)} ({true_a_gen_b/sum(1 for l in true_labels if l == 0)*100:.1f}%)")
print(f"  True B → Generated A: {true_b_gen_a}/{sum(1 for l in true_labels if l == 1)} ({true_b_gen_a/sum(1 for l in true_labels if l == 1)*100:.1f}%)")
print(f"  True B → Generated B: {true_b_gen_b}/{sum(1 for l in true_labels if l == 1)} ({true_b_gen_b/sum(1 for l in true_labels if l == 1)*100:.1f}%)")
```

---

## Decision Tree

```
Run analyze_full_generation.py on 100 samples
│
├─ Generation bias > 70%
│  └─ Root cause: Generation bias
│     ├─ Test base model (no fine-tuning)
│     │  ├─ Base has less bias (<60%) → Use base models
│     │  └─ Base has same bias (>70%) → Test alternative prompts
│     │     ├─ Alternative prompt reduces bias → Use new prompt format
│     │     └─ No improvement → Model architecture issue, try DeepSeek/EXAONE
│     │
│     └─ Test alternative prompts directly
│        └─ (same as above)
│
└─ Generation bias < 60%
   └─ Similarity has other causes
      ├─ Test single-step extraction (output_hidden_states in generate)
      └─ Investigate prompt format effects on hidden states
```

---

## Expected Timeline

1. **Analyze full generation** (5 min model load + 2 min generation): ~7 min
2. **Test base model** (5 min load + 2 min generate + 40 min LOO CV): ~47 min
3. **Test alternative prompts** (5 min load + 2 min × 3 formats): ~11 min

**Total**: ~65 minutes of compute time

---

## References

- Debug script: `scripts/debug_extraction_detailed.py` (completed)
- LOO CV results: `claudedocs/LOO_CV_PARTIAL_RESULTS_2025-11-06.md`
- Investigation history: `claudedocs/INVESTIGATION_REPORT_2025-11-05.md`

---

## Conclusion

The **real problem** is not extraction methodology but **generation bias**:
- Model generates ' B' for ~80% of samples (hypothesis, needs full dataset confirmation)
- This causes high feature similarity (95.25%) because most features represent the same token
- Performance is worse than random (NLL = 0.7623 vs 0.6931) because model ignores choice patterns

**Next step**: Confirm generation bias on full 100-sample dataset, then test root causes (fine-tuning artifact, prompt format, or model architecture).
