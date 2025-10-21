# Ko-CENTaUR Feature Extraction Bug Fix Report

**Date**: 2025-10-12
**Status**: 🔧 FIX IMPLEMENTED, VALIDATION IN PROGRESS

---

## 🐛 Bug Identified

### Root Cause
**File**: `evaluation/extract_features.py:24-30`

**Problem**: Feature extraction expected `task_description` field but choices13k dataset uses `text` field.

```python
# BUGGY CODE (Lines 24-30)
task_desc = sample.get('task_description', '')  # Returns '' for all samples!
system_prompt = sample.get('system_prompt', '')

if system_prompt:
    prompt = f"{system_prompt}\n\n{task_desc}"
else:
    prompt = task_desc  # ALL SAMPLES GET EMPTY STRING ''
```

### Impact
- All 100 samples extracted features from **empty string** `''`
- Result: **Identical features** for all samples (variance = 0.000000)
- LogisticRegression on constant features always predicts **majority class**
- Models showed 100% identical predictions (all Choice B)
- 56% accuracy = 56% class proportion (trivial baseline)

---

## 🔍 Investigation Timeline

### 1. User Observation (Korean)
> "이 결과가 좀 이상하지 않아? variance도 너무 크고"
>
> Translation: "Isn't this result strange? The variance is too large"

**User identified 3 suspicious patterns**:
1. Identical performance (56.0% ± 49.9% for both models)
2. Very large variance (49.9%)
3. Accuracy exactly matches class imbalance (56% B in dataset)

### 2. Variance Analysis
- **Calculation**: For binary with p=0.56, theoretical std = sqrt(0.56*0.44) = 0.4964
- **Result**: ✓ Variance is **NORMAL** for binary classification

### 3. Prediction Analysis
**Script**: `scripts/analyze_predictions.py`

**Findings**:
- Ko-CENTaUR: 0/100 Choice A, **100/100 Choice B**
- EXAONE-base: 0/100 Choice A, **100/100 Choice B**
- Identical predictions: **100%**
- Conclusion: ❌ **Trivial majority class baseline**

### 4. Feature Analysis
**Script**: `scripts/analyze_features.py`

**Findings**:
```
Feature Variance (across samples):
  Ko-CENTaUR: 0.000000
  EXAONE-base: 0.000000

Constant Features (std < 1e-6):
  Ko-CENTaUR: 4096/4096 (100%)
  EXAONE-base: 4096/4096 (100%)

Max diff from first sample:
  Ko-CENTaUR: 0.000000
  EXAONE-base: 0.000000

❌ All samples have IDENTICAL features!
```

### 5. Root Cause Identification
**Code inspection** revealed the bug in `extract_features.py`:
- Code expects: `sample['task_description']`
- Dataset provides: `sample['text']`
- Result: `.get('task_description', '')` returns `''` for all samples

---

## ✅ Fix Implemented

### Modified Files

#### 1. `evaluation/extract_features.py:12-50`
**Function**: `extract_features_for_sample()`

**Fix**: Support both dataset formats
```python
def extract_features_for_sample(model_manager, sample: Dict) -> torch.Tensor:
    """
    Extract features for a single sample

    Support both dataset formats:
    - choices13k: {'text': str, 'choice': int}
    - Psych-101: {'task_description': str, 'label': int, 'system_prompt': str}
    """
    # Build prompt - support both formats
    if 'text' in sample:
        # choices13k format
        prompt = sample['text']
    elif 'task_description' in sample:
        # Psych-101 format
        task_desc = sample['task_description']
        system_prompt = sample.get('system_prompt', '')
        prompt = f"{system_prompt}\n\n{task_desc}" if system_prompt else task_desc
    else:
        raise ValueError(
            f"Sample must have either 'text' or 'task_description' field. "
            f"Found keys: {list(sample.keys())}"
        )

    formatted_prompt = model_manager.format_prompt(prompt)
    features = model_manager.extract_features(formatted_prompt)

    return features
```

#### 2. `evaluation/extract_features.py:273-306`
**Function**: `validate_dataset()`

**Fix**: Validate both dataset formats
```python
def validate_dataset(samples: List[Dict], require_labels: bool = False):
    """
    Validate dataset schema

    Support both formats:
    - choices13k: 'text' + 'choice'
    - Psych-101: 'task_description' + 'label'
    """
    for i, sample in enumerate(samples):
        # Check required fields
        has_text = 'text' in sample
        has_task_desc = 'task_description' in sample

        if not has_text and not has_task_desc:
            raise ValueError(
                f"Sample {i}: must have either 'text' or 'task_description'"
            )

        # Check labels if required
        if require_labels:
            has_label = 'label' in sample
            has_choice = 'choice' in sample

            if not has_label and not has_choice:
                raise ValueError(
                    f"Sample {i}: must have either 'label' or 'choice' field"
                )
```

### Deployment
1. ✅ Fixed code copied to server: `scp evaluation/extract_features.py server:/scratch/connectome/connectome1/ko-centaur/evaluation/`
2. ✅ Cached buggy features deleted: `rm -rf results/choices13k_100/features`
3. ✅ Analysis scripts deployed: `scp scripts/analyze_*.py server:/scratch/connectome/connectome1/ko-centaur/scripts/`
4. 🔄 Re-running evaluation with fixed code

---

## 🧪 Validation Plan

### Expected Results After Fix

1. **Feature Variance**:
   - Should be **non-zero** across samples
   - Features should differ for different prompts

2. **Prediction Distribution**:
   - Should **NOT** be 100% Choice B
   - Some variation in predictions expected

3. **Model Differentiation**:
   - Ko-CENTaUR and EXAONE-base may show **different** results
   - Accuracy should deviate from class imbalance (56%)

### Validation Commands

```bash
# 1. Check feature variance
cd /scratch/connectome/connectome1/ko-centaur
python scripts/analyze_features.py

# 2. Check prediction distribution
python scripts/analyze_predictions.py results/choices13k_100_fixed/all_results.pth

# 3. Compare with buggy results
diff results/choices13k_100/all_results.pth results/choices13k_100_fixed/all_results.pth
```

---

## 📊 Previous (Buggy) Results

### 100-Sample Evaluation (INVALID)
```
Ko-CENTaUR:  56.0% ± 49.9% (100 folds)
EXAONE-base: 56.0% ± 49.9% (100 folds)

Predictions:
- Choice A: 0/100 (0%)
- Choice B: 100/100 (100%)  ← Trivial baseline
- Identical: 100/100 (100%)

Features:
- Variance: 0.000000  ← ALL IDENTICAL
- Constant features: 4096/4096 (100%)
```

### 20-Sample Test (INVALID)
```
Ko-CENTaUR:  70.0% ± 47.0% (20 folds)
EXAONE-base: 70.0% ± 47.0% (20 folds)

Class distribution: 30% A, 70% B
Result: 70% accuracy = 70% class B proportion (trivial baseline)
```

---

## 🎯 Current Status

**Time**: 2025-10-12 11:31 PST
**Action**: Running fixed evaluation (100 samples)
**Command**:
```bash
python scripts/run_full_eval.py \
  --dataset data/choices13k_100.jsonl \
  --baselines exaone-base \
  --n_samples 100 \
  --output_dir results/choices13k_100_fixed
```

**Progress**:
- ✅ Dataset loaded (100 samples)
- ✅ Ko-CENTaUR model loaded
- ✅ Ko-CENTaUR features extracted (torch.Size([100, 4096]))
- ✅ EXAONE-base model loaded
- 🔄 EXAONE-base feature extraction (in progress, stdout buffering)
- ⏳ LOO CV pending
- ⏳ Results analysis pending

**Note**: Process is running but stdout buffering delays output display. Extraction of 100 different prompts (vs. 100 identical empty strings) takes longer.

---

## 📝 Lessons Learned

### Dataset Format Heterogeneity
- Multiple dataset formats in use (Psych-101 vs choices13k)
- Feature extraction must support both formats
- Validation functions must check both schemas

### Silent Failures
- `.get()` with default `''` silently fails without error
- Results appear valid (models run, accuracy computed)
- Only careful analysis reveals trivial baseline

### Importance of Validation
- User's instinct about "strange results" was 100% correct
- Systematic validation caught:
  1. Variance analysis (ruled out)
  2. Prediction analysis (found trivial baseline)
  3. Feature analysis (found identical features)
  4. Code inspection (found root cause)

### Test-Driven Development Value
- Bug bypassed existing tests (wrong assumption about data format)
- Need integration tests with both dataset formats
- Validation scripts essential for catching silent failures

---

## ✅ Next Steps

1. **Monitor evaluation completion** (~2-3 more minutes expected)
2. **Run validation scripts** when results available
3. **Analyze corrected results**:
   - Feature variance (should be non-zero)
   - Prediction distribution (should vary)
   - Model comparison (may differ)
4. **Update EVALUATION_STATUS.md** with corrected results
5. **Add regression tests** for both dataset formats
