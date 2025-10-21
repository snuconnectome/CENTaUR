# Ko-CENTaUR Evaluation Status Report

**Date**: October 14, 2025
**Status**: Initial LOO Cross-Validation Complete
**Critical Finding**: Ko-CENTaUR shows identical performance to EXAONE-base

---

## Executive Summary

Initial Leave-One-Out (LOO) cross-validation evaluations have been completed on two test sets:
- **choices13k_test** (20 samples)
- **choices13k_100** (100 samples)

**Critical Finding**: Ko-CENTaUR (fine-tuned) and EXAONE-base (pretrained) show **identical performance** on both test sets, suggesting the current fine-tuning approach may not be producing meaningful changes in the model's cognitive representations.

---

## Evaluation Results

### Test Set 1: choices13k_test (20 samples)

**Ko-CENTaUR**:
- Mean Accuracy: **0.700 ± 0.470**
- LOO CV: 20 folds
- Status: ✅ Complete

**EXAONE-base (Baseline)**:
- Mean Accuracy: **0.700 ± 0.470**
- LOO CV: 20 folds
- Status: ✅ Complete

**Difference**: **0.000** (identical)

---

### Test Set 2: choices13k_100 (100 samples)

**Ko-CENTaUR**:
- Mean Accuracy: **0.560 ± 0.499**
- LOO CV: 100 folds
- Status: ✅ Complete

**EXAONE-base (Baseline)**:
- Mean Accuracy: **0.560 ± 0.499**
- LOO CV: 100 folds
- Status: ✅ Complete

**Difference**: **0.000** (identical)

---

## Analysis: Why Identical Performance?

### Possible Explanations

1. **LoRA Adaptation Not Affecting Last-Layer Representations**
   - LoRA may be modifying internal representations without changing the final hidden states
   - The fine-tuning might be learning task-specific patterns that don't manifest in last-layer features
   - Solution: Extract features from multiple layers or different components

2. **Insufficient Training**
   - Model may not have trained long enough to develop distinct cognitive representations
   - Learning rate might be too low for meaningful adaptation
   - Solution: Train longer or increase learning rate

3. **Task-Model Mismatch**
   - The psych101 training data might not align well with choices13k evaluation
   - Solution: Fine-tune directly on decision-making tasks (risky choice, explore-exploit)

4. **Feature Extraction Method**
   - Last-layer hidden states might not capture the relevant cognitive features
   - Original CENTaUR used specific intermediate representations
   - Solution: Experiment with different layer extractions

5. **Binomial Regression Finding Same Patterns**
   - Both models might contain similar decision-making patterns in their pretrained representations
   - Fine-tuning hasn't created sufficiently distinct patterns
   - Solution: Use more distinctive training objectives

---

## Training Details

**Model**: EXAONE 4.0-32B
**Method**: LoRA fine-tuning on psych101 dataset
**Checkpoint Used**: `models/exaone-psych101-full/checkpoint-22536`
**Training Data**: `data/raw/psych101_train.jsonl`
**Evaluation Data**:
- `data/choices13k_test.jsonl` (20 samples from choices13k)
- `data/choices13k_100.jsonl` (100 samples from choices13k)

---

## Next Steps & Recommendations

### Immediate Actions

1. **Verify Feature Extraction**
   ```bash
   # Check if Ko-CENTaUR features differ from baseline
   python -c "
   import torch
   kocentaur = torch.load('results/choices13k_100/kocentaur_features.pt')
   baseline = torch.load('results/choices13k_100/baseline_features.pt')
   print('Feature difference:', (kocentaur - baseline).abs().mean())
   "
   ```

2. **Examine Training Logs**
   - Check loss curves for evidence of learning
   - Verify checkpoint selection (did we use the best checkpoint?)
   - Review training metrics for any anomalies

3. **Test Different Checkpoints**
   - Evaluate earlier/later checkpoints
   - Check if performance varies across training progression

### Medium-Term Solutions

1. **Multi-Layer Feature Extraction**
   - Extract features from layers [8, 16, 24, 32] instead of just last layer
   - Concatenate or average across multiple layers
   - Test if internal representations show more variation

2. **Task-Specific Fine-Tuning**
   - Fine-tune directly on choices13k training data (if available)
   - Use decision-making specific prompts and objectives
   - Consider contrastive learning approaches

3. **Alternative Baselines**
   - Test against random features baseline
   - Compare with other model sizes (EXAONE-7B, EXAONE-13B)
   - Verify the evaluation pipeline is working correctly

### Long-Term Investigations

1. **Representation Analysis**
   - Use CKA (Centered Kernel Alignment) to compare representations
   - Visualize feature spaces with t-SNE or UMAP
   - Compute representational similarity matrices

2. **Probing Tasks**
   - Test if fine-tuned model shows different behavior on auxiliary tasks
   - Measure change in token probabilities for decision-related tokens
   - Analyze attention patterns on decision-making contexts

3. **Training Methodology Review**
   - Consider full fine-tuning instead of LoRA
   - Experiment with different LoRA ranks (current: unclear)
   - Try adapter methods or prefix tuning

---

## Technical Notes

### Evaluation Pipeline

```python
# Standard LOO CV procedure:
for test_idx in range(n_samples):
    train_idx = all_indices - {test_idx}

    # Fit binomial regression
    clf = LogisticRegression(max_iter=1000)
    clf.fit(features[train_idx], labels[train_idx])

    # Predict on held-out sample
    pred = clf.predict(features[test_idx])
    accuracy = (pred == labels[test_idx])
```

### Feature Dimensions

- **Ko-CENTaUR**: torch.Size([n_samples, 4096])
- **EXAONE-base**: torch.Size([n_samples, 4096])
- Both use last-layer hidden states from 32B parameter models

### Computational Resources

- Models loaded in 4-bit quantization
- Feature extraction: ~0.1s per sample
- LOO CV: ~2-3 minutes for 100 samples
- Total evaluation time: ~5 minutes per model per dataset

---

## Conclusion

The identical performance between Ko-CENTaUR and EXAONE-base indicates that the current fine-tuning approach is not producing measurably different cognitive representations in the last-layer hidden states. This is a valuable finding that highlights the need for:

1. Verification of the training and evaluation pipeline
2. Exploration of alternative feature extraction methods
3. Investigation of different fine-tuning strategies
4. Consideration of task-specific training approaches

**Status**: Investigation phase - need to understand why fine-tuning isn't differentiating the models before proceeding with large-scale evaluation.

---

## Files & Artifacts

**Evaluation Results**:
- `results/choices13k_test/` - 20-sample LOO CV results
- `results/choices13k_100/` - 100-sample LOO CV results

**Checkpoints**:
- `models/exaone-psych101-full/checkpoint-22536` - Current Ko-CENTaUR

**Scripts**:
- `scripts/run_full_eval.py` - LOO cross-validation evaluation
- `scripts/run_quick_eval.py` - Fast evaluation for debugging

**Logs**:
- Background processes show successful completion
- Some connection issues during execution (non-critical)
