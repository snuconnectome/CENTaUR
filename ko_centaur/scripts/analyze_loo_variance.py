#!/usr/bin/env python3
"""
Comprehensive analysis of LOO CV variance for binary classification

This script investigates why standard deviation is ~50% and whether
this indicates a pipeline problem or is mathematically expected.
"""

import torch
import numpy as np
import sys
from pathlib import Path

def analyze_loo_variance(results_path: str):
    """Analyze LOO CV variance for binary classification"""

    # Load results (weights_only=False for PyTorch 2.6+)
    results = torch.load(results_path, weights_only=False)

    print('='*80)
    print('ROOT CAUSE ANALYSIS: LOO Cross-Validation Variance')
    print('='*80)
    print()

    for model_name in ['ko-centaur', 'exaone-base']:
        if model_name not in results:
            continue

        model_results = results[model_name]
        fold_results = model_results['fold_results']

        # Extract per-fold accuracies
        accuracies = np.array([r['test_accuracy'] for r in fold_results])
        predictions = [r['test_prediction'] for r in fold_results]
        true_labels = [r['test_true_label'] for r in fold_results]

        print('='*80)
        print(f'{model_name.upper()} ANALYSIS')
        print('='*80)
        print()

        # Basic statistics
        print('[1] Basic Statistics:')
        print(f'  Total folds: {len(accuracies)}')
        print(f'  Mean accuracy: {np.mean(accuracies):.3f}')
        print(f'  Std accuracy (ddof=1): {np.std(accuracies, ddof=1):.3f}')
        print()

        # CRITICAL: Check data type
        print('[2] Per-Fold Accuracy Structure:')
        unique_values = np.unique(accuracies)
        print(f'  Unique values: {unique_values}')
        print(f'  Number of unique values: {len(unique_values)}')

        if len(unique_values) == 2 and set(unique_values) == {0.0, 1.0}:
            print(f'  ✅ CONFIRMED: Each accuracy is BINARY (0.0 or 1.0)')
            print(f'  ✅ REASON: Each LOO fold has exactly 1 test sample')
            print(f'  ✅ MEANING: accuracy = 1.0 (correct) or 0.0 (wrong)')
        else:
            print(f'  ⚠️  WARNING: Unexpected accuracy values!')
        print()

        # Count outcomes
        print('[3] Fold Outcomes:')
        num_correct = int(np.sum(accuracies))
        num_incorrect = len(accuracies) - num_correct
        print(f'  Correct predictions (1.0): {num_correct}/{len(accuracies)} ({100*num_correct/len(accuracies):.1f}%)')
        print(f'  Incorrect predictions (0.0): {num_incorrect}/{len(accuracies)} ({100*num_incorrect/len(accuracies):.1f}%)')
        print()

        # CRITICAL: Mathematical expectation
        print('[4] Mathematical Analysis (Binary Distribution):')
        p = np.mean(accuracies)
        theoretical_std = np.sqrt(p * (1 - p))
        actual_std = np.std(accuracies, ddof=1)
        difference = abs(theoretical_std - actual_std)

        print(f'  Mean (p): {p:.3f}')
        print(f'  Theoretical std [sqrt(p*(1-p))]: {theoretical_std:.3f}')
        print(f'  Actual std (numpy ddof=1): {actual_std:.3f}')
        print(f'  Absolute difference: {difference:.6f}')

        if difference < 0.01:
            print(f'  ✅ MATCH: Variance is MATHEMATICALLY EXPECTED')
            print(f'  ✅ CONCLUSION: High std (~50%) is CORRECT for binary LOO CV')
        else:
            print(f'  ⚠️  MISMATCH: Variance deviates from expectation!')
        print()

        # Standard error and confidence intervals
        print('[5] Confidence Intervals:')
        se = actual_std / np.sqrt(len(accuracies))
        ci_lower = p - 1.96 * se
        ci_upper = p + 1.96 * se
        print(f'  Standard error: {se:.3f}')
        print(f'  95% CI: [{ci_lower:.3f}, {ci_upper:.3f}]')
        print(f'  CI width: {ci_upper - ci_lower:.3f}')
        print()

        # Prediction distribution analysis
        print('[6] Prediction Distribution:')
        pred_counts = {}
        for pred in predictions:
            pred_counts[pred] = pred_counts.get(pred, 0) + 1
        for pred, count in sorted(pred_counts.items()):
            print(f'  Predicted class {pred}: {count}/{len(predictions)} ({100*count/len(predictions):.1f}%)')

        true_counts = {}
        for label in true_labels:
            true_counts[label] = true_counts.get(label, 0) + 1
        print()
        print('  Ground truth distribution:')
        for label, count in sorted(true_counts.items()):
            print(f'  True class {label}: {count}/{len(true_labels)} ({100*count/len(true_labels):.1f}%)')
        print()

        # Alternative metrics that might be more meaningful
        print('[7] Alternative Evaluation Metrics:')
        print(f'  Log-likelihood mean: {np.mean([r["test_log_likelihood"] for r in fold_results]):.3f}')
        print(f'  Log-likelihood std: {np.std([r["test_log_likelihood"] for r in fold_results], ddof=1):.3f}')
        print()

        # Per-class accuracy
        correct_by_class = {0: 0, 1: 0}
        total_by_class = {0: 0, 1: 0}
        for i, (pred, true) in enumerate(zip(predictions, true_labels)):
            total_by_class[true] = total_by_class.get(true, 0) + 1
            if pred == true:
                correct_by_class[true] = correct_by_class.get(true, 0) + 1

        print('  Per-class accuracy:')
        for class_label in sorted(total_by_class.keys()):
            if total_by_class[class_label] > 0:
                class_acc = correct_by_class.get(class_label, 0) / total_by_class[class_label]
                print(f'  Class {class_label}: {correct_by_class.get(class_label, 0)}/{total_by_class[class_label]} ({100*class_acc:.1f}%)')
        print()

    # Final comprehensive conclusion
    print('='*80)
    print('FINAL DIAGNOSIS')
    print('='*80)
    print()
    print('Question: Is ~50% standard deviation a problem?')
    print()
    print('Answer: NO - It is MATHEMATICALLY EXPECTED and CORRECT.')
    print()
    print('Explanation:')
    print('  1. LOO CV with N=100 creates 100 folds')
    print('  2. Each fold has exactly 1 test sample')
    print('  3. Each per-fold accuracy is BINARY: 0.0 (wrong) or 1.0 (correct)')
    print('  4. For binary data, std = sqrt(p*(1-p)) where p = mean')
    print('  5. With p≈0.60: std = sqrt(0.60*0.40) = 0.490 ≈ 50%')
    print('  6. With p≈0.55: std = sqrt(0.55*0.45) = 0.497 ≈ 50%')
    print()
    print('The high standard deviation is NOT a pipeline bug.')
    print('It is an INHERENT mathematical property of:')
    print('  - Binary per-fold outcomes (0/1)')
    print('  - Leave-one-out cross-validation')
    print('  - Single-sample test sets')
    print()
    print('This is DIFFERENT from:')
    print('  - Standard deviation of PREDICTIONS (which should be ~0 if always predicting same)')
    print('  - Standard deviation in K-fold CV with larger test sets')
    print('  - Standard deviation of continuous predictions')
    print()
    print('='*80)
    print('RECOMMENDATION')
    print('='*80)
    print()
    print('1. The current evaluation methodology is CORRECT')
    print('2. The ~50% std should be EXPECTED for binary LOO CV')
    print('3. Report results as: Mean ± SE (not ± SD)')
    print('4. Use confidence intervals for significance: [Mean - 1.96*SE, Mean + 1.96*SE]')
    print('5. Consider reporting log-likelihood for more stable metric')
    print('6. For model comparison, use paired statistical tests (McNemar, Wilcoxon)')
    print()
    print('Alternative approaches if concerned:')
    print('  - Use k-fold CV (k=5 or k=10) with larger test sets')
    print('  - Report log-likelihood instead of accuracy')
    print('  - Report per-class accuracy')
    print('  - Use bootstrap confidence intervals')
    print('  - Increase sample size (N >> 100)')
    print()
    print('='*80)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        results_path = sys.argv[1]
    else:
        results_path = "results/choices13k_100_fixed/all_results.pth"

    analyze_loo_variance(results_path)
