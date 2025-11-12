#!/usr/bin/env python3
"""
Random Baseline Computation for CENTaUR Evaluation

Computes negative log-likelihood (NLL) for a chance-level model that predicts
50% probability for each choice (random guessing).
"""

import torch
import numpy as np
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.append('/scratch/connectome/connectome1/ko-centaur')


def compute_random_baseline(labels_path, output_path):
    """
    Compute random baseline NLL (50% probability for each choice)

    Args:
        labels_path: Path to features file containing labels
        output_path: Path to save baseline results
    """
    print(f"\n{'='*80}")
    print(f"Random Baseline Computation")
    print(f"{'='*80}\n")

    # Load labels from any features file (they're all the same)
    print(f"1. Loading labels from {labels_path}")
    data = torch.load(labels_path)
    labels = data['labels'].float()
    n_samples = len(labels)

    print(f"   Number of samples: {n_samples}")
    print(f"   Label distribution: {labels.sum().item()} choice B, {(1-labels).sum().item()} choice A")

    # Random baseline: 50% probability for each choice
    print(f"\n2. Computing random baseline NLL")
    print(f"   Model: P(choice=B) = 0.5 for all samples")

    # NLL for binary classification with p=0.5
    # NLL = -log(0.5) = log(2) ≈ 0.6931 per sample
    nll_per_sample = -np.log(0.5)

    test_nlls = [nll_per_sample] * n_samples
    test_probs = [0.5] * n_samples
    test_labels = labels.numpy().tolist()

    avg_nll = np.mean(test_nlls)
    std_nll = np.std(test_nlls)
    total_nll = np.sum(test_nlls)

    # Accuracy: should be ~50% for balanced dataset
    predictions = [1 if p > 0.5 else 0 for p in test_probs]  # Always predicts 0 (tie)
    accuracy = np.mean([pred == label for pred, label in zip(predictions, test_labels)])

    print(f"\n{'='*80}")
    print(f"RESULTS: Random Baseline")
    print(f"{'='*80}")
    print(f"Negative Log-Likelihood (NLL):")
    print(f"  Average NLL:  {avg_nll:.4f} ± {std_nll:.4f}")
    print(f"  Total NLL:    {total_nll:.2f}")
    print(f"  Theoretical:  -log(0.5) = {nll_per_sample:.4f}")
    print(f"\nAccuracy (for reference only):")
    print(f"  Accuracy:     {accuracy:.1%}")
    print(f"  Note: With p=0.5, predictions are ambiguous (tie)")
    print(f"{'='*80}\n")

    # Save results
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_name': 'Random Baseline',
        'n_samples': n_samples,
        'test_nlls': test_nlls,
        'test_probs': test_probs,
        'test_labels': test_labels,
        'avg_nll': avg_nll,
        'std_nll': std_nll,
        'total_nll': total_nll,
        'accuracy': accuracy,
        'theoretical_nll': nll_per_sample
    }, output_path)

    print(f"✅ Results saved to: {output_path}\n")

    return {
        'avg_nll': avg_nll,
        'std_nll': std_nll,
        'total_nll': total_nll,
        'accuracy': accuracy
    }


if __name__ == "__main__":
    # Use any features file to get labels (they're all the same)
    labels_path = "/scratch/connectome/connectome1/ko-centaur/data/features/centaur_features_qwen25.pth"
    output_path = "/scratch/connectome/connectome1/ko-centaur/data/results/loo_cv_results_random.pth"

    compute_random_baseline(labels_path, output_path)
