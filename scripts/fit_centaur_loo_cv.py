#!/usr/bin/env python3
"""
100-fold Leave-One-Out Cross-Validation with Binomial Regression
Following original Binz & Schulz (2023) CENTaUR methodology
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import sys
sys.path.append('/scratch/connectome/connectome1/ko-centaur')
from models import BinomialRegression


def nested_cv_select_alpha(X_train, y_train, alpha_grid, n_folds=11, seed=42):
    """
    Nested cross-validation to select optimal alpha (L2 regularization)

    Args:
        X_train: Training features (n_train, hidden_dim)
        y_train: Training labels (n_train,) - binary 0/1
        alpha_grid: List of alpha values to try
        n_folds: Number of folds for nested CV
        seed: Random seed

    Returns:
        best_alpha: Optimal alpha value
    """
    np.random.seed(seed)
    n_train = len(X_train)

    # Create fold indices
    indices = np.arange(n_train)
    np.random.shuffle(indices)
    fold_size = n_train // n_folds

    # Track validation NLL for each alpha
    alpha_scores = {alpha: [] for alpha in alpha_grid}

    # 11-fold CV
    for fold in range(n_folds):
        # Split into train/val
        val_start = fold * fold_size
        val_end = val_start + fold_size if fold < n_folds - 1 else n_train

        val_idx = indices[val_start:val_end]
        train_idx = np.concatenate([indices[:val_start], indices[val_end:]])

        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        X_fold_val = X_train[val_idx]
        y_fold_val = y_train[val_idx]

        # Try each alpha
        for alpha in alpha_grid:
            # Train model
            model = BinomialRegression(num_inputs=X_train.shape[1], alpha=alpha)

            # Convert to proper format for BinomialRegression
            # y_train is binary 0/1, we need num_choices=1, num_B_choices=y_train
            num_choices = torch.ones(len(X_fold_train), dtype=torch.long)
            num_B_choices = y_fold_train.long()

            model.fit(X_fold_train, num_choices, num_B_choices, num_iterations=100)

            # Compute validation NLL
            with torch.no_grad():
                logits = model(X_fold_val)
                probs_B = torch.sigmoid(logits)

                # NLL for binary classification
                val_nll = 0.0
                for i in range(len(y_fold_val)):
                    if y_fold_val[i] == 1:
                        val_nll -= torch.log(probs_B[i] + 1e-10)
                    else:
                        val_nll -= torch.log(1 - probs_B[i] + 1e-10)

                alpha_scores[alpha].append(val_nll.item())

    # Select alpha with lowest average validation NLL
    avg_scores = {alpha: np.mean(scores) for alpha, scores in alpha_scores.items()}
    best_alpha = min(avg_scores.keys(), key=lambda x: avg_scores[x])

    return best_alpha, avg_scores


def loo_cv_fold(features, labels, test_idx, alpha_grid, fold_id):
    """
    Single fold of Leave-One-Out Cross-Validation

    Args:
        features: All features (100, 5120)
        labels: All labels (100,)
        test_idx: Index of test sample
        alpha_grid: Grid of alpha values for nested CV
        fold_id: Fold number for progress tracking

    Returns:
        dict with test_nll, best_alpha, test_prob, test_label
    """
    # Split train/test
    train_idx = [i for i in range(len(features)) if i != test_idx]

    X_train = features[train_idx]
    y_train = labels[train_idx]
    X_test = features[test_idx:test_idx+1]
    y_test = labels[test_idx]

    # Normalize features using training set statistics
    train_mean = X_train.mean(dim=0, keepdim=True)
    train_std = X_train.std(dim=0, keepdim=True) + 1e-8

    X_train_norm = (X_train - train_mean) / train_std
    X_test_norm = (X_test - train_mean) / train_std

    # Select alpha via nested CV
    best_alpha, alpha_scores = nested_cv_select_alpha(
        X_train_norm, y_train, alpha_grid, n_folds=11, seed=42
    )

    # Train final model on all training data with best alpha
    model = BinomialRegression(num_inputs=X_train_norm.shape[1], alpha=best_alpha)

    num_choices = torch.ones(len(X_train_norm), dtype=torch.long)
    num_B_choices = y_train.long()

    model.fit(X_train_norm, num_choices, num_B_choices, num_iterations=100)

    # Evaluate on test sample
    with torch.no_grad():
        logits = model(X_test_norm)
        prob_B = torch.sigmoid(logits).item()

        # Compute test NLL
        if y_test == 1:
            test_nll = -np.log(prob_B + 1e-10)
        else:
            test_nll = -np.log(1 - prob_B + 1e-10)

    return {
        'fold_id': fold_id,
        'test_idx': test_idx,
        'test_nll': test_nll,
        'test_prob_B': prob_B,
        'test_label': y_test.item(),
        'best_alpha': best_alpha,
        'alpha_scores': alpha_scores
    }


def run_loo_cv(features_path, output_path, model_name):
    """
    Run 100-fold Leave-One-Out Cross-Validation

    Args:
        features_path: Path to extracted features (.pth file)
        output_path: Path to save results
        model_name: Name of model for logging
    """
    print(f"\n{'='*80}")
    print(f"100-fold LOO Cross-Validation: {model_name}")
    print(f"{'='*80}\n")

    # Load features
    print(f"1. Loading features from {features_path}")
    data = torch.load(features_path)
    features = data['features'].float()  # (100, 5120) - convert to float32
    labels = data['labels'].float()      # (100,) - convert to float32

    print(f"   Features shape: {features.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Label distribution: {labels.sum().item()} choice B, {(1-labels).sum().item()} choice A")

    # Alpha grid (from original paper)
    alpha_grid = [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    print(f"\n2. Alpha grid: {alpha_grid}")
    print(f"   Nested CV: 11-fold for each outer fold")

    # Run LOO CV
    print(f"\n3. Running 100-fold Leave-One-Out Cross-Validation...")
    results = []

    for fold_id in tqdm(range(len(features)), desc="LOO CV Progress"):
        fold_result = loo_cv_fold(
            features, labels,
            test_idx=fold_id,
            alpha_grid=alpha_grid,
            fold_id=fold_id
        )
        results.append(fold_result)

        # Progress update every 10 folds
        if (fold_id + 1) % 10 == 0:
            current_avg_nll = np.mean([r['test_nll'] for r in results])
            print(f"\n   Fold {fold_id + 1}/100 - Current avg NLL: {current_avg_nll:.4f}")

    # Compute overall statistics
    test_nlls = [r['test_nll'] for r in results]
    test_probs = [r['test_prob_B'] for r in results]
    test_labels = [r['test_label'] for r in results]
    best_alphas = [r['best_alpha'] for r in results]

    avg_nll = np.mean(test_nlls)
    std_nll = np.std(test_nlls)
    total_nll = np.sum(test_nlls)

    # Compute accuracy (for comparison)
    predictions = [1 if p > 0.5 else 0 for p in test_probs]
    accuracy = np.mean([pred == label for pred, label in zip(predictions, test_labels)])

    print(f"\n{'='*80}")
    print(f"RESULTS: {model_name}")
    print(f"{'='*80}")
    print(f"Negative Log-Likelihood (NLL):")
    print(f"  Average NLL:  {avg_nll:.4f} ± {std_nll:.4f}")
    print(f"  Total NLL:    {total_nll:.2f}")
    print(f"\nAccuracy (for reference only):")
    print(f"  Accuracy:     {accuracy:.1%}")
    print(f"\nAlpha Selection:")
    alpha_counts = {alpha: best_alphas.count(alpha) for alpha in set(best_alphas)}
    for alpha, count in sorted(alpha_counts.items()):
        print(f"  alpha={alpha:6.4f}: {count:3d} folds ({count/100:.1%})")
    print(f"{'='*80}\n")

    # Save results
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        'model_name': model_name,
        'features_path': features_path,
        'n_samples': len(features),
        'hidden_dim': features.shape[1],
        'alpha_grid': alpha_grid,
        'results': results,
        'test_nlls': test_nlls,
        'test_probs': test_probs,
        'test_labels': test_labels,
        'best_alphas': best_alphas,
        'avg_nll': avg_nll,
        'std_nll': std_nll,
        'total_nll': total_nll,
        'accuracy': accuracy,
        'alpha_counts': alpha_counts
    }, output_path)

    print(f"✅ Results saved to: {output_path}\n")

    return {
        'avg_nll': avg_nll,
        'std_nll': std_nll,
        'total_nll': total_nll,
        'accuracy': accuracy
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="100-fold LOO CV with Binomial Regression for CENTaUR evaluation"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["qwen25", "deepseek"],
        help="Model to evaluate"
    )
    args = parser.parse_args()

    # Model configurations
    if args.model == "qwen25":
        features_path = "/scratch/connectome/connectome1/ko-centaur/data/centaur_features_qwen25.pth"
        output_path = "/scratch/connectome/connectome1/ko-centaur/data/loo_cv_results_qwen25.pth"
        model_name = "Qwen2.5-32B-QLoRA"
    elif args.model == "deepseek":
        features_path = "/scratch/connectome/connectome1/ko-centaur/data/centaur_features_deepseek.pth"
        output_path = "/scratch/connectome/connectome1/ko-centaur/data/loo_cv_results_deepseek.pth"
        model_name = "DeepSeek-R1-32B-QLoRA"

    # Run evaluation
    run_loo_cv(features_path, output_path, model_name)
