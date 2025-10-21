#!/usr/bin/env python3
"""
Full Evaluation Script for Ko-CENTaUR

Runs complete Leave-One-Out cross-validation with nested hyperparameter tuning.
Uses sklearn LogisticRegression with L2 regularization (alpha parameter).

Usage:
    python scripts/run_full_eval.py \
        --dataset /scratch/connectome/connectome1/ko-centaur/data/raw/psych101_train.jsonl \
        --baselines exaone-base \
        --n_samples 20 \
        --output_dir /scratch/connectome/connectome1/ko-centaur/results/full_eval
"""

import argparse
import json
import sys
from pathlib import Path
import torch
from datetime import datetime
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.load_baselines import BaselineModelManager
from evaluation.extract_features import extract_features_batch


def load_jsonl_dataset(path: str, n_samples: int = None):
    """Load JSONL dataset"""
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if n_samples and i >= n_samples:
                break
            samples.append(json.loads(line))
    return samples


def run_single_fold_with_inner_cv(
    train_features: torch.Tensor,
    train_labels: torch.Tensor,
    test_features: torch.Tensor,
    test_labels: torch.Tensor,
    alpha_grid: list,
    inner_cv_folds: int = 5
):
    """
    Run single LOO fold with nested inner CV for hyperparameter tuning

    Args:
        train_features: Training features (n-1 samples)
        train_labels: Training labels (n-1 samples)
        test_features: Test features (1 sample)
        test_labels: Test labels (1 sample)
        alpha_grid: List of C values to try (C = 1/alpha in sklearn)
        inner_cv_folds: Number of folds for inner CV

    Returns:
        Dict with test accuracy, best alpha, and predictions
    """
    # Convert to numpy
    X_train = train_features.cpu().numpy()
    y_train = train_labels.cpu().numpy()
    X_test = test_features.cpu().numpy()
    y_test = test_labels.cpu().numpy()

    # Normalize features using training statistics
    train_mean = X_train.mean(axis=0)
    train_std = X_train.std(axis=0)
    # Only normalize features with non-zero std, keep constant features as-is
    train_std = np.where(train_std > 0, train_std, 1.0)
    X_train_norm = (X_train - train_mean) / train_std
    X_test_norm = (X_test - train_mean) / train_std

    # Inner CV for hyperparameter selection
    if len(X_train) < inner_cv_folds:
        # If too few samples, just try each alpha without CV
        best_alpha = alpha_grid[0]
        best_score = 0.0

        for alpha in alpha_grid:
            C = 1.0 / alpha if alpha > 0 else 1e10
            model = LogisticRegression(
                C=C,
                max_iter=1000,
                random_state=42,
                solver='lbfgs'
            )
            model.fit(X_train_norm, y_train)
            score = model.score(X_train_norm, y_train)

            if score > best_score:
                best_score = score
                best_alpha = alpha
    else:
        # Use KFold CV for hyperparameter selection
        kf = KFold(n_splits=min(inner_cv_folds, len(X_train)), shuffle=True, random_state=42)

        best_alpha = alpha_grid[0]
        best_score = 0.0

        for alpha in alpha_grid:
            C = 1.0 / alpha if alpha > 0 else 1e10

            fold_scores = []
            for inner_train_idx, inner_val_idx in kf.split(X_train_norm):
                X_inner_train = X_train_norm[inner_train_idx]
                y_inner_train = y_train[inner_train_idx]
                X_inner_val = X_train_norm[inner_val_idx]
                y_inner_val = y_train[inner_val_idx]

                # Skip fold if it has only one class (can't train binary classifier)
                if len(np.unique(y_inner_train)) < 2:
                    continue

                try:
                    model = LogisticRegression(
                        C=C,
                        max_iter=1000,
                        random_state=42,
                        solver='lbfgs'
                    )
                    model.fit(X_inner_train, y_inner_train)
                    score = model.score(X_inner_val, y_inner_val)
                    fold_scores.append(score)
                except ValueError:
                    # Skip fold if fitting fails (e.g., only one class)
                    continue

            # Only update if we got valid scores from at least one fold
            if len(fold_scores) > 0:
                mean_score = np.mean(fold_scores)
                if mean_score > best_score:
                    best_score = mean_score
                    best_alpha = alpha

    # Train final model with best alpha on full training set
    C = 1.0 / best_alpha if best_alpha > 0 else 1e10
    final_model = LogisticRegression(
        C=C,
        max_iter=1000,
        random_state=42,
        solver='lbfgs'
    )
    final_model.fit(X_train_norm, y_train)

    # Evaluate on test set
    test_pred = final_model.predict(X_test_norm)
    test_accuracy = (test_pred == y_test).mean()

    # Get prediction probabilities
    test_proba = final_model.predict_proba(X_test_norm)
    test_log_likelihood = np.log(test_proba[0, y_test[0]] + 1e-10)

    return {
        'test_accuracy': float(test_accuracy),
        'test_prediction': int(test_pred[0]),
        'test_true_label': int(y_test[0]),
        'test_log_likelihood': float(test_log_likelihood),
        'best_alpha': float(best_alpha),
        'best_C': float(C)
    }


def main():
    parser = argparse.ArgumentParser(description='Ko-CENTaUR Full Evaluation with LOO CV')
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to Psych-101 JSONL dataset'
    )
    parser.add_argument(
        '--baselines',
        nargs='+',
        default=['exaone-base'],
        help='Baseline model names'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=20,
        help='Number of samples for LOO CV (default: 20)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/scratch/connectome/connectome1/ko-centaur/results/full_eval',
        help='Output directory for results'
    )
    parser.add_argument(
        '--skip_features',
        action='store_true',
        help='Skip feature extraction (use cached features)'
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Ko-CENTaUR Full LOO Cross-Validation")
    print("="*70)
    print(f"Start time: {datetime.now()}")
    print(f"Dataset: {args.dataset}")
    print(f"Baselines: {', '.join(args.baselines)}")
    print(f"Samples: {args.n_samples}")
    print(f"Output: {output_dir}")
    print()

    # Alpha grid for regularization (C = 1/alpha)
    alpha_grid = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]

    # Load dataset
    print(f"[1/5] Loading dataset...")
    try:
        dataset = load_jsonl_dataset(args.dataset, args.n_samples)
        print(f"✓ Loaded {len(dataset)} samples")

        # Check for labels
        if 'label' in dataset[0]:
            labels = torch.tensor([s['label'] for s in dataset])
        elif 'choice' in dataset[0]:
            labels = torch.tensor([s['choice'] for s in dataset])
        else:
            print("⚠ No labels found, using zeros")
            labels = torch.zeros(len(dataset), dtype=torch.long)

    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return 1

    # Extract or load features
    features_cache_dir = output_dir / "features"
    features_cache_dir.mkdir(exist_ok=True)

    if not args.skip_features:
        # Load Ko-CENTaUR
        print(f"\n[2/5] Loading Ko-CENTaUR...")
        try:
            kocentaur = BaselineModelManager("ko-centaur")
            print(f"✓ Ko-CENTaUR loaded")
        except Exception as e:
            print(f"✗ Error loading Ko-CENTaUR: {e}")
            return 1

        # Extract Ko-CENTaUR features
        print(f"\n[3/5] Extracting Ko-CENTaUR features...")
        try:
            kocentaur_features = extract_features_batch(kocentaur, dataset)
            print(f"✓ Ko-CENTaUR features: {kocentaur_features.shape}")

            # Cache features
            torch.save(kocentaur_features, features_cache_dir / "ko_centaur_features.pth")
            print(f"✓ Features cached")

        except Exception as e:
            print(f"✗ Error extracting Ko-CENTaUR features: {e}")
            import traceback
            traceback.print_exc()
            return 1

        # Extract baseline features
        print(f"\n[4/5] Extracting baseline features...")
        baseline_features = {}
        for baseline_name in args.baselines:
            try:
                print(f"  Loading {baseline_name}...")
                baseline = BaselineModelManager(baseline_name)

                print(f"  Extracting features...")
                features = extract_features_batch(baseline, dataset)
                baseline_features[baseline_name] = features
                print(f"  ✓ {baseline_name}: {features.shape}")

                # Cache features
                torch.save(features, features_cache_dir / f"{baseline_name}_features.pth")

            except Exception as e:
                print(f"  ✗ Error with {baseline_name}: {e}")
                continue

    else:
        # Load cached features
        print(f"\n[2-4/5] Loading cached features...")
        try:
            kocentaur_features = torch.load(features_cache_dir / "ko_centaur_features.pth")
            print(f"✓ Ko-CENTaUR features: {kocentaur_features.shape}")

            baseline_features = {}
            for baseline_name in args.baselines:
                features = torch.load(features_cache_dir / f"{baseline_name}_features.pth")
                baseline_features[baseline_name] = features
                print(f"✓ {baseline_name} features: {features.shape}")

        except Exception as e:
            print(f"✗ Error loading cached features: {e}")
            return 1

    # Run LOO cross-validation
    print(f"\n[5/5] Running Leave-One-Out Cross-Validation ({args.n_samples} folds)...")

    # Ko-CENTaUR LOO CV
    print(f"\nKo-CENTaUR LOO CV:")
    kocentaur_results = []

    for fold_id in range(args.n_samples):
        # Create LOO split
        train_idx = [i for i in range(args.n_samples) if i != fold_id]
        test_idx = fold_id

        train_features = kocentaur_features[train_idx]
        train_labels = labels[train_idx]
        test_features = kocentaur_features[test_idx:test_idx+1]
        test_labels = labels[test_idx:test_idx+1]

        # Run fold with inner CV
        result = run_single_fold_with_inner_cv(
            train_features, train_labels,
            test_features, test_labels,
            alpha_grid
        )
        result['fold_id'] = fold_id
        kocentaur_results.append(result)

        if (fold_id + 1) % 5 == 0:
            print(f"  Completed {fold_id + 1}/{args.n_samples} folds")

    # Aggregate Ko-CENTaUR results
    kocentaur_accuracies = [r['test_accuracy'] for r in kocentaur_results]
    kocentaur_mean_acc = np.mean(kocentaur_accuracies)
    kocentaur_std_acc = np.std(kocentaur_accuracies, ddof=1)

    print(f"✓ Ko-CENTaUR Mean Accuracy: {kocentaur_mean_acc:.3f} ± {kocentaur_std_acc:.3f}")

    # Save Ko-CENTaUR results
    kocentaur_dir = output_dir / "ko-centaur"
    kocentaur_dir.mkdir(exist_ok=True)
    torch.save({
        'fold_results': kocentaur_results,
        'mean_accuracy': kocentaur_mean_acc,
        'std_accuracy': kocentaur_std_acc,
        'n_folds': args.n_samples
    }, kocentaur_dir / "loo_results.pth")

    # Run baseline LOO CV
    all_results = {'ko-centaur': {
        'mean_accuracy': kocentaur_mean_acc,
        'std_accuracy': kocentaur_std_acc,
        'fold_results': kocentaur_results
    }}

    for baseline_name in args.baselines:
        if baseline_name not in baseline_features:
            continue

        print(f"\n{baseline_name} LOO CV:")
        baseline_results = []

        for fold_id in range(args.n_samples):
            # Create LOO split
            train_idx = [i for i in range(args.n_samples) if i != fold_id]
            test_idx = fold_id

            train_features = baseline_features[baseline_name][train_idx]
            train_labels = labels[train_idx]
            test_features = baseline_features[baseline_name][test_idx:test_idx+1]
            test_labels = labels[test_idx:test_idx+1]

            # Run fold with inner CV
            result = run_single_fold_with_inner_cv(
                train_features, train_labels,
                test_features, test_labels,
                alpha_grid
            )
            result['fold_id'] = fold_id
            baseline_results.append(result)

            if (fold_id + 1) % 5 == 0:
                print(f"  Completed {fold_id + 1}/{args.n_samples} folds")

        # Aggregate baseline results
        baseline_accuracies = [r['test_accuracy'] for r in baseline_results]
        baseline_mean_acc = np.mean(baseline_accuracies)
        baseline_std_acc = np.std(baseline_accuracies, ddof=1)

        print(f"✓ {baseline_name} Mean Accuracy: {baseline_mean_acc:.3f} ± {baseline_std_acc:.3f}")

        # Save baseline results
        baseline_dir = output_dir / baseline_name
        baseline_dir.mkdir(exist_ok=True)
        torch.save({
            'fold_results': baseline_results,
            'mean_accuracy': baseline_mean_acc,
            'std_accuracy': baseline_std_acc,
            'n_folds': args.n_samples
        }, baseline_dir / "loo_results.pth")

        all_results[baseline_name] = {
            'mean_accuracy': baseline_mean_acc,
            'std_accuracy': baseline_std_acc,
            'fold_results': baseline_results
        }

    # Save aggregated results
    print(f"\nSaving aggregated results...")
    torch.save(all_results, output_dir / "all_results.pth")

    # Print summary
    print(f"\n{'='*70}")
    print("Summary")
    print("="*70)
    for model_name, results in all_results.items():
        print(f"{model_name}: {results['mean_accuracy']:.3f} ± {results['std_accuracy']:.3f}")

    print(f"\nComplete! Results saved to: {output_dir}")
    print(f"End time: {datetime.now()}")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
