#!/usr/bin/env python3
"""
Create k-fold cross-validation splits for choices13k_1000 dataset

Implements stratified k-fold splitting to maintain class balance across folds.
Written using TDD (Test-Driven Development) - see tests/test_create_kfold_splits.py

Usage:
    python scripts/create_kfold_splits.py \
        --data data/choices13k_1000.jsonl \
        --n_splits 10 \
        --output_dir data/kfold_splits \
        --seed 42
"""

import argparse
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from sklearn.model_selection import StratifiedKFold


def load_data(data_path: str) -> List[Dict]:
    """
    Load JSONL dataset

    Args:
        data_path: Path to JSONL file

    Returns:
        List of data samples
    """
    data = []
    with open(data_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


def create_stratified_kfold(
    data_path: str,
    n_splits: int = 10,
    seed: int = 42
) -> List[Dict[str, List[int]]]:
    """
    Create stratified k-fold cross-validation splits

    Maintains class balance across folds to ensure representative training/test sets.

    Args:
        data_path: Path to JSONL data file
        n_splits: Number of folds (default: 10)
        seed: Random seed for reproducibility (default: 42)

    Returns:
        List of fold dictionaries, each containing:
        - 'train': List of training indices
        - 'test': List of test indices
        - 'fold_id': Fold number (0-indexed)
        - 'train_size': Number of training samples
        - 'test_size': Number of test samples
        - 'train_class_balance': Class distribution in training set
        - 'test_class_balance': Class distribution in test set
    """
    # Load data
    data = load_data(data_path)

    # Extract labels for stratification
    labels = np.array([sample['choice'] for sample in data])

    # Create stratified k-fold splitter
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    # Generate folds
    folds = []
    for fold_id, (train_idx, test_idx) in enumerate(skf.split(np.zeros(len(labels)), labels)):
        # Calculate class balance
        train_labels = labels[train_idx]
        test_labels = labels[test_idx]

        train_class_0 = int(np.sum(train_labels == 0))
        train_class_1 = int(np.sum(train_labels == 1))
        test_class_0 = int(np.sum(test_labels == 0))
        test_class_1 = int(np.sum(test_labels == 1))

        fold = {
            'fold_id': fold_id,
            'train': train_idx.tolist(),
            'test': test_idx.tolist(),
            'train_size': len(train_idx),
            'test_size': len(test_idx),
            'train_class_balance': {
                'class_0': train_class_0,
                'class_1': train_class_1,
                'class_0_pct': train_class_0 / len(train_idx) * 100,
                'class_1_pct': train_class_1 / len(train_idx) * 100
            },
            'test_class_balance': {
                'class_0': test_class_0,
                'class_1': test_class_1,
                'class_0_pct': test_class_0 / len(test_idx) * 100,
                'class_1_pct': test_class_1 / len(test_idx) * 100
            }
        }

        folds.append(fold)

    return folds


def save_folds(folds: List[Dict], output_dir: str):
    """
    Save folds to JSON files

    Args:
        folds: List of fold dictionaries
        output_dir: Directory to save fold files
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Save each fold
    for fold in folds:
        fold_path = os.path.join(output_dir, f"fold_{fold['fold_id']}.json")
        with open(fold_path, 'w', encoding='utf-8') as f:
            json.dump(fold, f, indent=2)

    print(f"✓ Saved {len(folds)} folds to {output_dir}/")


def validate_folds(folds: List[Dict], data: List[Dict]) -> Dict:
    """
    Validate k-fold splits and compute quality metrics

    Args:
        folds: List of fold dictionaries
        data: Original dataset

    Returns:
        Dictionary with validation metrics
    """
    n_samples = len(data)
    n_splits = len(folds)

    # Check all indices used exactly once as test
    all_test_indices = set()
    for fold in folds:
        all_test_indices.update(fold['test'])

    coverage = len(all_test_indices) / n_samples

    # Compute fold size statistics
    test_sizes = [fold['test_size'] for fold in folds]
    train_sizes = [fold['train_size'] for fold in folds]

    # Compute class balance statistics
    test_class_0_pcts = [fold['test_class_balance']['class_0_pct'] for fold in folds]
    test_class_1_pcts = [fold['test_class_balance']['class_1_pct'] for fold in folds]

    # Overall class balance
    all_labels = [sample['choice'] for sample in data]
    overall_class_0 = sum(1 for c in all_labels if c == 0)
    overall_class_1 = sum(1 for c in all_labels if c == 1)

    metrics = {
        'n_splits': n_splits,
        'total_samples': n_samples,
        'coverage': coverage,
        'all_samples_tested': coverage == 1.0,

        'fold_sizes': {
            'test_min': min(test_sizes),
            'test_max': max(test_sizes),
            'test_mean': np.mean(test_sizes),
            'test_std': np.std(test_sizes),
            'train_min': min(train_sizes),
            'train_max': max(train_sizes),
            'train_mean': np.mean(train_sizes),
            'train_std': np.std(train_sizes),
        },

        'class_balance': {
            'overall_class_0': overall_class_0,
            'overall_class_1': overall_class_1,
            'overall_class_0_pct': overall_class_0 / n_samples * 100,
            'overall_class_1_pct': overall_class_1 / n_samples * 100,

            'test_class_0_pct_mean': np.mean(test_class_0_pcts),
            'test_class_0_pct_std': np.std(test_class_0_pcts),
            'test_class_0_pct_min': np.min(test_class_0_pcts),
            'test_class_0_pct_max': np.max(test_class_0_pcts),

            'test_class_1_pct_mean': np.mean(test_class_1_pcts),
            'test_class_1_pct_std': np.std(test_class_1_pcts),
            'test_class_1_pct_min': np.min(test_class_1_pcts),
            'test_class_1_pct_max': np.max(test_class_1_pcts),
        }
    }

    return metrics


def print_validation_report(metrics: Dict):
    """Print validation report"""
    print(f"\n{'='*60}")
    print("K-FOLD VALIDATION REPORT")
    print(f"{'='*60}")

    print(f"\nOverall:")
    print(f"  Splits: {metrics['n_splits']}")
    print(f"  Total samples: {metrics['total_samples']}")
    print(f"  Coverage: {metrics['coverage']:.1%}")
    if metrics['all_samples_tested']:
        print(f"  ✅ All samples tested exactly once")
    else:
        print(f"  ❌ Warning: Not all samples tested")

    print(f"\nFold Sizes:")
    print(f"  Test set: {metrics['fold_sizes']['test_mean']:.1f} ± {metrics['fold_sizes']['test_std']:.1f}")
    print(f"    Range: [{metrics['fold_sizes']['test_min']}, {metrics['fold_sizes']['test_max']}]")
    print(f"  Train set: {metrics['fold_sizes']['train_mean']:.1f} ± {metrics['fold_sizes']['train_std']:.1f}")
    print(f"    Range: [{metrics['fold_sizes']['train_min']}, {metrics['fold_sizes']['train_max']}]")

    print(f"\nClass Balance:")
    cb = metrics['class_balance']
    print(f"  Overall:")
    print(f"    Class 0: {cb['overall_class_0']} ({cb['overall_class_0_pct']:.1f}%)")
    print(f"    Class 1: {cb['overall_class_1']} ({cb['overall_class_1_pct']:.1f}%)")

    print(f"\n  Test sets (stratified):")
    print(f"    Class 0: {cb['test_class_0_pct_mean']:.1f}% ± {cb['test_class_0_pct_std']:.1f}%")
    print(f"      Range: [{cb['test_class_0_pct_min']:.1f}%, {cb['test_class_0_pct_max']:.1f}%]")
    print(f"    Class 1: {cb['test_class_1_pct_mean']:.1f}% ± {cb['test_class_1_pct_std']:.1f}%")
    print(f"      Range: [{cb['test_class_1_pct_min']:.1f}%, {cb['test_class_1_pct_max']:.1f}%]")

    # Check if stratification is good (< 2% deviation)
    class_0_deviation = abs(cb['test_class_0_pct_mean'] - cb['overall_class_0_pct'])
    if class_0_deviation < 2.0:
        print(f"\n  ✅ Stratification quality: Excellent (deviation < 2%)")
    elif class_0_deviation < 5.0:
        print(f"\n  ⚠️  Stratification quality: Good (deviation < 5%)")
    else:
        print(f"\n  ❌ Stratification quality: Poor (deviation >= 5%)")

    print(f"\n{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description='Create k-fold CV splits')
    parser.add_argument(
        '--data',
        type=str,
        default='data/choices13k_1000.jsonl',
        help='Path to data file'
    )
    parser.add_argument(
        '--n_splits',
        type=int,
        default=10,
        help='Number of folds (default: 10)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='data/kfold_splits',
        help='Output directory for fold files'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    print(f"{'='*60}")
    print("CREATE K-FOLD CROSS-VALIDATION SPLITS")
    print(f"{'='*60}")
    print(f"\nParameters:")
    print(f"  Data: {args.data}")
    print(f"  Number of splits: {args.n_splits}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Random seed: {args.seed}")

    # Load data for validation
    print(f"\nLoading data...")
    data = load_data(args.data)
    print(f"✓ Loaded {len(data)} samples")

    # Create folds
    print(f"\nCreating {args.n_splits}-fold stratified splits...")
    folds = create_stratified_kfold(
        data_path=args.data,
        n_splits=args.n_splits,
        seed=args.seed
    )
    print(f"✓ Created {len(folds)} folds")

    # Validate folds
    print(f"\nValidating folds...")
    metrics = validate_folds(folds, data)
    print_validation_report(metrics)

    # Save folds
    print(f"Saving folds...")
    save_folds(folds, args.output_dir)

    # Save validation metrics
    metrics_path = os.path.join(args.output_dir, 'validation_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved validation metrics to {metrics_path}")

    print(f"\n{'='*60}")
    print("COMPLETE")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
