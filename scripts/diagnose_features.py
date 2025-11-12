#!/usr/bin/env python3
"""
Feature Extraction Diagnostics
Investigates why fine-tuned models perform worse than random baseline
"""

import torch
import numpy as np
from pathlib import Path
import sys

sys.path.append('/scratch/connectome/connectome1/ko-centaur')


def analyze_features(feature_path, model_name):
    """
    Analyze extracted features for anomalies

    Checks:
    1. Feature statistics (mean, std, range)
    2. NaN/Inf values
    3. Feature dimensionality
    4. Label distribution
    5. Feature diversity (all same vs varied)
    """
    print(f"\n{'='*80}")
    print(f"Analyzing: {model_name}")
    print(f"{'='*80}\n")

    # Load features
    print(f"1. Loading features from {feature_path}")
    data = torch.load(feature_path, weights_only=False)

    features = data['features']  # (n_samples, hidden_dim)
    labels = data['labels']      # (n_samples,)

    print(f"   Features shape: {features.shape}")
    print(f"   Labels shape:   {labels.shape}")
    print(f"   Hidden dim:     {features.shape[1]}")

    # 2. Check for NaN/Inf
    print(f"\n2. Checking for invalid values")
    has_nan = torch.isnan(features).any()
    has_inf = torch.isinf(features).any()
    print(f"   NaN values:  {'❌ FOUND' if has_nan else '✅ None'}")
    print(f"   Inf values:  {'❌ FOUND' if has_inf else '✅ None'}")

    if has_nan or has_inf:
        print(f"   ⚠️  CRITICAL: Invalid values detected!")
        return

    # 3. Feature statistics
    print(f"\n3. Feature statistics")
    feat_mean = features.mean(dim=0)  # Mean per dimension
    feat_std = features.std(dim=0)    # Std per dimension
    feat_min = features.min(dim=0).values
    feat_max = features.max(dim=0).values

    print(f"   Global mean:     {feat_mean.mean().item():.6f}")
    print(f"   Global std:      {feat_std.mean().item():.6f}")
    print(f"   Global range:    [{feat_min.min().item():.4f}, {feat_max.max().item():.4f}]")

    # Check if features are all zeros or all same
    if features.abs().max() < 1e-6:
        print(f"   ⚠️  WARNING: Features are near zero!")

    # Check feature diversity
    sample_var = features.var(dim=0)  # Variance per dimension
    low_var_dims = (sample_var < 1e-6).sum().item()
    print(f"   Low variance dims: {low_var_dims}/{features.shape[1]}")

    if low_var_dims > features.shape[1] * 0.5:
        print(f"   ⚠️  WARNING: >50% dimensions have low variance!")

    # 4. Label distribution
    print(f"\n4. Label distribution")
    n_A = (labels == 0).sum().item()
    n_B = (labels == 1).sum().item()
    print(f"   Choice A: {n_A} ({n_A/len(labels)*100:.1f}%)")
    print(f"   Choice B: {n_B} ({n_B/len(labels)*100:.1f}%)")

    # 5. Pairwise sample similarity (check if all samples are similar)
    print(f"\n5. Feature diversity check")

    # Compute pairwise distances for first 10 samples
    if len(features) >= 10:
        subset = features[:10]
        # Normalize for fair comparison
        subset_norm = subset / (subset.norm(dim=1, keepdim=True) + 1e-8)
        similarity = torch.mm(subset_norm, subset_norm.t())

        # Off-diagonal similarities
        off_diag = similarity - torch.eye(10)
        avg_sim = off_diag.abs().mean().item()
        max_sim = off_diag.abs().max().item()

        print(f"   Avg pairwise similarity: {avg_sim:.4f}")
        print(f"   Max pairwise similarity: {max_sim:.4f}")

        if avg_sim > 0.99:
            print(f"   ⚠️  WARNING: Samples are very similar (avg sim > 0.99)!")

    # 6. Sample a few feature vectors
    print(f"\n6. Sample feature vectors (first 3 dimensions, first 3 samples)")
    for i in range(min(3, len(features))):
        sample = features[i, :3]
        label = labels[i].item()
        print(f"   Sample {i} (label={label}): [{sample[0]:.6f}, {sample[1]:.6f}, {sample[2]:.6f}, ...]")

    # 7. Check metadata
    print(f"\n7. Metadata")
    for key in ['model_name', 'n_samples', 'extraction_time', 'quantized', 'timestamp']:
        if key in data:
            print(f"   {key}: {data[key]}")

    print(f"\n{'='*80}\n")

    return {
        'shape': features.shape,
        'has_nan': has_nan,
        'has_inf': has_inf,
        'mean': feat_mean.mean().item(),
        'std': feat_std.mean().item(),
        'range': (feat_min.min().item(), feat_max.max().item()),
        'low_var_dims': low_var_dims,
        'label_dist': (n_A, n_B)
    }


def compare_features(qwen25_path, qwen25_base_path, deepseek_path):
    """
    Compare fine-tuned vs base model features
    """
    print(f"\n{'='*80}")
    print(f"COMPARATIVE ANALYSIS")
    print(f"{'='*80}\n")

    results = {}

    # Analyze each model
    if Path(qwen25_path).exists():
        results['qwen25'] = analyze_features(qwen25_path, "Qwen2.5-32B-QLoRA")

    if Path(qwen25_base_path).exists():
        results['qwen25_base'] = analyze_features(qwen25_base_path, "Qwen2.5-32B-Base")

    if Path(deepseek_path).exists():
        results['deepseek'] = analyze_features(deepseek_path, "DeepSeek-R1-32B-QLoRA")

    # Compare fine-tuned vs base
    if 'qwen25' in results and 'qwen25_base' in results:
        print(f"\n{'='*80}")
        print(f"QWEN2.5: Fine-tuned vs Base Comparison")
        print(f"{'='*80}\n")

        ft = results['qwen25']
        base = results['qwen25_base']

        print(f"Feature dimensionality:")
        print(f"  Fine-tuned: {ft['shape'][1]}")
        print(f"  Base:       {base['shape'][1]}")

        if ft['shape'][1] != base['shape'][1]:
            print(f"  ⚠️  WARNING: Dimension mismatch!")

        print(f"\nFeature statistics:")
        print(f"  Fine-tuned mean: {ft['mean']:.6f}")
        print(f"  Base mean:       {base['mean']:.6f}")
        print(f"  Difference:      {abs(ft['mean'] - base['mean']):.6f}")

        print(f"\n  Fine-tuned std: {ft['std']:.6f}")
        print(f"  Base std:       {base['std']:.6f}")
        print(f"  Difference:     {abs(ft['std'] - base['std']):.6f}")

        print(f"\nLow variance dimensions:")
        print(f"  Fine-tuned: {ft['low_var_dims']}/{ft['shape'][1]} ({ft['low_var_dims']/ft['shape'][1]*100:.1f}%)")
        print(f"  Base:       {base['low_var_dims']}/{base['shape'][1]} ({base['low_var_dims']/base['shape'][1]*100:.1f}%)")

    return results


def check_alpha_selection(loo_cv_path, model_name):
    """
    Analyze alpha selection patterns
    """
    print(f"\n{'='*80}")
    print(f"Alpha Selection Analysis: {model_name}")
    print(f"{'='*80}\n")

    if not Path(loo_cv_path).exists():
        print(f"❌ File not found: {loo_cv_path}")
        return

    data = torch.load(loo_cv_path, weights_only=False)

    print(f"1. Overall results")
    print(f"   Average NLL: {data['avg_nll']:.4f} ± {data['std_nll']:.4f}")
    print(f"   Accuracy:    {data['accuracy']:.1%}")

    # Check alpha distribution
    if 'alpha_counts' in data:
        print(f"\n2. Alpha selection distribution")
        alpha_counts = data['alpha_counts']

        # Sort by alpha value
        sorted_alphas = sorted(alpha_counts.items(), key=lambda x: x[0])

        for alpha, count in sorted_alphas:
            pct = count / data['n_samples'] * 100
            bar = '█' * int(pct / 2)  # Scale to 50 chars max
            print(f"   alpha={alpha:6.4f}: {count:3d} folds ({pct:5.1f}%) {bar}")

        # Check if all folds selected same alpha
        max_count = max(alpha_counts.values())
        if max_count == data['n_samples']:
            selected_alpha = [a for a, c in alpha_counts.items() if c == max_count][0]
            print(f"\n   ⚠️  ALL {data['n_samples']} folds selected alpha={selected_alpha}!")
            print(f"   This suggests:")
            if selected_alpha == 0.0:
                print(f"     - Model might be underfitting (no regularization needed)")
            elif selected_alpha >= 0.1:
                print(f"     - Model might be overfitting (max regularization needed)")
                print(f"     - OR features are not informative")

    # Check for convergence issues
    if 'test_nlls' in data:
        test_nlls = data['test_nlls']
        print(f"\n3. NLL distribution across folds")
        print(f"   Min NLL:  {min(test_nlls):.4f}")
        print(f"   Max NLL:  {max(test_nlls):.4f}")
        print(f"   Median:   {np.median(test_nlls):.4f}")

        # Check for outliers
        q1 = np.percentile(test_nlls, 25)
        q3 = np.percentile(test_nlls, 75)
        iqr = q3 - q1
        outliers = [nll for nll in test_nlls if nll < q1 - 1.5*iqr or nll > q3 + 1.5*iqr]

        if outliers:
            print(f"   Outliers: {len(outliers)} folds")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    data_dir = Path("/scratch/connectome/connectome1/ko-centaur/data")

    # Feature paths
    qwen25_path = data_dir / "features/centaur_features_qwen25.pth"
    qwen25_base_path = data_dir / "features/centaur_features_qwen25_base.pth"
    deepseek_path = data_dir / "features/centaur_features_deepseek.pth"

    # LOO CV results paths
    qwen25_loo_path = data_dir / "loo_cv_results_qwen25.pth"
    deepseek_loo_path = data_dir / "results/loo_cv_results_deepseek.pth"

    print("\n" + "="*80)
    print("CENTaUR FEATURE EXTRACTION DIAGNOSTICS")
    print("="*80)

    # 1. Analyze and compare features
    feature_results = compare_features(qwen25_path, qwen25_base_path, deepseek_path)

    # 2. Analyze alpha selection
    check_alpha_selection(qwen25_loo_path, "Qwen2.5-32B-QLoRA")
    check_alpha_selection(deepseek_loo_path, "DeepSeek-R1-32B-QLoRA")

    print("\n" + "="*80)
    print("DIAGNOSTIC COMPLETE")
    print("="*80 + "\n")
