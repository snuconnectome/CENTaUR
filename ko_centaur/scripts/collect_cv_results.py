#!/usr/bin/env python3
"""
Collect Cross-Validation Results

Aggregates results from SLURM array job folds into final evaluation metrics.
Configured for connectome server.

Usage:
    python scripts/collect_cv_results.py \
        --input_dir /scratch/connectome/connectome1/ko-centaur/results/full_eval \
        --n_folds 100 \
        --output /scratch/connectome/connectome1/ko-centaur/results/full_eval/aggregated_results.pth
"""

import argparse
import json
import sys
from pathlib import Path
import torch
from datetime import datetime
import numpy as np

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.compare_statistical import (
    compute_paired_ttest,
    compute_cohens_d,
    generate_statistical_report
)


def collect_model_results(input_dir: Path, model_name: str, n_folds: int):
    """
    Collect results for a single model across all folds

    Returns:
        dict with keys: fold_results, accuracies, log_likelihoods, mean_accuracy, std_accuracy
    """
    fold_results = []
    accuracies = []
    log_likelihoods = []
    missing_folds = []

    model_dir = input_dir / model_name

    for fold_id in range(n_folds):
        fold_path = model_dir / f"fold_{fold_id}.pth"

        if not fold_path.exists():
            missing_folds.append(fold_id)
            continue

        try:
            result = torch.load(fold_path)
            fold_results.append(result)
            accuracies.append(result['test_accuracy'])

            if 'test_log_likelihood' in result:
                log_likelihoods.append(result['test_log_likelihood'])

        except Exception as e:
            print(f"⚠ Error loading fold {fold_id}: {e}")
            missing_folds.append(fold_id)

    if missing_folds:
        print(f"⚠ {model_name}: Missing {len(missing_folds)} folds: {missing_folds[:10]}...")

    return {
        'model_name': model_name,
        'n_folds': len(fold_results),
        'n_folds_expected': n_folds,
        'missing_folds': missing_folds,
        'fold_results': fold_results,
        'accuracies': accuracies,
        'log_likelihoods': log_likelihoods if log_likelihoods else None,
        'mean_accuracy': np.mean(accuracies) if accuracies else 0,
        'std_accuracy': np.std(accuracies) if accuracies else 0,
        'mean_log_likelihood': np.mean(log_likelihoods) if log_likelihoods else None,
        'std_log_likelihood': np.std(log_likelihoods) if log_likelihoods else None,
    }


def main():
    parser = argparse.ArgumentParser(description='Collect CV Results from SLURM Array Job')
    parser.add_argument(
        '--input_dir',
        type=str,
        required=True,
        help='Input directory containing fold results'
    )
    parser.add_argument(
        '--n_folds',
        type=int,
        default=100,
        help='Expected number of folds'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output path for aggregated results (default: input_dir/aggregated_results.pth)'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=None,
        help='Model names to collect (default: auto-detect from directories)'
    )

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"✗ Input directory does not exist: {input_dir}")
        return 1

    print("="*70)
    print("Collecting Cross-Validation Results")
    print("="*70)
    print(f"Input directory: {input_dir}")
    print(f"Expected folds: {args.n_folds}")
    print()

    # Auto-detect model directories if not specified
    if args.models is None:
        model_dirs = [d for d in input_dir.iterdir() if d.is_dir() and d.name not in ['features', 'logs', 'tmp']]
        args.models = [d.name for d in model_dirs]
        print(f"Auto-detected models: {', '.join(args.models)}")
    else:
        print(f"Collecting models: {', '.join(args.models)}")

    print()

    # Collect results for each model
    all_results = {}

    for model_name in args.models:
        print(f"Collecting {model_name}...")
        results = collect_model_results(input_dir, model_name, args.n_folds)

        print(f"  Folds completed: {results['n_folds']}/{results['n_folds_expected']}")
        print(f"  Mean accuracy: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")

        if results['mean_log_likelihood'] is not None:
            print(f"  Mean log-likelihood: {results['mean_log_likelihood']:.4f} ± {results['std_log_likelihood']:.4f}")

        all_results[model_name] = results
        print()

    # Generate pairwise comparisons
    if len(all_results) > 1:
        print("Pairwise Statistical Comparisons")
        print("-" * 70)

        model_names = list(all_results.keys())
        comparisons = {}

        for i, model_a in enumerate(model_names):
            for model_b in model_names[i+1:]:
                acc_a = all_results[model_a]['accuracies']
                acc_b = all_results[model_b]['accuracies']

                # Ensure same length for paired test
                min_len = min(len(acc_a), len(acc_b))
                acc_a = acc_a[:min_len]
                acc_b = acc_b[:min_len]

                # Paired t-test
                t_stat, p_value = compute_paired_ttest(
                    np.array(acc_a), np.array(acc_b)
                )

                # Cohen's d
                cohens_d = compute_cohens_d(
                    np.array(acc_a), np.array(acc_b)
                )

                comparison_key = f"{model_a} vs {model_b}"
                comparisons[comparison_key] = {
                    't_statistic': float(t_stat),
                    'p_value': float(p_value),
                    'significant': p_value < 0.05,
                    'cohens_d': float(cohens_d),
                    'interpretation': 'large' if abs(cohens_d) >= 0.8 else ('medium' if abs(cohens_d) >= 0.5 else 'small')
                }

                print(f"\n{comparison_key}:")
                print(f"  t-statistic: {t_stat:.3f}")
                print(f"  p-value: {p_value:.4f} {'*' if p_value < 0.05 else ''}")
                print(f"  Cohen's d: {cohens_d:.3f} ({comparisons[comparison_key]['interpretation']})")

    # Determine best model
    best_model = max(all_results.keys(), key=lambda k: all_results[k]['mean_accuracy'])
    print()
    print("="*70)
    print(f"Best Model: {best_model}")
    print(f"Accuracy: {all_results[best_model]['mean_accuracy']:.4f} ± {all_results[best_model]['std_accuracy']:.4f}")
    print("="*70)

    # Save aggregated results
    if args.output is None:
        args.output = input_dir / "aggregated_results.pth"
    else:
        args.output = Path(args.output)

    output_data = {
        'timestamp': datetime.now().isoformat(),
        'n_folds_expected': args.n_folds,
        'model_results': all_results,
        'pairwise_comparisons': comparisons if len(all_results) > 1 else {},
        'best_model': best_model,
        'ranking': sorted(all_results.keys(), key=lambda k: all_results[k]['mean_accuracy'], reverse=True)
    }

    torch.save(output_data, args.output)
    print(f"\n✓ Aggregated results saved to: {args.output}")

    # Also save JSON version for easy inspection
    json_output = args.output.with_suffix('.json')
    with open(json_output, 'w', encoding='utf-8') as f:
        # Convert numpy arrays to lists for JSON serialization
        json_data = {
            'timestamp': output_data['timestamp'],
            'n_folds_expected': output_data['n_folds_expected'],
            'model_results': {
                k: {
                    'model_name': v['model_name'],
                    'n_folds': v['n_folds'],
                    'mean_accuracy': float(v['mean_accuracy']),
                    'std_accuracy': float(v['std_accuracy']),
                    'mean_log_likelihood': float(v['mean_log_likelihood']) if v['mean_log_likelihood'] is not None else None,
                    'std_log_likelihood': float(v['std_log_likelihood']) if v['std_log_likelihood'] is not None else None,
                    'missing_folds': v['missing_folds']
                }
                for k, v in output_data['model_results'].items()
            },
            'pairwise_comparisons': output_data['pairwise_comparisons'],
            'best_model': output_data['best_model'],
            'ranking': output_data['ranking']
        }
        json.dump(json_data, f, ensure_ascii=False, indent=2)

    print(f"✓ JSON version saved to: {json_output}")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
