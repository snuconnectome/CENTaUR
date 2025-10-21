#!/usr/bin/env python3
"""
Generate Statistical Reports

Creates comprehensive statistical analysis reports from aggregated CV results.
Configured for connectome server.

Usage:
    python scripts/generate_reports.py \
        --results_file /scratch/connectome/connectome1/ko-centaur/results/full_eval/aggregated_results.pth \
        --output_dir /scratch/connectome/connectome1/ko-centaur/results/reports
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
    compute_confidence_interval,
    bonferroni_correction
)


def generate_summary_table(model_results):
    """Generate summary table of model performance"""
    table = []
    table.append("="*90)
    table.append("Model Performance Summary")
    table.append("="*90)
    table.append(f"{'Model':<25} {'Accuracy':>15} {'Log-Likelihood':>15} {'Folds':>10}")
    table.append("-"*90)

    for model_name, results in sorted(model_results.items(), key=lambda x: x[1]['mean_accuracy'], reverse=True):
        acc = results['mean_accuracy']
        acc_std = results['std_accuracy']
        ll = results['mean_log_likelihood']
        ll_std = results['std_log_likelihood']
        n_folds = results['n_folds']

        acc_str = f"{acc:.4f} ± {acc_std:.4f}"
        ll_str = f"{ll:.4f} ± {ll_std:.4f}" if ll is not None else "N/A"

        table.append(f"{model_name:<25} {acc_str:>15} {ll_str:>15} {n_folds:>10}")

    table.append("="*90)

    return "\n".join(table)


def generate_comparison_table(comparisons, alpha=0.05):
    """Generate pairwise comparison table"""
    table = []
    table.append("\n" + "="*100)
    table.append("Pairwise Statistical Comparisons")
    table.append("="*100)
    table.append(f"{'Comparison':<35} {'t-statistic':>12} {'p-value':>12} {'Sig':>5} {'Cohen\'s d':>12} {'Effect':>10}")
    table.append("-"*100)

    for comp_name, comp_data in comparisons.items():
        t_stat = comp_data['t_statistic']
        p_val = comp_data['p_value']
        sig = "✓" if comp_data['significant'] else ""
        cohens_d = comp_data['cohens_d']
        effect = comp_data['interpretation']

        table.append(f"{comp_name:<35} {t_stat:>12.3f} {p_val:>12.4f} {sig:>5} {cohens_d:>12.3f} {effect:>10}")

    table.append("="*100)
    table.append(f"\nSignificance level: α = {alpha}")
    table.append("Effect size interpretation: |d| < 0.5 = small, 0.5-0.8 = medium, > 0.8 = large")

    return "\n".join(table)


def generate_ranking_table(ranking, model_results):
    """Generate model ranking table"""
    table = []
    table.append("\n" + "="*70)
    table.append("Model Ranking (by Accuracy)")
    table.append("="*70)
    table.append(f"{'Rank':<8} {'Model':<25} {'Accuracy':>15} {'95% CI':>20}")
    table.append("-"*70)

    for rank, model_name in enumerate(ranking, 1):
        results = model_results[model_name]
        acc = results['mean_accuracy']
        acc_std = results['std_accuracy']
        n = results['n_folds']

        # 95% confidence interval
        ci_low, ci_high = compute_confidence_interval(
            np.array(results['accuracies']), confidence=0.95
        )

        acc_str = f"{acc:.4f}"
        ci_str = f"[{ci_low:.4f}, {ci_high:.4f}]"

        table.append(f"{rank:<8} {model_name:<25} {acc_str:>15} {ci_str:>20}")

    table.append("="*70)

    return "\n".join(table)


def generate_detailed_statistics(model_results):
    """Generate detailed statistics for each model"""
    sections = []

    for model_name, results in model_results.items():
        section = []
        section.append(f"\n{'='*70}")
        section.append(f"Detailed Statistics: {model_name}")
        section.append("="*70)

        # Accuracy statistics
        section.append("\nAccuracy:")
        section.append(f"  Mean: {results['mean_accuracy']:.4f}")
        section.append(f"  Std Dev: {results['std_accuracy']:.4f}")
        section.append(f"  Min: {min(results['accuracies']):.4f}")
        section.append(f"  Max: {max(results['accuracies']):.4f}")
        section.append(f"  Median: {np.median(results['accuracies']):.4f}")

        # 95% Confidence Interval
        ci_low, ci_high = compute_confidence_interval(
            np.array(results['accuracies']), confidence=0.95
        )
        section.append(f"  95% CI: [{ci_low:.4f}, {ci_high:.4f}]")

        # Log-likelihood statistics (if available)
        if results['log_likelihoods']:
            section.append("\nLog-Likelihood:")
            section.append(f"  Mean: {results['mean_log_likelihood']:.4f}")
            section.append(f"  Std Dev: {results['std_log_likelihood']:.4f}")
            section.append(f"  Min: {min(results['log_likelihoods']):.4f}")
            section.append(f"  Max: {max(results['log_likelihoods']):.4f}")
            section.append(f"  Median: {np.median(results['log_likelihoods']):.4f}")

        # Cross-validation info
        section.append("\nCross-Validation:")
        section.append(f"  Folds completed: {results['n_folds']}/{results['n_folds_expected']}")
        if results['missing_folds']:
            section.append(f"  Missing folds: {len(results['missing_folds'])}")

        sections.append("\n".join(section))

    return "\n".join(sections)


def generate_markdown_report(
    model_results,
    comparisons,
    ranking,
    best_model,
    alpha=0.05
):
    """Generate comprehensive Markdown report"""

    report = []

    # Header
    report.append("# Ko-CENTaUR Evaluation Report")
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\nBest Model: **{best_model}**")
    report.append(f"Accuracy: {model_results[best_model]['mean_accuracy']:.4f} ± {model_results[best_model]['std_accuracy']:.4f}")

    # Executive Summary
    report.append("\n## Executive Summary\n")
    report.append(f"- Evaluated {len(model_results)} models")
    report.append(f"- {model_results[list(model_results.keys())[0]]['n_folds_expected']}-fold Leave-One-Out cross-validation")
    report.append(f"- Best model: {best_model}")
    report.append(f"- Number of pairwise comparisons: {len(comparisons)}")

    # Model Performance Table
    report.append("\n## Model Performance\n")
    report.append("| Rank | Model | Accuracy | 95% CI | Log-Likelihood |")
    report.append("|------|-------|----------|--------|----------------|")

    for rank, model_name in enumerate(ranking, 1):
        results = model_results[model_name]
        acc = results['mean_accuracy']
        ci_low, ci_high = compute_confidence_interval(
            np.array(results['accuracies']), confidence=0.95
        )
        ll = results['mean_log_likelihood']

        ll_str = f"{ll:.4f}" if ll is not None else "N/A"
        report.append(f"| {rank} | {model_name} | {acc:.4f} | [{ci_low:.4f}, {ci_high:.4f}] | {ll_str} |")

    # Pairwise Comparisons
    if comparisons:
        report.append("\n## Pairwise Statistical Comparisons\n")
        report.append(f"Significance level: α = {alpha}\n")
        report.append("| Comparison | t-statistic | p-value | Significant | Cohen's d | Effect Size |")
        report.append("|------------|-------------|---------|-------------|-----------|-------------|")

        for comp_name, comp_data in comparisons.items():
            sig = "✓" if comp_data['significant'] else ""
            report.append(
                f"| {comp_name} | {comp_data['t_statistic']:.3f} | "
                f"{comp_data['p_value']:.4f} | {sig} | "
                f"{comp_data['cohens_d']:.3f} | {comp_data['interpretation']} |"
            )

    # Detailed Statistics
    report.append("\n## Detailed Statistics\n")
    for model_name, results in model_results.items():
        report.append(f"\n### {model_name}\n")
        report.append(f"- **Accuracy**: {results['mean_accuracy']:.4f} ± {results['std_accuracy']:.4f}")
        report.append(f"- **Range**: [{min(results['accuracies']):.4f}, {max(results['accuracies']):.4f}]")
        report.append(f"- **Median**: {np.median(results['accuracies']):.4f}")
        report.append(f"- **Folds**: {results['n_folds']}/{results['n_folds_expected']}")

        if results['log_likelihoods']:
            report.append(f"- **Log-Likelihood**: {results['mean_log_likelihood']:.4f} ± {results['std_log_likelihood']:.4f}")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description='Generate Statistical Reports')
    parser.add_argument(
        '--results_file',
        type=str,
        required=True,
        help='Path to aggregated results file (.pth)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/scratch/connectome/connectome1/ko-centaur/results/reports',
        help='Output directory for reports'
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help='Significance level (default: 0.05)'
    )

    args = parser.parse_args()

    results_file = Path(args.results_file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Generating Statistical Reports")
    print("="*70)
    print(f"Results file: {results_file}")
    print(f"Output directory: {output_dir}")
    print()

    # Load aggregated results
    print("Loading results...")
    try:
        data = torch.load(results_file)
        model_results = data['model_results']
        comparisons = data.get('pairwise_comparisons', {})
        ranking = data['ranking']
        best_model = data['best_model']
        print(f"✓ Loaded results for {len(model_results)} models")
    except Exception as e:
        print(f"✗ Error loading results: {e}")
        return 1

    # Generate text report
    print("\nGenerating text report...")
    text_report = []
    text_report.append(generate_summary_table(model_results))
    text_report.append(generate_ranking_table(ranking, model_results))
    if comparisons:
        text_report.append(generate_comparison_table(comparisons, args.alpha))
    text_report.append(generate_detailed_statistics(model_results))

    text_output = output_dir / "statistical_report.txt"
    with open(text_output, 'w', encoding='utf-8') as f:
        f.write("\n".join(text_report))
    print(f"✓ Text report saved: {text_output}")

    # Generate Markdown report
    print("Generating Markdown report...")
    md_report = generate_markdown_report(
        model_results, comparisons, ranking, best_model, args.alpha
    )

    md_output = output_dir / "statistical_report.md"
    with open(md_output, 'w', encoding='utf-8') as f:
        f.write(md_report)
    print(f"✓ Markdown report saved: {md_output}")

    # Generate JSON report
    print("Generating JSON report...")
    json_data = {
        'timestamp': datetime.now().isoformat(),
        'alpha': args.alpha,
        'best_model': best_model,
        'ranking': ranking,
        'summary': {
            model_name: {
                'mean_accuracy': float(results['mean_accuracy']),
                'std_accuracy': float(results['std_accuracy']),
                'mean_log_likelihood': float(results['mean_log_likelihood']) if results['mean_log_likelihood'] is not None else None,
                'std_log_likelihood': float(results['std_log_likelihood']) if results['std_log_likelihood'] is not None else None,
                'n_folds': results['n_folds']
            }
            for model_name, results in model_results.items()
        },
        'pairwise_comparisons': comparisons
    }

    json_output = output_dir / "statistical_report.json"
    with open(json_output, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    print(f"✓ JSON report saved: {json_output}")

    # Print summary to console
    print()
    print(generate_summary_table(model_results))
    if comparisons:
        print(generate_comparison_table(comparisons, args.alpha))

    print()
    print("="*70)
    print("Report generation complete!")
    print("="*70)

    return 0


if __name__ == "__main__":
    sys.exit(main())
