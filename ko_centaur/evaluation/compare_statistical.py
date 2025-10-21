"""
Statistical comparison system for Ko-CENTaUR

TDD Implementation: Passes tests in test_statistical_comparison.py

Provides comprehensive statistical testing for model comparison with proper corrections.
"""
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy import stats
from itertools import combinations


def paired_ttest(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    alpha: float = 0.05
) -> Dict:
    """
    Perform paired t-test between two models

    Args:
        scores_a: Performance scores from model A
        scores_b: Performance scores from model B
        alpha: Significance level (default 0.05)

    Returns:
        Dict with t_statistic, p_value, significant
    """
    # Handle edge case: identical arrays
    if np.allclose(scores_a, scores_b):
        result = {
            "t_statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "alpha": alpha
        }
        return result

    # Perform paired t-test
    t_statistic, p_value = stats.ttest_rel(scores_a, scores_b)

    result = {
        "t_statistic": float(t_statistic),
        "p_value": float(p_value),
        "significant": p_value < alpha,
        "alpha": alpha
    }

    return result


def cohens_d(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """
    Compute Cohen's d effect size

    Args:
        scores_a: Scores from model A
        scores_b: Scores from model B

    Returns:
        Cohen's d value
    """
    # Calculate means
    mean_a = np.mean(scores_a)
    mean_b = np.mean(scores_b)

    # Calculate pooled standard deviation
    n_a = len(scores_a)
    n_b = len(scores_b)
    var_a = np.var(scores_a, ddof=1)
    var_b = np.var(scores_b, ddof=1)

    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    # Cohen's d
    d = (mean_a - mean_b) / pooled_std

    return float(d)


def interpret_cohens_d(d: float) -> str:
    """
    Interpret Cohen's d effect size

    Args:
        d: Cohen's d value

    Returns:
        Interpretation string
    """
    abs_d = abs(d)

    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def cohens_d_with_interpretation(
    scores_a: np.ndarray,
    scores_b: np.ndarray
) -> Dict:
    """
    Compute Cohen's d with interpretation

    Args:
        scores_a: Scores from model A
        scores_b: Scores from model B

    Returns:
        Dict with cohens_d and interpretation
    """
    d = cohens_d(scores_a, scores_b)
    interpretation = interpret_cohens_d(d)

    return {
        "cohens_d": d,
        "interpretation": interpretation
    }


def bonferroni_correction(
    p_values: List[float],
    alpha: float = 0.05
) -> Dict:
    """
    Apply Bonferroni correction for multiple comparisons

    Args:
        p_values: List of p-values from multiple tests
        alpha: Family-wise error rate

    Returns:
        Dict with corrected_alpha and significant list
    """
    n_tests = len(p_values)
    corrected_alpha = alpha / n_tests

    significant = [p < corrected_alpha for p in p_values]

    return {
        "corrected_alpha": corrected_alpha,
        "significant": significant,
        "n_tests": n_tests,
        "original_alpha": alpha
    }


def confidence_interval(
    scores: np.ndarray,
    confidence_level: float = 0.95
) -> Dict:
    """
    Calculate confidence interval for scores

    Args:
        scores: Performance scores
        confidence_level: Confidence level (default 0.95)

    Returns:
        Dict with mean, lower, upper, margin_of_error
    """
    mean = np.mean(scores)
    sem = stats.sem(scores)
    n = len(scores)

    # t-distribution critical value
    alpha = 1 - confidence_level
    df = n - 1
    t_critical = stats.t.ppf(1 - alpha / 2, df)

    margin_of_error = t_critical * sem
    lower = mean - margin_of_error
    upper = mean + margin_of_error

    return {
        "mean": float(mean),
        "lower": float(lower),
        "upper": float(upper),
        "margin_of_error": float(margin_of_error),
        "confidence_level": confidence_level
    }


def paired_difference_ci(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    confidence_level: float = 0.95
) -> Dict:
    """
    Calculate confidence interval for paired differences

    Args:
        scores_a: Scores from model A
        scores_b: Scores from model B
        confidence_level: Confidence level

    Returns:
        Dict with mean_difference, lower, upper, includes_zero
    """
    differences = scores_a - scores_b
    ci = confidence_interval(differences, confidence_level)

    # Check if CI includes zero (no significant difference)
    includes_zero = ci["lower"] <= 0 <= ci["upper"]

    return {
        "mean_difference": ci["mean"],
        "lower": ci["lower"],
        "upper": ci["upper"],
        "margin_of_error": ci["margin_of_error"],
        "confidence_level": confidence_level,
        "includes_zero": includes_zero
    }


def generate_comparison_table(
    results: Dict[str, Dict[str, np.ndarray]],
    include_stats: bool = False
) -> Dict:
    """
    Generate comparison table from results

    Args:
        results: Dict mapping model names to metric arrays
        include_stats: Whether to include statistical measures

    Returns:
        Comparison table dict
    """
    table = {
        "models": {},
        "metrics": list(next(iter(results.values())).keys())
    }

    for model_name, metrics in results.items():
        model_data = {}

        for metric_name, scores in metrics.items():
            if include_stats:
                # Calculate statistics
                mean = float(np.mean(scores))
                std = float(np.std(scores, ddof=1))
                ci = confidence_interval(scores)

                model_data[f"mean_{metric_name}"] = mean
                model_data[f"std_{metric_name}"] = std
                model_data[f"ci_{metric_name}"] = (ci["lower"], ci["upper"])
            else:
                model_data[metric_name] = scores.tolist() if isinstance(scores, np.ndarray) else scores

        table["models"][model_name] = model_data

    return table


def pairwise_comparison(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    model_a_name: str,
    model_b_name: str,
    alpha: float = 0.05
) -> Dict:
    """
    Comprehensive pairwise comparison between two models

    Args:
        scores_a: Scores from model A
        scores_b: Scores from model B
        model_a_name: Name of model A
        model_b_name: Name of model B
        alpha: Significance level

    Returns:
        Dict with all comparison statistics
    """
    # T-test
    t_test = paired_ttest(scores_a, scores_b, alpha)

    # Effect size
    effect_size = cohens_d_with_interpretation(scores_a, scores_b)

    # Confidence interval for difference
    ci = paired_difference_ci(scores_a, scores_b)

    result = {
        "model_a": model_a_name,
        "model_b": model_b_name,
        "t_test": t_test,
        "effect_size": effect_size,
        "confidence_interval": ci,
        "mean_a": float(np.mean(scores_a)),
        "mean_b": float(np.mean(scores_b)),
        "difference": float(np.mean(scores_a) - np.mean(scores_b))
    }

    return result


def all_pairwise_comparisons(
    results: Dict[str, np.ndarray],
    alpha: float = 0.05
) -> List[Dict]:
    """
    Perform all pairwise comparisons across multiple models

    Args:
        results: Dict mapping model names to score arrays
        alpha: Significance level

    Returns:
        List of pairwise comparison dicts
    """
    model_names = list(results.keys())
    comparisons = []

    # Generate all pairs
    for model_a, model_b in combinations(model_names, 2):
        scores_a = results[model_a]
        scores_b = results[model_b]

        comparison = pairwise_comparison(
            scores_a, scores_b,
            model_a, model_b,
            alpha
        )

        comparisons.append(comparison)

    return comparisons


def rank_models(
    results: Dict[str, Dict[str, float]],
    metric: str = "accuracy"
) -> List[Dict]:
    """
    Rank models by single metric

    Args:
        results: Dict mapping model names to metrics
        metric: Metric to rank by

    Returns:
        List of dicts with model and score, sorted descending
    """
    rankings = []

    for model_name, metrics in results.items():
        score = metrics[metric]
        rankings.append({
            "model": model_name,
            "score": score
        })

    # Sort descending by score
    rankings.sort(key=lambda x: x["score"], reverse=True)

    # Add rank numbers
    for i, ranking in enumerate(rankings):
        ranking["rank"] = i + 1

    return rankings


def rank_models_with_significance(
    results: Dict[str, np.ndarray],
    alpha: float = 0.05
) -> List[Dict]:
    """
    Rank models with significance testing

    Args:
        results: Dict mapping model names to score arrays
        alpha: Significance level

    Returns:
        List of rankings with significance information
    """
    # Calculate means for ranking
    means = {name: float(np.mean(scores)) for name, scores in results.items()}

    # Rank by mean
    rankings = rank_models(
        {name: {"accuracy": score} for name, score in means.items()},
        metric="accuracy"
    )

    # Perform pairwise comparisons
    comparisons = all_pairwise_comparisons(results, alpha)

    # Add significance information
    for ranking in rankings:
        model = ranking["model"]
        significantly_better_than = []

        for comparison in comparisons:
            # Check if this model is significantly better
            if comparison["model_a"] == model:
                if comparison["t_test"]["significant"] and comparison["difference"] > 0:
                    significantly_better_than.append(comparison["model_b"])
            elif comparison["model_b"] == model:
                if comparison["t_test"]["significant"] and comparison["difference"] < 0:
                    significantly_better_than.append(comparison["model_a"])

        ranking["significantly_better_than"] = significantly_better_than

    return rankings


def generate_statistical_report(
    results: Dict[str, Dict[str, np.ndarray]],
    alpha: float = 0.05
) -> Dict:
    """
    Generate comprehensive statistical report

    Args:
        results: Dict mapping model names to metric arrays
        alpha: Significance level

    Returns:
        Statistical report dict
    """
    report = {
        "summary": {},
        "pairwise_comparisons": {},
        "best_model": None,
        "rankings": {}
    }

    # Generate summary statistics for each model
    for model_name, metrics in results.items():
        model_summary = {}

        for metric_name, scores in metrics.items():
            model_summary[f"mean_{metric_name}"] = float(np.mean(scores))
            model_summary[f"std_{metric_name}"] = float(np.std(scores, ddof=1))

            ci = confidence_interval(scores)
            model_summary[f"ci_{metric_name}"] = (ci["lower"], ci["upper"])

        report["summary"][model_name] = model_summary

    # Pairwise comparisons for each metric
    for metric_name in next(iter(results.values())).keys():
        metric_results = {name: metrics[metric_name] for name, metrics in results.items()}
        comparisons = all_pairwise_comparisons(metric_results, alpha)
        report["pairwise_comparisons"][metric_name] = comparisons

    # Rankings for each metric
    for metric_name in next(iter(results.values())).keys():
        metric_results = {name: metrics[metric_name] for name, metrics in results.items()}
        rankings = rank_models_with_significance(metric_results, alpha)
        report["rankings"][metric_name] = rankings

    # Determine best model (by first metric)
    first_metric = list(next(iter(results.values())).keys())[0]
    best_model = report["rankings"][first_metric][0]["model"]
    report["best_model"] = best_model

    return report


def save_statistical_report(report: Dict, output_path: Path):
    """
    Save statistical report to JSON

    Args:
        report: Statistical report dict
        output_path: Path to save file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to native Python types
    def convert_numpy(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(item) for item in obj]
        else:
            return obj

    serializable_report = convert_numpy(report)

    with open(output_path, 'w') as f:
        json.dump(serializable_report, f, indent=2)


if __name__ == "__main__":
    print("Statistical comparison module for Ko-CENTaUR")
    print("Run tests with: pytest tests/test_statistical_comparison.py -v")
