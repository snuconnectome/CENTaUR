"""
Cross-validation framework for Ko-CENTaUR

TDD Implementation: Passes tests in test_cross_validation.py

Provides comprehensive cross-validation with hyperparameter tuning and SLURM integration.
"""
import numpy as np
import torch
import os
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional, Any
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold


def generate_loo_splits(n_samples: int) -> List[Tuple[List[int], List[int]]]:
    """
    Generate Leave-One-Out cross-validation splits

    Args:
        n_samples: Number of samples

    Returns:
        List of (train_indices, test_indices) tuples
    """
    splits = []

    for i in range(n_samples):
        # Test index is current sample
        test_idx = [i]
        # Train indices are all others
        train_idx = [j for j in range(n_samples) if j != i]

        splits.append((train_idx, test_idx))

    return splits


def check_data_leakage(
    outer_train: List[int],
    outer_test: List[int],
    inner_train: List[int],
    inner_test: List[int]
) -> bool:
    """
    Check if there is data leakage in nested CV

    Args:
        outer_train: Outer loop training indices
        outer_test: Outer loop test indices
        inner_train: Inner loop training indices
        inner_test: Inner loop test indices

    Returns:
        True if leakage detected, False otherwise
    """
    # Convert to sets
    outer_test_set = set(outer_test)
    inner_test_set = set(inner_test)

    # Check if any inner test samples are in outer test
    leakage = len(outer_test_set & inner_test_set) > 0

    return leakage


def select_best_hyperparameter(
    inner_results: Dict[float, Dict[str, float]],
    metric: str = "mean_accuracy"
) -> float:
    """
    Select best hyperparameter from inner CV results

    Args:
        inner_results: Dict mapping hyperparameters to metrics
        metric: Metric to optimize

    Returns:
        Best hyperparameter value
    """
    best_alpha = None
    best_score = -float('inf')

    for alpha, metrics in inner_results.items():
        score = metrics[metric]
        if score > best_score:
            best_score = score
            best_alpha = alpha

    return best_alpha


def grid_search(
    features: torch.Tensor,
    labels: torch.Tensor,
    alpha_grid: List[float],
    n_folds: int,
    fit_eval_func: Callable
) -> Dict[float, Dict]:
    """
    Perform grid search over hyperparameters

    Args:
        features: Feature tensor
        labels: Label tensor
        alpha_grid: List of alpha values to try
        n_folds: Number of cross-validation folds
        fit_eval_func: Function to fit and evaluate model

    Returns:
        Dict mapping alpha to results
    """
    # Convert to numpy for sklearn
    n_samples = len(features)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    results = {alpha: [] for alpha in alpha_grid}

    for train_idx, test_idx in kf.split(range(n_samples)):
        train_features = features[train_idx]
        train_labels = labels[train_idx]
        test_features = features[test_idx]
        test_labels = labels[test_idx]

        for alpha in alpha_grid:
            fold_result = fit_eval_func(
                train_features, train_labels,
                test_features, test_labels,
                alpha
            )
            results[alpha].append(fold_result)

    # Aggregate results for each alpha
    aggregated = {}
    for alpha, fold_results in results.items():
        aggregated[alpha] = aggregate_cv_results(fold_results)

    return aggregated


def find_best_params(
    grid_results: Dict[float, Dict],
    metric: str = "mean_score"
) -> Dict:
    """
    Find best parameters from grid search results

    Args:
        grid_results: Results from grid search
        metric: Metric to optimize

    Returns:
        Dict with best alpha and score
    """
    best_alpha = None
    best_score = -float('inf')

    for alpha, metrics in grid_results.items():
        score = metrics[metric]
        if score > best_score:
            best_score = score
            best_alpha = alpha

    return {
        "alpha": best_alpha,
        "score": best_score
    }


def nested_cross_validation(
    features: torch.Tensor,
    labels: torch.Tensor,
    alpha_grid: List[float],
    outer_folds: int,
    inner_folds: int,
    fit_func: Callable,
    eval_func: Callable
) -> Dict:
    """
    Perform nested cross-validation

    Args:
        features: Feature tensor
        labels: Label tensor
        alpha_grid: Hyperparameter grid
        outer_folds: Number of outer CV folds
        inner_folds: Number of inner CV folds
        fit_func: Model fitting function
        eval_func: Model evaluation function

    Returns:
        Nested CV results
    """
    n_samples = len(features)
    outer_kf = KFold(n_splits=outer_folds, shuffle=True, random_state=42)

    outer_results = []
    best_alphas = []

    for outer_train_idx, outer_test_idx in outer_kf.split(range(n_samples)):
        # Outer split
        outer_train_features = features[outer_train_idx]
        outer_train_labels = labels[outer_train_idx]
        outer_test_features = features[outer_test_idx]
        outer_test_labels = labels[outer_test_idx]

        # Inner CV for hyperparameter selection
        inner_results = {}
        inner_kf = KFold(n_splits=inner_folds, shuffle=True, random_state=42)

        for alpha in alpha_grid:
            alpha_results = []

            for inner_train_idx, inner_test_idx in inner_kf.split(range(len(outer_train_features))):
                inner_train_features = outer_train_features[inner_train_idx]
                inner_train_labels = outer_train_labels[inner_train_idx]
                inner_test_features = outer_train_features[inner_test_idx]
                inner_test_labels = outer_train_labels[inner_test_idx]

                # Fit and evaluate
                model = fit_func(inner_train_features, inner_train_labels, alpha)
                result = eval_func(model, inner_test_features, inner_test_labels)
                alpha_results.append(result)

            # Aggregate inner results
            inner_results[alpha] = aggregate_cv_results(alpha_results)

        # Select best hyperparameter
        best_alpha = select_best_hyperparameter(inner_results)
        best_alphas.append(best_alpha)

        # Train on full outer training set with best alpha
        final_model = fit_func(outer_train_features, outer_train_labels, best_alpha)

        # Evaluate on outer test set
        outer_result = eval_func(final_model, outer_test_features, outer_test_labels)
        outer_result["best_alpha"] = best_alpha
        outer_results.append(outer_result)

    return {
        "outer_results": outer_results,
        "best_alphas": best_alphas,
        "aggregated": aggregate_cv_results(outer_results)
    }


def aggregate_cv_results(fold_results: List[Dict]) -> Dict:
    """
    Aggregate cross-validation results across folds

    Args:
        fold_results: List of result dicts from each fold

    Returns:
        Aggregated metrics with mean and std
    """
    # Extract all metrics
    metrics = {}
    for result in fold_results:
        for key, value in result.items():
            if isinstance(value, (int, float)):
                if key not in metrics:
                    metrics[key] = []
                metrics[key].append(value)

    # Compute mean and std
    aggregated = {}
    for key, values in metrics.items():
        aggregated[f"mean_{key}"] = float(np.mean(values))
        aggregated[f"std_{key}"] = float(np.std(values, ddof=1))

    return aggregated


def cv_confidence_intervals(
    fold_scores: np.ndarray,
    confidence_level: float = 0.95
) -> Dict:
    """
    Compute confidence intervals from CV fold scores

    Args:
        fold_scores: Array of scores from each fold
        confidence_level: Confidence level

    Returns:
        Dict with mean, lower, upper bounds
    """
    from scipy import stats

    mean = np.mean(fold_scores)
    sem = stats.sem(fold_scores)
    n = len(fold_scores)

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
        "confidence_level": confidence_level
    }


def generate_slurm_script(
    job_name: str,
    n_folds: int,
    time_limit: str,
    memory: str,
    command: str,
    **kwargs
) -> str:
    """
    Generate SLURM array job script

    Args:
        job_name: Name for SLURM job
        n_folds: Number of folds (array size)
        time_limit: Time limit per job (HH:MM:SS)
        memory: Memory per job
        command: Command to execute

    Returns:
        SLURM script as string
    """
    script = f"""#!/bin/bash
#SBATCH --job-name={job_name}
#SBATCH --array=0-{n_folds - 1}
#SBATCH --time={time_limit}
#SBATCH --mem={memory}
#SBATCH --output=logs/{job_name}_%A_%a.out
#SBATCH --error=logs/{job_name}_%A_%a.err

# Set up environment
module load python/3.10
source venv/bin/activate

# Run command
{command}
"""

    return script


def get_fold_from_task_id() -> int:
    """
    Get fold ID from SLURM array task ID

    Returns:
        Fold ID (0-indexed)
    """
    task_id = os.environ.get('SLURM_ARRAY_TASK_ID', '0')
    return int(task_id)


def collect_slurm_results(
    output_dir: Path,
    n_folds: int
) -> List[Dict]:
    """
    Collect results from SLURM array job

    Args:
        output_dir: Directory containing result files
        n_folds: Number of folds

    Returns:
        List of results from each fold
    """
    results = []

    for fold in range(n_folds):
        result_file = output_dir / f"fold_{fold}.pth"

        if result_file.exists():
            result = torch.load(result_file)
            results.append(result)
        else:
            print(f"Warning: Missing result for fold {fold}")

    return results


def save_cv_results(results: Dict, output_path: Path):
    """
    Save cross-validation results

    Args:
        results: CV results dict
        output_path: Path to save file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, output_path)


def load_cv_results(result_path: Path) -> Dict:
    """
    Load cross-validation results

    Args:
        result_path: Path to result file

    Returns:
        CV results dict
    """
    return torch.load(result_path)


def run_cv_pipeline(
    features: torch.Tensor,
    labels: torch.Tensor,
    model: Any,
    alpha_grid: List[float],
    n_outer_folds: int,
    n_inner_folds: int,
    output_dir: Path
) -> Dict:
    """
    Run complete cross-validation pipeline

    Args:
        features: Feature tensor
        labels: Label tensor
        model: Model object with fit/evaluate methods
        alpha_grid: Hyperparameter grid
        n_outer_folds: Outer CV folds
        n_inner_folds: Inner CV folds
        output_dir: Output directory

    Returns:
        Pipeline results
    """
    # Define fit and eval functions
    def fit_func(train_features, train_labels, alpha):
        return model.fit(train_features, train_labels, alpha=alpha)

    def eval_func(fitted_model, test_features, test_labels):
        return model.evaluate(test_features, test_labels)

    # Run nested CV
    cv_results = nested_cross_validation(
        features, labels,
        alpha_grid=alpha_grid,
        outer_folds=n_outer_folds,
        inner_folds=n_inner_folds,
        fit_func=fit_func,
        eval_func=eval_func
    )

    # Find best hyperparameters
    best_alphas = cv_results["best_alphas"]
    best_alpha = max(set(best_alphas), key=best_alphas.count)  # Most common

    # Save results
    output_dir.mkdir(parents=True, exist_ok=True)
    save_cv_results(cv_results, output_dir / "cv_results.pth")

    return {
        "cv_results": cv_results,
        "best_hyperparameters": {"alpha": best_alpha},
        "aggregated_metrics": cv_results["aggregated"]
    }


def run_cv_with_normalization(
    features: torch.Tensor,
    labels: torch.Tensor,
    model: Any,
    n_folds: int
) -> Dict:
    """
    Run cross-validation with proper feature normalization

    Args:
        features: Feature tensor
        labels: Label tensor
        model: Model object
        n_folds: Number of folds

    Returns:
        CV results with normalization
    """
    n_samples = len(features)
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []
    normalization_stats = []

    for train_idx, test_idx in kf.split(range(n_samples)):
        train_features = features[train_idx]
        train_labels = labels[train_idx]
        test_features = features[test_idx]
        test_labels = labels[test_idx]

        # Normalize using training statistics
        train_mean = train_features.mean(dim=0)
        train_std = train_features.std(dim=0)

        normalized_train = (train_features - train_mean) / (train_std + 1e-8)
        normalized_test = (test_features - train_mean) / (train_std + 1e-8)

        # Store normalization stats
        normalization_stats.append({
            "mean": train_mean,
            "std": train_std
        })

        # Fit and evaluate
        fitted_model = model.fit(normalized_train, train_labels)
        result = model.evaluate(normalized_test, test_labels)
        fold_results.append(result)

    return {
        "fold_results": fold_results,
        "aggregated": aggregate_cv_results(fold_results),
        "normalization_stats": normalization_stats
    }


def generate_stratified_cv_splits(
    labels: np.ndarray,
    n_folds: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate stratified CV splits preserving class distribution

    Args:
        labels: Label array
        n_folds: Number of folds

    Returns:
        List of (train_indices, test_indices) tuples
    """
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    splits = []
    for train_idx, test_idx in skf.split(range(len(labels)), labels):
        splits.append((train_idx, test_idx))

    return splits


def generate_group_cv_splits(
    groups: np.ndarray,
    n_folds: int
) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Generate grouped CV splits (no subject leakage)

    Args:
        groups: Group/subject IDs for each sample
        n_folds: Number of folds

    Returns:
        List of (train_indices, test_indices) tuples
    """
    gkf = GroupKFold(n_splits=n_folds)

    splits = []
    for train_idx, test_idx in gkf.split(range(len(groups)), groups=groups):
        splits.append((train_idx, test_idx))

    return splits


if __name__ == "__main__":
    print("Cross-validation framework for Ko-CENTaUR")
    print("Run tests with: pytest tests/test_cross_validation.py -v")
