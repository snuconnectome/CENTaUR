"""
Test suite for cross-validation framework

TDD approach: Define expected cross-validation behavior
"""
import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import tempfile


class TestLOOCrossValidation:
    """Test Leave-One-Out cross-validation"""

    def test_loo_split_generation(self):
        """Test generating LOO splits"""
        from evaluation.cross_validation import generate_loo_splits

        n_samples = 10
        splits = generate_loo_splits(n_samples)

        assert len(splits) == n_samples
        # Each split should have n-1 train, 1 test
        for train_idx, test_idx in splits:
            assert len(train_idx) == n_samples - 1
            assert len(test_idx) == 1

    def test_loo_no_overlap(self):
        """Test LOO splits have no train/test overlap"""
        from evaluation.cross_validation import generate_loo_splits

        n_samples = 10
        splits = generate_loo_splits(n_samples)

        for train_idx, test_idx in splits:
            # No overlap between train and test
            assert len(set(train_idx) & set(test_idx)) == 0
            # Union covers all samples
            assert len(set(train_idx) | set(test_idx)) == n_samples

    def test_loo_coverage(self):
        """Test LOO covers all samples as test once"""
        from evaluation.cross_validation import generate_loo_splits

        n_samples = 10
        splits = generate_loo_splits(n_samples)

        # Collect all test indices
        test_indices = set()
        for _, test_idx in splits:
            test_indices.update(test_idx)

        # Should cover all samples exactly once
        assert test_indices == set(range(n_samples))


class TestNestedCrossValidation:
    """Test nested cross-validation for hyperparameter tuning"""

    def test_nested_cv_structure(self):
        """Test nested CV structure"""
        from evaluation.cross_validation import nested_cross_validation

        # Mock data
        features = torch.randn(50, 4096)
        labels = torch.randint(0, 2, (50,))
        alpha_grid = [0.001, 0.01, 0.1]

        # Mock model fitting
        def mock_fit(train_features, train_labels, alpha):
            return {"alpha": alpha, "weights": torch.randn(4096)}

        # Mock evaluation
        def mock_evaluate(model, test_features, test_labels):
            return {"accuracy": 0.85, "log_likelihood": -0.45}

        result = nested_cross_validation(
            features, labels,
            alpha_grid=alpha_grid,
            outer_folds=5,
            inner_folds=3,
            fit_func=mock_fit,
            eval_func=mock_evaluate
        )

        assert "outer_results" in result
        assert "best_alphas" in result
        assert len(result["outer_results"]) == 5

    def test_hyperparameter_selection(self):
        """Test hyperparameter selection in inner CV"""
        from evaluation.cross_validation import select_best_hyperparameter

        # Mock inner CV results
        inner_results = {
            0.001: {"mean_accuracy": 0.80, "std_accuracy": 0.05},
            0.01: {"mean_accuracy": 0.85, "std_accuracy": 0.03},
            0.1: {"mean_accuracy": 0.82, "std_accuracy": 0.04}
        }

        best_alpha = select_best_hyperparameter(inner_results, metric="mean_accuracy")

        assert best_alpha == 0.01  # Highest mean accuracy

    def test_nested_cv_no_data_leakage(self):
        """Test nested CV prevents data leakage"""
        from evaluation.cross_validation import check_data_leakage

        outer_train = [0, 1, 2, 3, 4]
        outer_test = [5]
        inner_train = [0, 1, 2, 3]
        inner_test = [4]

        # Should not leak (inner test in outer train)
        assert check_data_leakage(outer_train, outer_test, inner_train, inner_test) == False

        # Should leak (inner test in outer test)
        inner_test_leak = [5]
        assert check_data_leakage(outer_train, outer_test, inner_train, inner_test_leak) == True


class TestHyperparameterGrid:
    """Test hyperparameter grid search"""

    def test_grid_search_basic(self):
        """Test basic grid search"""
        from evaluation.cross_validation import grid_search

        features = torch.randn(20, 4096)
        labels = torch.randint(0, 2, (20,))
        alpha_grid = [0.001, 0.01, 0.1]

        def mock_fit_eval(train_features, train_labels, test_features, test_labels, alpha):
            return {"accuracy": 0.8 + alpha, "log_likelihood": -0.5}

        results = grid_search(
            features, labels,
            alpha_grid=alpha_grid,
            n_folds=3,
            fit_eval_func=mock_fit_eval
        )

        assert len(results) == len(alpha_grid)
        assert all(alpha in results for alpha in alpha_grid)

    def test_grid_search_best_params(self):
        """Test finding best parameters from grid search"""
        from evaluation.cross_validation import find_best_params

        grid_results = {
            0.001: {"mean_score": 0.80, "std_score": 0.05},
            0.01: {"mean_score": 0.85, "std_score": 0.03},
            0.1: {"mean_score": 0.82, "std_score": 0.04}
        }

        best_params = find_best_params(grid_results, metric="mean_score")

        assert best_params["alpha"] == 0.01
        assert best_params["score"] == 0.85


class TestCrossValidationMetrics:
    """Test cross-validation metric aggregation"""

    def test_aggregate_cv_results(self):
        """Test aggregating cross-validation results"""
        from evaluation.cross_validation import aggregate_cv_results

        fold_results = [
            {"accuracy": 0.85, "log_likelihood": -0.45},
            {"accuracy": 0.87, "log_likelihood": -0.42},
            {"accuracy": 0.83, "log_likelihood": -0.48}
        ]

        aggregated = aggregate_cv_results(fold_results)

        assert "mean_accuracy" in aggregated
        assert "std_accuracy" in aggregated
        assert "mean_log_likelihood" in aggregated
        assert abs(aggregated["mean_accuracy"] - 0.85) < 0.01

    def test_cv_confidence_intervals(self):
        """Test computing confidence intervals from CV results"""
        from evaluation.cross_validation import cv_confidence_intervals

        fold_scores = np.array([0.85, 0.87, 0.83, 0.86, 0.84])

        ci = cv_confidence_intervals(fold_scores, confidence_level=0.95)

        assert "mean" in ci
        assert "lower" in ci
        assert "upper" in ci
        assert ci["lower"] < ci["mean"] < ci["upper"]


class TestSLURMIntegration:
    """Test SLURM array job integration"""

    def test_generate_slurm_script(self):
        """Test generating SLURM array job script"""
        from evaluation.cross_validation import generate_slurm_script

        script = generate_slurm_script(
            job_name="cv_fold",
            n_folds=100,
            time_limit="02:00:00",
            memory="16GB",
            command="python fit_model.py --fold $SLURM_ARRAY_TASK_ID"
        )

        assert "#SBATCH --job-name=cv_fold" in script
        assert "#SBATCH --array=0-99" in script
        assert "#SBATCH --time=02:00:00" in script
        assert "#SBATCH --mem=16GB" in script
        assert "python fit_model.py --fold $SLURM_ARRAY_TASK_ID" in script

    def test_slurm_fold_assignment(self):
        """Test SLURM array task ID to fold assignment"""
        from evaluation.cross_validation import get_fold_from_task_id

        # Mock SLURM environment
        with patch.dict('os.environ', {'SLURM_ARRAY_TASK_ID': '5'}):
            fold_id = get_fold_from_task_id()
            assert fold_id == 5

    def test_collect_slurm_results(self):
        """Test collecting results from SLURM array job"""
        from evaluation.cross_validation import collect_slurm_results

        # Create temporary directory with mock results
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create mock result files
            for fold in range(3):
                result_file = output_dir / f"fold_{fold}.pth"
                torch.save({"accuracy": 0.85 + fold * 0.01}, result_file)

            results = collect_slurm_results(output_dir, n_folds=3)

            assert len(results) == 3
            assert all("accuracy" in r for r in results)


class TestCVResultsIO:
    """Test saving and loading cross-validation results"""

    def test_save_cv_results(self):
        """Test saving CV results to file"""
        from evaluation.cross_validation import save_cv_results

        results = {
            "fold_0": {"accuracy": 0.85, "log_likelihood": -0.45},
            "fold_1": {"accuracy": 0.87, "log_likelihood": -0.42},
            "aggregated": {"mean_accuracy": 0.86, "std_accuracy": 0.01}
        }

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            save_cv_results(results, Path(tmp.name))

            # Verify file exists and is loadable
            loaded = torch.load(tmp.name)
            assert "fold_0" in loaded
            assert loaded["aggregated"]["mean_accuracy"] == 0.86

    def test_load_cv_results(self):
        """Test loading CV results from file"""
        from evaluation.cross_validation import load_cv_results

        # Create temporary result file
        results = {
            "fold_0": {"accuracy": 0.85},
            "aggregated": {"mean_accuracy": 0.86}
        }

        with tempfile.NamedTemporaryFile(suffix=".pth", delete=False) as tmp:
            torch.save(results, tmp.name)

            loaded = load_cv_results(Path(tmp.name))
            assert loaded["fold_0"]["accuracy"] == 0.85
            assert loaded["aggregated"]["mean_accuracy"] == 0.86


class TestCVPipeline:
    """Test end-to-end cross-validation pipeline"""

    def test_full_cv_pipeline(self):
        """Test complete CV pipeline execution"""
        from evaluation.cross_validation import run_cv_pipeline

        # Mock data
        features = torch.randn(30, 4096)
        labels = torch.randint(0, 2, (30,))
        alpha_grid = [0.01, 0.1]

        # Mock model
        mock_model = Mock()
        mock_model.fit.return_value = {"weights": torch.randn(4096)}
        mock_model.evaluate.return_value = {"accuracy": 0.85}

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            result = run_cv_pipeline(
                features=features,
                labels=labels,
                model=mock_model,
                alpha_grid=alpha_grid,
                n_outer_folds=5,
                n_inner_folds=3,
                output_dir=output_dir
            )

            assert "cv_results" in result
            assert "best_hyperparameters" in result
            assert "aggregated_metrics" in result

    def test_cv_with_feature_normalization(self):
        """Test CV pipeline with feature normalization"""
        from evaluation.cross_validation import run_cv_with_normalization

        features = torch.randn(30, 4096)
        labels = torch.randint(0, 2, (30,))

        mock_model = Mock()
        mock_model.fit.return_value = {"weights": torch.randn(4096)}
        mock_model.evaluate.return_value = {"accuracy": 0.85}

        result = run_cv_with_normalization(
            features=features,
            labels=labels,
            model=mock_model,
            n_folds=5
        )

        # Check normalization was applied
        assert "normalization_stats" in result
        # Each fold should normalize using only training data
        assert "fold_results" in result


class TestCVUtilities:
    """Test cross-validation utility functions"""

    def test_stratified_cv_splits(self):
        """Test stratified CV split generation"""
        from evaluation.cross_validation import generate_stratified_cv_splits

        # Imbalanced dataset
        labels = np.array([0] * 80 + [1] * 20)
        n_folds = 5

        splits = generate_stratified_cv_splits(labels, n_folds)

        # Check each fold maintains class distribution
        for train_idx, test_idx in splits:
            test_labels = labels[test_idx]
            # Should have roughly 80/20 split in each fold
            assert 0.15 < test_labels.mean() < 0.25

    def test_group_cv_splits(self):
        """Test grouped CV (no subject leakage)"""
        from evaluation.cross_validation import generate_group_cv_splits

        # 10 samples from 5 subjects
        groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
        n_folds = 5

        splits = generate_group_cv_splits(groups, n_folds)

        # Check no subject appears in both train and test
        for train_idx, test_idx in splits:
            train_groups = set(groups[train_idx])
            test_groups = set(groups[test_idx])
            assert len(train_groups & test_groups) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
