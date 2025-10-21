"""
Integration test for full evaluation workflow

TDD approach: Define complete 100-fold LOO cross-validation workflow
"""
import pytest
import torch
import numpy as np
from pathlib import Path
import tempfile
from unittest.mock import Mock, MagicMock


class TestFullEvaluationWorkflow:
    """Integration test for complete 100-fold LOO evaluation"""

    def test_loo_cross_validation_setup(self):
        """Test setting up 100-fold LOO cross-validation"""
        from evaluation.cross_validation import generate_loo_splits

        n_samples = 100
        splits = generate_loo_splits(n_samples)

        assert len(splits) == 100

        # Verify all samples used as test exactly once
        test_indices = set()
        for train_idx, test_idx in splits:
            assert len(test_idx) == 1
            test_indices.add(test_idx[0])

        assert test_indices == set(range(100))

    def test_nested_cv_for_hyperparameter_tuning(self):
        """Test nested CV selects optimal hyperparameters"""
        from evaluation.cross_validation import nested_cross_validation

        # Mock dataset
        features = torch.randn(50, 4096)
        labels = torch.randint(0, 2, (50,))

        alpha_grid = [0.001, 0.01, 0.1]

        # Mock model functions
        def mock_fit(train_features, train_labels, alpha):
            return {"alpha": alpha, "weights": torch.randn(train_features.shape[1])}

        def mock_evaluate(model, test_features, test_labels):
            return {"accuracy": 0.80 + model["alpha"], "log_likelihood": -0.5}

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
        assert "aggregated" in result

        # Should have selected best alpha per fold
        assert len(result["best_alphas"]) == 5

    def test_cv_with_feature_normalization(self):
        """Test CV properly normalizes features per fold"""
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

        assert "fold_results" in result
        assert "aggregated" in result
        assert "normalization_stats" in result

        # Each fold should have normalization stats
        assert len(result["normalization_stats"]) == 5


class TestStatisticalComparison:
    """Test statistical comparison after CV"""

    def test_paired_ttest_across_folds(self):
        """Test paired t-test comparing models across folds"""
        from evaluation.compare_statistical import paired_ttest

        # Mock CV results for two models
        kocentaur_scores = np.array([0.85, 0.87, 0.86, 0.88, 0.84, 0.86, 0.85, 0.87, 0.86, 0.85])
        baseline_scores = np.array([0.80, 0.82, 0.81, 0.83, 0.79, 0.81, 0.80, 0.82, 0.81, 0.80])

        result = paired_ttest(kocentaur_scores, baseline_scores, alpha=0.05)

        assert "t_statistic" in result
        assert "p_value" in result
        assert "significant" in result

        # Ko-CENTaUR should be significantly better
        assert result["t_statistic"] > 0
        assert result["significant"] == True

    def test_effect_size_computation(self):
        """Test Cohen's d effect size between models"""
        from evaluation.compare_statistical import cohens_d_with_interpretation

        kocentaur_scores = np.array([0.85, 0.87, 0.86, 0.88, 0.84])
        baseline_scores = np.array([0.75, 0.77, 0.76, 0.78, 0.74])

        result = cohens_d_with_interpretation(kocentaur_scores, baseline_scores)

        assert "cohens_d" in result
        assert "interpretation" in result

        # Large difference should give large effect size
        assert result["cohens_d"] > 0.8
        assert result["interpretation"] == "large"

    def test_bonferroni_correction_multiple_comparisons(self):
        """Test Bonferroni correction for comparing multiple baselines"""
        from evaluation.compare_statistical import bonferroni_correction

        # Ko-CENTaUR vs 4 baselines = 4 comparisons
        p_values = [0.01, 0.02, 0.03, 0.04]
        alpha = 0.05

        corrected = bonferroni_correction(p_values, alpha)

        assert "corrected_alpha" in corrected
        assert "significant" in corrected

        # Corrected alpha = 0.05 / 4 = 0.0125
        assert corrected["corrected_alpha"] == 0.0125

        # Only first comparison remains significant
        assert corrected["significant"][0] == True
        assert corrected["significant"][3] == False


class TestComprehensiveReporting:
    """Test generating comprehensive evaluation reports"""

    def test_generate_full_statistical_report(self):
        """Test generating complete statistical report"""
        from evaluation.compare_statistical import generate_statistical_report

        # Mock results for Ko-CENTaUR and 2 baselines
        results = {
            "ko-centaur": {
                "accuracy": np.array([0.85, 0.87, 0.86, 0.88, 0.84]),
                "log_likelihood": np.array([-0.45, -0.42, -0.44, -0.41, -0.46])
            },
            "exaone-base": {
                "accuracy": np.array([0.80, 0.82, 0.81, 0.83, 0.79]),
                "log_likelihood": np.array([-0.55, -0.52, -0.54, -0.51, -0.56])
            },
            "llama-3.2-3b": {
                "accuracy": np.array([0.78, 0.80, 0.79, 0.81, 0.77]),
                "log_likelihood": np.array([-0.60, -0.57, -0.59, -0.56, -0.61])
            }
        }

        report = generate_statistical_report(results, alpha=0.05)

        assert "summary" in report
        assert "pairwise_comparisons" in report
        assert "best_model" in report
        assert "rankings" in report

        # Ko-CENTaUR should be best model
        assert report["best_model"] == "ko-centaur"

        # Should have rankings for each metric
        assert "accuracy" in report["rankings"]
        assert "log_likelihood" in report["rankings"]

    def test_save_evaluation_report(self):
        """Test saving evaluation report to file"""
        from evaluation.compare_statistical import save_statistical_report

        report = {
            "summary": {"ko-centaur": {"mean_accuracy": 0.86}},
            "best_model": "ko-centaur",
            "pairwise_comparisons": {}
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "evaluation_report.json"

            save_statistical_report(report, output_path)

            # Verify file created
            assert output_path.exists()

            # Verify can reload
            import json
            with open(output_path) as f:
                loaded = json.load(f)

            assert loaded["best_model"] == "ko-centaur"


class TestSLURMIntegration:
    """Test SLURM integration for parallel execution"""

    def test_generate_slurm_array_script(self):
        """Test generating SLURM script for 100-fold LOO"""
        from evaluation.cross_validation import generate_slurm_script

        script = generate_slurm_script(
            job_name="kocentaur_loo_cv",
            n_folds=100,
            time_limit="04:00:00",
            memory="32GB",
            command="python run_fold.py --fold $SLURM_ARRAY_TASK_ID"
        )

        assert "#SBATCH --job-name=kocentaur_loo_cv" in script
        assert "#SBATCH --array=0-99" in script
        assert "#SBATCH --time=04:00:00" in script
        assert "#SBATCH --mem=32GB" in script

    def test_collect_slurm_results(self):
        """Test collecting results from SLURM array job"""
        from evaluation.cross_validation import collect_slurm_results

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            # Create mock result files for 10 folds
            for fold in range(10):
                result = {
                    "fold_id": fold,
                    "accuracy": 0.85 + np.random.randn() * 0.02,
                    "log_likelihood": -0.45 + np.random.randn() * 0.05
                }
                torch.save(result, output_dir / f"fold_{fold}.pth")

            # Collect results
            results = collect_slurm_results(output_dir, n_folds=10)

            assert len(results) == 10
            assert all("accuracy" in r for r in results)
            assert all("log_likelihood" in r for r in results)


class TestEndToEndEvaluation:
    """Test complete end-to-end evaluation"""

    def test_complete_evaluation_pipeline(self):
        """Test running complete evaluation from start to finish"""
        # This is the master integration test that ties everything together

        # 1. Mock data loading
        samples = [
            {"task_description": f"심리학 질문 {i}", "label": i % 2}
            for i in range(100)
        ]

        # 2. Mock model loading
        mock_kocentaur = Mock()
        mock_kocentaur.model_type = "ko-centaur"
        mock_kocentaur.extract_features.return_value = torch.randn(1, 4096)

        mock_baseline = Mock()
        mock_baseline.model_type = "exaone-base"
        mock_baseline.extract_features.return_value = torch.randn(1, 4096)

        # 3. Feature extraction (mocked)
        kocentaur_features = torch.randn(100, 4096)
        baseline_features = torch.randn(100, 4096)
        labels = torch.tensor([s["label"] for s in samples])

        # 4. Cross-validation (simplified for test)
        from evaluation.cross_validation import generate_loo_splits

        splits = generate_loo_splits(100)
        assert len(splits) == 100

        # 5. Statistical comparison
        from evaluation.compare_statistical import paired_ttest

        # Mock fold results
        kocentaur_scores = 0.85 + np.random.randn(10) * 0.02
        baseline_scores = 0.80 + np.random.randn(10) * 0.02

        result = paired_ttest(kocentaur_scores, baseline_scores)
        assert "t_statistic" in result

        # Pipeline complete!
        assert True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
