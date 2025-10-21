"""
Test suite for statistical comparison system

TDD approach: Define expected statistical testing behavior
"""
import pytest
import numpy as np
import torch
from pathlib import Path
from unittest.mock import Mock, patch
from scipy import stats


class TestPairedTTest:
    """Test paired t-test functionality"""

    def test_paired_ttest_basic(self):
        """Test basic paired t-test computation"""
        from evaluation.compare_statistical import paired_ttest

        # Model A performs better on average
        scores_a = np.array([0.85, 0.87, 0.86, 0.88, 0.84])
        scores_b = np.array([0.80, 0.82, 0.81, 0.83, 0.79])

        result = paired_ttest(scores_a, scores_b)

        assert "t_statistic" in result
        assert "p_value" in result
        assert "significant" in result
        assert result["t_statistic"] > 0  # A > B
        assert isinstance(result["p_value"], float)

    def test_paired_ttest_equal_means(self):
        """Test t-test with equal means"""
        from evaluation.compare_statistical import paired_ttest

        scores_a = np.array([0.80, 0.82, 0.81, 0.83, 0.79])
        scores_b = np.array([0.80, 0.82, 0.81, 0.83, 0.79])

        result = paired_ttest(scores_a, scores_b)

        assert abs(result["t_statistic"]) < 0.01  # Near zero
        assert result["p_value"] > 0.05  # Not significant
        assert result["significant"] == False

    def test_paired_ttest_significance_levels(self):
        """Test different significance level thresholds"""
        from evaluation.compare_statistical import paired_ttest

        scores_a = np.array([0.85, 0.87, 0.86, 0.88, 0.84])
        scores_b = np.array([0.80, 0.82, 0.81, 0.83, 0.79])

        # Test with alpha=0.05
        result_05 = paired_ttest(scores_a, scores_b, alpha=0.05)
        # Test with alpha=0.01
        result_01 = paired_ttest(scores_a, scores_b, alpha=0.01)

        assert "significant" in result_05
        assert "significant" in result_01
        # More stringent threshold harder to reject
        if result_01["significant"]:
            assert result_05["significant"]


class TestEffectSize:
    """Test effect size calculations"""

    def test_cohens_d_basic(self):
        """Test Cohen's d computation"""
        from evaluation.compare_statistical import cohens_d

        # Large effect size
        scores_a = np.array([0.9, 0.92, 0.91, 0.93, 0.89])
        scores_b = np.array([0.7, 0.72, 0.71, 0.73, 0.69])

        d = cohens_d(scores_a, scores_b)

        assert isinstance(d, float)
        assert d > 0  # A > B
        # Large effect size (> 0.8)
        assert abs(d) > 0.8

    def test_cohens_d_interpretation(self):
        """Test Cohen's d with interpretation"""
        from evaluation.compare_statistical import cohens_d_with_interpretation

        scores_a = np.array([0.85, 0.87, 0.86, 0.88, 0.84])
        scores_b = np.array([0.80, 0.82, 0.81, 0.83, 0.79])

        result = cohens_d_with_interpretation(scores_a, scores_b)

        assert "cohens_d" in result
        assert "interpretation" in result
        assert result["interpretation"] in ["negligible", "small", "medium", "large"]

    def test_effect_size_thresholds(self):
        """Test effect size threshold classifications"""
        from evaluation.compare_statistical import interpret_cohens_d

        assert interpret_cohens_d(0.1) == "negligible"
        assert interpret_cohens_d(0.3) == "small"
        assert interpret_cohens_d(0.6) == "medium"
        assert interpret_cohens_d(0.9) == "large"


class TestMultipleComparisons:
    """Test multiple comparison corrections"""

    def test_bonferroni_correction(self):
        """Test Bonferroni correction for multiple tests"""
        from evaluation.compare_statistical import bonferroni_correction

        p_values = [0.01, 0.02, 0.03, 0.04, 0.05]
        alpha = 0.05

        corrected = bonferroni_correction(p_values, alpha)

        assert "corrected_alpha" in corrected
        assert "significant" in corrected
        assert len(corrected["significant"]) == len(p_values)
        # Corrected alpha should be stricter
        assert corrected["corrected_alpha"] < alpha
        assert corrected["corrected_alpha"] == alpha / len(p_values)

    def test_bonferroni_all_significant(self):
        """Test when all tests remain significant after correction"""
        from evaluation.compare_statistical import bonferroni_correction

        # Very small p-values
        p_values = [0.001, 0.002, 0.003]
        alpha = 0.05

        corrected = bonferroni_correction(p_values, alpha)

        # All should remain significant
        assert all(corrected["significant"])

    def test_bonferroni_none_significant(self):
        """Test when no tests remain significant after correction"""
        from evaluation.compare_statistical import bonferroni_correction

        # Borderline p-values
        p_values = [0.04, 0.045, 0.049]
        alpha = 0.05

        corrected = bonferroni_correction(p_values, alpha)

        # None should remain significant (0.05/3 = 0.0167)
        assert not any(corrected["significant"])


class TestConfidenceIntervals:
    """Test confidence interval computation"""

    def test_confidence_interval_basic(self):
        """Test basic confidence interval calculation"""
        from evaluation.compare_statistical import confidence_interval

        scores = np.array([0.80, 0.82, 0.84, 0.86, 0.88])
        confidence_level = 0.95

        ci = confidence_interval(scores, confidence_level)

        assert "mean" in ci
        assert "lower" in ci
        assert "upper" in ci
        assert "margin_of_error" in ci
        assert ci["lower"] < ci["mean"] < ci["upper"]
        assert ci["margin_of_error"] > 0

    def test_confidence_interval_different_levels(self):
        """Test confidence intervals at different levels"""
        from evaluation.compare_statistical import confidence_interval

        scores = np.array([0.80, 0.82, 0.84, 0.86, 0.88])

        ci_95 = confidence_interval(scores, 0.95)
        ci_99 = confidence_interval(scores, 0.99)

        # 99% CI should be wider than 95% CI
        width_95 = ci_95["upper"] - ci_95["lower"]
        width_99 = ci_99["upper"] - ci_99["lower"]
        assert width_99 > width_95

    def test_paired_difference_ci(self):
        """Test confidence interval for paired differences"""
        from evaluation.compare_statistical import paired_difference_ci

        scores_a = np.array([0.85, 0.87, 0.86, 0.88, 0.84])
        scores_b = np.array([0.80, 0.82, 0.81, 0.83, 0.79])

        ci = paired_difference_ci(scores_a, scores_b, confidence_level=0.95)

        assert "mean_difference" in ci
        assert "lower" in ci
        assert "upper" in ci
        assert ci["mean_difference"] > 0  # A > B
        # If CI doesn't include 0, difference is significant
        assert "includes_zero" in ci


class TestComparisonTable:
    """Test comparison table generation"""

    def test_generate_comparison_table(self):
        """Test generating comparison table from results"""
        from evaluation.compare_statistical import generate_comparison_table

        results = {
            "ko-centaur": {
                "accuracy": np.array([0.85, 0.87, 0.86, 0.88, 0.84]),
                "log_likelihood": np.array([-0.45, -0.42, -0.44, -0.41, -0.46])
            },
            "exaone-base": {
                "accuracy": np.array([0.80, 0.82, 0.81, 0.83, 0.79]),
                "log_likelihood": np.array([-0.55, -0.52, -0.54, -0.51, -0.56])
            }
        }

        table = generate_comparison_table(results)

        assert "models" in table
        assert "metrics" in table
        assert "ko-centaur" in table["models"]
        assert "exaone-base" in table["models"]
        assert "accuracy" in table["metrics"]

    def test_comparison_table_with_statistics(self):
        """Test comparison table includes statistical measures"""
        from evaluation.compare_statistical import generate_comparison_table

        results = {
            "ko-centaur": {
                "accuracy": np.array([0.85, 0.87, 0.86])
            },
            "exaone-base": {
                "accuracy": np.array([0.80, 0.82, 0.81])
            }
        }

        table = generate_comparison_table(results, include_stats=True)

        # Should include mean, std, CI for each model
        for model in table["models"]:
            assert "mean_accuracy" in table["models"][model]
            assert "std_accuracy" in table["models"][model]
            assert "ci_accuracy" in table["models"][model]


class TestPairwiseComparison:
    """Test pairwise model comparison"""

    def test_pairwise_comparison_two_models(self):
        """Test comprehensive pairwise comparison"""
        from evaluation.compare_statistical import pairwise_comparison

        scores_a = np.array([0.85, 0.87, 0.86, 0.88, 0.84])
        scores_b = np.array([0.80, 0.82, 0.81, 0.83, 0.79])

        result = pairwise_comparison(scores_a, scores_b, "ko-centaur", "exaone-base")

        assert "model_a" in result
        assert "model_b" in result
        assert "t_test" in result
        assert "effect_size" in result
        assert "confidence_interval" in result
        assert result["model_a"] == "ko-centaur"
        assert result["model_b"] == "exaone-base"

    def test_pairwise_all_models(self):
        """Test all pairwise comparisons across multiple models"""
        from evaluation.compare_statistical import all_pairwise_comparisons

        results = {
            "ko-centaur": np.array([0.85, 0.87, 0.86]),
            "exaone-base": np.array([0.80, 0.82, 0.81]),
            "llama-centaur": np.array([0.88, 0.90, 0.89])
        }

        comparisons = all_pairwise_comparisons(results)

        # Should have 3 comparisons: (A,B), (A,C), (B,C)
        assert len(comparisons) == 3
        assert all("model_a" in c for c in comparisons)
        assert all("model_b" in c for c in comparisons)


class TestStatisticalReport:
    """Test statistical report generation"""

    def test_generate_statistical_report(self):
        """Test generating comprehensive statistical report"""
        from evaluation.compare_statistical import generate_statistical_report

        results = {
            "ko-centaur": {
                "accuracy": np.array([0.85, 0.87, 0.86, 0.88, 0.84]),
                "log_likelihood": np.array([-0.45, -0.42, -0.44, -0.41, -0.46])
            },
            "exaone-base": {
                "accuracy": np.array([0.80, 0.82, 0.81, 0.83, 0.79]),
                "log_likelihood": np.array([-0.55, -0.52, -0.54, -0.51, -0.56])
            }
        }

        report = generate_statistical_report(results)

        assert "summary" in report
        assert "pairwise_comparisons" in report
        assert "best_model" in report
        assert "rankings" in report

    def test_save_statistical_report(self):
        """Test saving statistical report to file"""
        from evaluation.compare_statistical import save_statistical_report

        report = {
            "summary": {"ko-centaur": {"mean_accuracy": 0.86}},
            "pairwise_comparisons": [],
            "best_model": "ko-centaur"
        }

        with patch('builtins.open', create=True) as mock_open, \
             patch('json.dump') as mock_dump:

            save_statistical_report(report, Path("/tmp/stats_report.json"))

            mock_open.assert_called_once()
            mock_dump.assert_called_once()


class TestModelRanking:
    """Test model ranking based on performance"""

    def test_rank_models_by_metric(self):
        """Test ranking models by single metric"""
        from evaluation.compare_statistical import rank_models

        results = {
            "ko-centaur": {"accuracy": 0.86},
            "exaone-base": {"accuracy": 0.81},
            "llama-centaur": {"accuracy": 0.89}
        }

        rankings = rank_models(results, metric="accuracy")

        assert len(rankings) == 3
        assert rankings[0]["model"] == "llama-centaur"
        assert rankings[1]["model"] == "ko-centaur"
        assert rankings[2]["model"] == "exaone-base"

    def test_rank_with_statistical_significance(self):
        """Test ranking includes significance testing"""
        from evaluation.compare_statistical import rank_models_with_significance

        results = {
            "ko-centaur": np.array([0.85, 0.87, 0.86]),
            "exaone-base": np.array([0.80, 0.82, 0.81]),
            "llama-centaur": np.array([0.88, 0.90, 0.89])
        }

        rankings = rank_models_with_significance(results)

        assert len(rankings) == 3
        # Should indicate which differences are significant
        assert "significantly_better_than" in rankings[0]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
