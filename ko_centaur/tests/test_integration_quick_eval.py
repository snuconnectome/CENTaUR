"""
Integration test for quick evaluation workflow

TDD approach: Define complete 50-sample evaluation workflow
"""
import pytest
import torch
import json
from pathlib import Path
import tempfile
from unittest.mock import Mock, MagicMock


class TestQuickEvaluationWorkflow:
    """Integration test for 50-sample quick evaluation"""

    def test_end_to_end_quick_eval(self):
        """Test complete quick evaluation workflow with 50 samples"""
        from evaluation.quick_eval import run_quick_eval_pipeline

        # Create mock models
        mock_kocentaur = Mock()
        mock_kocentaur.model_type = "ko-centaur"
        mock_kocentaur.extract_features.return_value = torch.randn(1, 4096)

        mock_baseline = Mock()
        mock_baseline.model_type = "exaone-base"
        mock_baseline.extract_features.return_value = torch.randn(1, 4096)

        # Create 50-sample dataset
        samples = [
            {"task_description": f"심리학 질문 {i}"}
            for i in range(50)
        ]
        labels = torch.randint(0, 2, (50,))

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            result = run_quick_eval_pipeline(
                model_a=mock_kocentaur,
                model_b=mock_baseline,
                samples=samples,
                labels=labels,
                output_dir=output_dir
            )

            # Verify results structure
            assert "comparison" in result
            assert "report" in result
            assert "report_path" in result

            # Verify both models evaluated
            assert "ko-centaur" in result["comparison"]
            assert "exaone-base" in result["comparison"]

            # Verify metrics computed
            kocentaur_metrics = result["comparison"]["ko-centaur"]
            assert "accuracy" in kocentaur_metrics
            assert "log_likelihood" in kocentaur_metrics

    def test_quick_eval_with_feature_extraction(self):
        """Test quick eval extracts features efficiently"""
        from evaluation.extract_features import extract_features_batch

        mock_model = Mock()
        mock_model.extract_features.return_value = torch.randn(1, 4096)

        samples = [
            {"task_description": f"질문 {i}"}
            for i in range(50)
        ]

        features = extract_features_batch(mock_model, samples)

        assert features.shape == (50, 4096)
        assert mock_model.extract_features.call_count == 50

    def test_quick_eval_comparison_metrics(self):
        """Test quick eval computes all comparison metrics"""
        from evaluation.quick_eval import rapid_comparison

        # Mock features from both models
        features_kocentaur = torch.randn(50, 4096)
        features_baseline = torch.randn(50, 4096)
        labels = torch.randint(0, 2, (50,))

        result = rapid_comparison(
            features_a=features_kocentaur,
            features_b=features_baseline,
            labels=labels,
            model_names=("ko-centaur", "exaone-base")
        )

        # Verify both models have results
        assert "ko-centaur" in result
        assert "exaone-base" in result

        # Verify metrics present
        for model_name in ["ko-centaur", "exaone-base"]:
            assert "accuracy" in result[model_name]
            assert "log_likelihood" in result[model_name]
            assert "predictions" in result[model_name]

    def test_quick_eval_report_generation(self):
        """Test quick eval generates comparison report"""
        from evaluation.quick_eval import generate_comparison_report

        mock_results = {
            "ko-centaur": {
                "accuracy": 0.85,
                "log_likelihood": -0.45,
                "predictions": torch.randint(0, 2, (50,))
            },
            "exaone-base": {
                "accuracy": 0.78,
                "log_likelihood": -0.55,
                "predictions": torch.randint(0, 2, (50,))
            }
        }

        report = generate_comparison_report(mock_results)

        assert "summary" in report
        assert "winner" in report
        assert "improvement" in report

        # Ko-CENTaUR should win
        assert report["winner"] == "ko-centaur"
        assert abs(report["improvement"] - 0.07) < 1e-9  # Use approximate comparison for float


class TestQuickEvalAnalysis:
    """Test qualitative analysis in quick eval"""

    def test_agreement_disagreement_analysis(self):
        """Test analyzing where models agree/disagree"""
        from evaluation.quick_eval import qualitative_analysis

        features_a = torch.randn(50, 4096)
        features_b = torch.randn(50, 4096)
        labels = torch.randint(0, 2, (50,))

        analysis = qualitative_analysis(features_a, features_b, labels)

        assert "agreement" in analysis
        assert "disagreement" in analysis
        assert "feature_distance" in analysis

        # Agreement rate should be between 0 and 1
        assert 0 <= analysis["agreement"] <= 1

    def test_feature_space_distance(self):
        """Test computing distance in feature space"""
        from evaluation.quick_eval import compute_feature_distance

        features_a = torch.randn(50, 4096)
        features_b = torch.randn(50, 4096)

        distance = compute_feature_distance(features_a, features_b)

        assert distance > 0
        assert isinstance(distance, float)

    def test_disagreement_breakdown(self):
        """Test breaking down disagreement cases"""
        from evaluation.quick_eval import analyze_disagreements

        # Create specific predictions for testing
        predictions_a = torch.tensor([0, 1, 0, 1, 0, 1])
        predictions_b = torch.tensor([0, 1, 1, 0, 0, 1])
        labels = torch.tensor([0, 1, 0, 0, 0, 1])

        analysis = analyze_disagreements(predictions_a, predictions_b, labels)

        assert "total_disagreements" in analysis
        assert "both_wrong" in analysis
        assert "a_correct" in analysis
        assert "b_correct" in analysis

        # Disagreements: indices 2, 3 (2 total)
        assert analysis["total_disagreements"] == 2


class TestQuickEvalVisualization:
    """Test visualization preparation for quick eval"""

    def test_tsne_preparation(self):
        """Test preparing features for t-SNE visualization"""
        from evaluation.quick_eval import prepare_tsne_features

        features = torch.randn(50, 4096)

        tsne_features = prepare_tsne_features(features, n_components=2)

        assert tsne_features.shape == (50, 2)
        assert isinstance(tsne_features, np.ndarray)


class TestQuickEvalPerformance:
    """Test quick eval runs efficiently"""

    def test_quick_eval_time_constraint(self):
        """Test quick eval completes in reasonable time"""
        import time
        from evaluation.quick_eval import rapid_comparison

        features_a = torch.randn(50, 4096)
        features_b = torch.randn(50, 4096)
        labels = torch.randint(0, 2, (50,))

        start = time.time()
        result = rapid_comparison(
            features_a, features_b, labels,
            model_names=("ko-centaur", "exaone-base")
        )
        elapsed = time.time() - start

        # Should complete in < 1 second
        assert elapsed < 1.0

    def test_memory_efficient_processing(self):
        """Test quick eval doesn't use excessive memory"""
        features = torch.randn(50, 4096)

        # Check memory footprint
        memory_mb = features.element_size() * features.numel() / (1024 * 1024)

        # 50 samples * 4096 features * 4 bytes = ~0.8MB
        assert memory_mb < 10  # Should be under 10MB


import numpy as np


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
