"""
Test suite for quick evaluation system

TDD approach: Define expected rapid evaluation behavior
"""
import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch
import json


class TestQuickEval:
    """Test quick evaluation functionality"""

    def test_create_mini_test_set(self):
        """Test creating mini test set from full dataset"""
        from evaluation.quick_eval import create_mini_test_set

        # Mock full dataset with 1000 samples
        full_dataset = [
            {"task_description": f"Sample {i}", "label": i % 2}
            for i in range(1000)
        ]

        mini_set = create_mini_test_set(full_dataset, n_samples=50)

        assert len(mini_set) == 50
        # Check balanced sampling (allow up to 12 difference in 50 samples)
        labels = [s["label"] for s in mini_set]
        assert abs(labels.count(0) - labels.count(1)) <= 12  # Roughly balanced

    def test_create_stratified_mini_set(self):
        """Test stratified sampling preserves class distribution"""
        from evaluation.quick_eval import create_mini_test_set

        # Imbalanced dataset: 80% class 0, 20% class 1
        full_dataset = [
            {"task_description": f"Sample {i}", "label": 0 if i < 800 else 1}
            for i in range(1000)
        ]

        mini_set = create_mini_test_set(
            full_dataset, n_samples=100, stratified=True
        )

        labels = [s["label"] for s in mini_set]
        # Should maintain roughly 80/20 split
        assert 70 <= labels.count(0) <= 90
        assert 10 <= labels.count(1) <= 30

    def test_rapid_comparison_two_models(self):
        """Test rapid comparison between two models"""
        from evaluation.quick_eval import rapid_comparison

        # Mock features from two models
        features_a = torch.randn(50, 4096)
        features_b = torch.randn(50, 4096)
        labels = torch.randint(0, 2, (50,))

        result = rapid_comparison(
            features_a=features_a,
            features_b=features_b,
            labels=labels,
            model_names=("ko-centaur", "exaone-base")
        )

        assert "ko-centaur" in result
        assert "exaone-base" in result
        assert "accuracy" in result["ko-centaur"]
        assert "log_likelihood" in result["ko-centaur"]

    def test_qualitative_analysis(self):
        """Test qualitative difference analysis"""
        from evaluation.quick_eval import qualitative_analysis

        features_a = torch.randn(50, 4096)
        features_b = torch.randn(50, 4096)
        labels = torch.randint(0, 2, (50,))

        analysis = qualitative_analysis(features_a, features_b, labels)

        assert "agreement" in analysis
        assert "disagreement" in analysis
        assert "feature_distance" in analysis
        assert isinstance(analysis["agreement"], float)

    def test_tsne_visualization_preparation(self):
        """Test t-SNE feature preparation"""
        from evaluation.quick_eval import prepare_tsne_features

        features = torch.randn(100, 4096)
        labels = torch.randint(0, 2, (100,))

        tsne_features = prepare_tsne_features(features, n_components=2)

        assert tsne_features.shape == (100, 2)
        assert isinstance(tsne_features, np.ndarray)

    def test_comparison_report_generation(self):
        """Test comparison report generation"""
        from evaluation.quick_eval import generate_comparison_report

        results = {
            "ko-centaur": {
                "accuracy": 0.85,
                "log_likelihood": -0.45,
                "predictions": torch.randint(0, 2, (50,))
            },
            "exaone-base": {
                "accuracy": 0.78,
                "log_likelihood": -0.62,
                "predictions": torch.randint(0, 2, (50,))
            }
        }

        report = generate_comparison_report(results)

        assert "summary" in report
        assert "winner" in report
        assert "improvement" in report

    def test_quick_eval_with_checkpoint_loading(self):
        """Test quick evaluation with model checkpoint loading"""
        from evaluation.quick_eval import quick_eval_with_checkpoints

        mock_manager_a = Mock()
        mock_manager_a.extract_features.return_value = torch.randn(1, 4096)

        mock_manager_b = Mock()
        mock_manager_b.extract_features.return_value = torch.randn(1, 4096)

        samples = [{"task_description": f"Sample {i}"} for i in range(10)]
        labels = torch.randint(0, 2, (10,))

        with patch('baselines.load_baselines.BaselineModelManager') as MockManager:
            MockManager.side_effect = [mock_manager_a, mock_manager_b]

            result = quick_eval_with_checkpoints(
                model_a_type="ko-centaur",
                model_b_type="exaone-base",
                samples=samples,
                labels=labels
            )

            assert "ko-centaur" in result
            assert "exaone-base" in result


class TestPredictionAnalysis:
    """Test prediction analysis functionality"""

    def test_compute_accuracy(self):
        """Test accuracy computation"""
        from evaluation.quick_eval import compute_accuracy

        predictions = torch.tensor([0, 1, 0, 1, 0])
        labels = torch.tensor([0, 1, 0, 0, 0])

        accuracy = compute_accuracy(predictions, labels)

        assert accuracy == 0.8  # 4/5 correct

    def test_compute_log_likelihood(self):
        """Test log-likelihood computation"""
        from evaluation.quick_eval import compute_log_likelihood

        # Mock probabilities for 5 samples, 2 classes
        probabilities = torch.tensor([
            [0.9, 0.1],  # Predicts class 0 with 90% confidence
            [0.2, 0.8],  # Predicts class 1 with 80% confidence
            [0.7, 0.3],  # Predicts class 0 with 70% confidence
            [0.4, 0.6],  # Predicts class 1 with 60% confidence
            [0.8, 0.2],  # Predicts class 0 with 80% confidence
        ])
        labels = torch.tensor([0, 1, 0, 1, 0])

        log_likelihood = compute_log_likelihood(probabilities, labels)

        # Should be negative (log of probabilities)
        assert log_likelihood < 0
        assert isinstance(log_likelihood, float)

    def test_confusion_matrix_computation(self):
        """Test confusion matrix computation"""
        from evaluation.quick_eval import compute_confusion_matrix

        predictions = torch.tensor([0, 1, 0, 1, 0, 1, 1, 0])
        labels = torch.tensor([0, 1, 0, 0, 0, 1, 0, 1])

        confusion = compute_confusion_matrix(predictions, labels, n_classes=2)

        assert confusion.shape == (2, 2)
        # Check that diagonal and off-diagonal sum to total samples
        assert confusion.sum() == len(predictions)

    def test_per_class_metrics(self):
        """Test per-class precision, recall, F1"""
        from evaluation.quick_eval import compute_per_class_metrics

        predictions = torch.tensor([0, 1, 0, 1, 0, 1, 1, 0])
        labels = torch.tensor([0, 1, 0, 0, 0, 1, 0, 1])

        metrics = compute_per_class_metrics(predictions, labels)

        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert len(metrics["precision"]) == 2  # 2 classes


class TestComparisonMetrics:
    """Test model comparison metrics"""

    def test_agreement_rate(self):
        """Test agreement rate between two models"""
        from evaluation.quick_eval import compute_agreement_rate

        predictions_a = torch.tensor([0, 1, 0, 1, 0])
        predictions_b = torch.tensor([0, 1, 0, 0, 0])

        agreement = compute_agreement_rate(predictions_a, predictions_b)

        assert agreement == 0.8  # 4/5 agree

    def test_feature_distance(self):
        """Test feature space distance between models"""
        from evaluation.quick_eval import compute_feature_distance

        features_a = torch.randn(50, 4096)
        features_b = torch.randn(50, 4096)

        distance = compute_feature_distance(features_a, features_b)

        assert distance > 0
        assert isinstance(distance, float)

    def test_disagreement_analysis(self):
        """Test analysis of disagreement cases"""
        from evaluation.quick_eval import analyze_disagreements

        predictions_a = torch.tensor([0, 1, 0, 1, 0])
        predictions_b = torch.tensor([0, 1, 0, 0, 0])
        labels = torch.tensor([0, 1, 0, 0, 0])

        analysis = analyze_disagreements(predictions_a, predictions_b, labels)

        assert "total_disagreements" in analysis
        assert "both_wrong" in analysis
        assert "a_correct" in analysis
        assert "b_correct" in analysis


class TestQuickEvalPipeline:
    """Test end-to-end quick evaluation pipeline"""

    def test_full_quick_eval_pipeline(self):
        """Test complete quick evaluation workflow"""
        from evaluation.quick_eval import run_quick_eval_pipeline

        # Mock models
        mock_manager_a = Mock()
        mock_manager_a.model_type = "ko-centaur"
        mock_manager_a.extract_features.return_value = torch.randn(1, 4096)

        mock_manager_b = Mock()
        mock_manager_b.model_type = "exaone-base"
        mock_manager_b.extract_features.return_value = torch.randn(1, 4096)

        # Mock dataset
        samples = [{"task_description": f"Sample {i}"} for i in range(20)]
        labels = torch.randint(0, 2, (20,))

        with patch('evaluation.quick_eval.save_report') as mock_save:
            result = run_quick_eval_pipeline(
                model_a=mock_manager_a,
                model_b=mock_manager_b,
                samples=samples,
                labels=labels,
                output_dir=Path("/tmp/quick_eval")
            )

            assert "comparison" in result
            assert "report" in result
            mock_save.assert_called_once()

    def test_save_quick_eval_report(self):
        """Test saving evaluation report"""
        from evaluation.quick_eval import save_report

        report = {
            "summary": {
                "ko-centaur": {"accuracy": 0.85},
                "exaone-base": {"accuracy": 0.78}
            },
            "winner": "ko-centaur",
            "improvement": 0.07
        }

        with patch('builtins.open', create=True) as mock_open, \
             patch('json.dump') as mock_dump:

            save_report(report, Path("/tmp/report.json"))

            mock_open.assert_called_once()
            mock_dump.assert_called_once()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
