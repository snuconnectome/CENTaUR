"""
Test suite for feature extraction pipeline

TDD approach: Define expected extraction behavior
"""
import pytest
import torch
import numpy as np
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import json


class TestFeatureExtraction:
    """Test feature extraction functionality"""

    def test_extract_single_sample(self):
        """Test feature extraction for single sample"""
        from evaluation.extract_features import extract_features_for_sample

        mock_manager = Mock()
        mock_manager.extract_features.return_value = torch.randn(1, 4096)

        sample = {
            "task_description": "환자가 우울증을 호소합니다.",
            "system_prompt": "전문 심리학자"
        }

        features = extract_features_for_sample(mock_manager, sample)

        assert features.shape == (1, 4096)
        assert isinstance(features, torch.Tensor)
        mock_manager.extract_features.assert_called_once()

    def test_extract_batch_samples(self):
        """Test batch feature extraction"""
        from evaluation.extract_features import extract_features_batch

        mock_manager = Mock()
        mock_manager.extract_features.return_value = torch.randn(1, 4096)

        samples = [
            {"task_description": "Sample 1"},
            {"task_description": "Sample 2"},
            {"task_description": "Sample 3"}
        ]

        features = extract_features_batch(mock_manager, samples)

        assert features.shape == (3, 4096)
        assert mock_manager.extract_features.call_count == 3

    def test_save_features_to_file(self):
        """Test saving features to .pth file"""
        from evaluation.extract_features import save_features

        features = torch.randn(10, 4096)
        metadata = {
            "model_type": "ko-centaur",
            "num_samples": 10,
            "hidden_size": 4096
        }

        with patch('torch.save') as mock_save:
            save_features(features, metadata, Path("/tmp/features.pth"))

            mock_save.assert_called_once()
            saved_data = mock_save.call_args[0][0]

            assert 'features' in saved_data
            assert 'metadata' in saved_data
            assert saved_data['metadata']['num_samples'] == 10

    def test_load_features_from_file(self):
        """Test loading features from .pth file"""
        from evaluation.extract_features import load_features

        mock_data = {
            'features': torch.randn(10, 4096),
            'metadata': {'model_type': 'ko-centaur', 'num_samples': 10}
        }

        with patch('torch.load', return_value=mock_data):
            features, metadata = load_features(Path("/tmp/features.pth"))

            assert features.shape == (10, 4096)
            assert metadata['model_type'] == 'ko-centaur'
            assert metadata['num_samples'] == 10

    def test_feature_normalization(self):
        """Test feature normalization (z-score)"""
        from evaluation.extract_features import normalize_features

        # Create features with known statistics
        features = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])

        normalized = normalize_features(features)

        # Check mean ≈ 0, std ≈ 1
        assert torch.abs(normalized.mean()) < 0.01
        assert torch.abs(normalized.std() - 1.0) < 0.01

    def test_extract_with_progress_tracking(self):
        """Test extraction with progress callback"""
        from evaluation.extract_features import extract_with_progress

        mock_manager = Mock()
        mock_manager.extract_features.return_value = torch.randn(1, 4096)

        samples = [{"task_description": f"Sample {i}"} for i in range(5)]

        progress_updates = []

        def progress_callback(current, total):
            progress_updates.append((current, total))

        features = extract_with_progress(mock_manager, samples, progress_callback)

        assert len(progress_updates) == 5
        assert progress_updates[-1] == (5, 5)
        assert features.shape == (5, 4096)

    def test_feature_cache_hit(self):
        """Test cache hit when features already extracted"""
        from evaluation.extract_features import extract_features_cached

        cache_path = Path("/tmp/cache/features.pth")

        # Mock cache exists
        mock_data = {
            'features': torch.randn(10, 4096),
            'metadata': {'model_type': 'ko-centaur'}
        }

        with patch('pathlib.Path.exists', return_value=True), \
             patch('torch.load', return_value=mock_data):

            features, from_cache = extract_features_cached(
                model_manager=Mock(),
                samples=[],
                cache_path=cache_path
            )

            assert from_cache == True
            assert features.shape == (10, 4096)

    def test_feature_cache_miss(self):
        """Test cache miss requires extraction"""
        from evaluation.extract_features import extract_features_cached

        mock_manager = Mock()
        mock_manager.extract_features.return_value = torch.randn(1, 4096)

        samples = [{"task_description": "Sample"}]
        cache_path = Path("/tmp/cache/features.pth")

        with patch('pathlib.Path.exists', return_value=False), \
             patch('evaluation.extract_features.save_features') as mock_save:

            features, from_cache = extract_features_cached(
                model_manager=mock_manager,
                samples=samples,
                cache_path=cache_path
            )

            assert from_cache == False
            mock_save.assert_called_once()

    def test_parallel_extraction(self):
        """Test parallel feature extraction across models"""
        from evaluation.extract_features import extract_parallel

        mock_managers = {
            "ko-centaur": Mock(),
            "exaone-base": Mock()
        }

        for manager in mock_managers.values():
            manager.extract_features.return_value = torch.randn(1, 4096)

        samples = [{"task_description": "Sample"}]

        results = extract_parallel(mock_managers, samples)

        assert "ko-centaur" in results
        assert "exaone-base" in results
        assert results["ko-centaur"].shape == (1, 4096)

    def test_feature_dimension_consistency(self):
        """Test all models produce consistent feature dimensions"""
        from evaluation.extract_features import validate_feature_dimensions

        features_dict = {
            "ko-centaur": torch.randn(10, 4096),
            "exaone-base": torch.randn(10, 4096),
            "llama-centaur": torch.randn(10, 8192)  # Different size OK
        }

        # Should not raise error - each model can have different dimensions
        validate_feature_dimensions(features_dict)

        # But each model's features must be consistent
        features_dict["ko-centaur"] = torch.randn(10, 2048)  # Inconsistent!

        with pytest.raises(ValueError):
            validate_feature_dimensions(features_dict, expected_sizes={"ko-centaur": 4096})


class TestDatasetLoading:
    """Test dataset loading for feature extraction"""

    def test_load_jsonl_dataset(self):
        """Test loading JSONL dataset"""
        from evaluation.extract_features import load_dataset

        jsonl_content = '{"task_description": "Sample 1"}\n{"task_description": "Sample 2"}\n'

        with patch('builtins.open', mock_open(read_data=jsonl_content)):
            samples = load_dataset(Path("/tmp/test.jsonl"))

            assert len(samples) == 2
            assert samples[0]["task_description"] == "Sample 1"

    def test_load_dataset_with_labels(self):
        """Test loading dataset with ground truth labels"""
        from evaluation.extract_features import load_dataset

        jsonl_content = '{"task_description": "S1", "label": 1}\n{"task_description": "S2", "label": 0}\n'

        with patch('builtins.open', mock_open(read_data=jsonl_content)):
            samples = load_dataset(Path("/tmp/test.jsonl"))

            assert all('label' in s for s in samples)
            assert samples[0]["label"] == 1

    def test_dataset_validation(self):
        """Test dataset schema validation"""
        from evaluation.extract_features import validate_dataset

        valid_samples = [
            {"task_description": "Sample 1", "label": 1},
            {"task_description": "Sample 2", "label": 0}
        ]

        # Should not raise
        validate_dataset(valid_samples)

        invalid_samples = [
            {"task_description": "Sample 1"},  # Missing label
            {"label": 1}  # Missing task_description
        ]

        with pytest.raises(ValueError):
            validate_dataset(invalid_samples, require_labels=True)


class TestExtractionPipeline:
    """Test end-to-end extraction pipeline"""

    def test_full_pipeline_single_model(self):
        """Test complete extraction pipeline for one model"""
        from evaluation.extract_features import run_extraction_pipeline

        mock_manager = Mock()
        mock_manager.model_type = "ko-centaur"
        mock_manager.extract_features.return_value = torch.randn(1, 4096)

        samples = [{"task_description": f"Sample {i}"} for i in range(5)]

        with patch('evaluation.extract_features.save_features') as mock_save:
            features, metadata = run_extraction_pipeline(
                model_manager=mock_manager,
                samples=samples,
                output_dir=Path("/tmp/output")
            )

            assert features.shape == (5, 4096)
            assert metadata['model_type'] == 'ko-centaur'
            assert metadata['num_samples'] == 5
            mock_save.assert_called_once()

    def test_full_pipeline_all_models(self):
        """Test extraction pipeline for all baseline models"""
        from evaluation.extract_features import run_extraction_all_models

        mock_managers = {
            "ko-centaur": Mock(),
            "exaone-base": Mock()
        }

        for name, manager in mock_managers.items():
            manager.model_type = name
            manager.extract_features.return_value = torch.randn(1, 4096)

        samples = [{"task_description": "Sample"}]

        with patch('evaluation.extract_features.save_features'):
            results = run_extraction_all_models(
                model_managers=mock_managers,
                samples=samples,
                output_dir=Path("/tmp/output")
            )

            assert "ko-centaur" in results
            assert "exaone-base" in results
            assert results["ko-centaur"]['features'].shape == (1, 4096)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
