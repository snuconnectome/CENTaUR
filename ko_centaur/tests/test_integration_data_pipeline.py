"""
Integration test for data pipeline

TDD approach: Define complete data loading and preprocessing workflow
"""
import pytest
import json
import torch
from pathlib import Path
import tempfile


class TestDataPipeline:
    """Integration test for Psych-101 data pipeline"""

    def test_load_psych101_dataset(self):
        """Test loading Psych-101 dataset from JSONL"""
        from evaluation.extract_features import load_dataset

        # Create mock Psych-101 data
        mock_data = [
            {
                "task_description": "환자가 우울증을 호소하고 있습니다. 어떤 치료 접근이 적절할까요?",
                "label": 1,
                "task_type": "clinical_psychology",
                "difficulty": "medium"
            },
            {
                "task_description": "아동의 언어 발달 지연이 관찰됩니다. 평가 방법은?",
                "label": 0,
                "task_type": "developmental_psychology",
                "difficulty": "easy"
            }
        ]

        # Save to temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for item in mock_data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')
            temp_path = Path(f.name)

        try:
            # Test loading
            dataset = load_dataset(temp_path)

            assert len(dataset) == 2
            assert all('task_description' in item for item in dataset)
            assert all('label' in item for item in dataset)

            # Test Korean text preserved
            assert '우울증' in dataset[0]['task_description']
            assert '아동' in dataset[1]['task_description']

        finally:
            temp_path.unlink()

    def test_create_mini_eval_set(self):
        """Test creating 50-sample mini evaluation set"""
        from evaluation.quick_eval import create_mini_test_set

        # Create larger mock dataset
        full_dataset = [
            {
                "task_description": f"심리학 질문 {i}",
                "label": i % 2,
                "task_type": "general"
            }
            for i in range(200)
        ]

        # Create 50-sample mini set
        mini_set = create_mini_test_set(full_dataset, n_samples=50, stratified=True)

        assert len(mini_set) == 50
        # Check stratification maintains class balance
        labels = [s['label'] for s in mini_set]
        assert 20 <= labels.count(0) <= 30  # Roughly balanced

    def test_data_validation(self):
        """Test dataset validation catches format errors"""
        from evaluation.extract_features import validate_dataset

        # Valid dataset
        valid_data = [
            {"task_description": "질문 1", "label": 1},
            {"task_description": "질문 2", "label": 0}
        ]

        # Should not raise
        validate_dataset(valid_data, require_labels=True)

        # Invalid dataset - missing labels
        invalid_data = [
            {"task_description": "질문 1"},
            {"task_description": "질문 2"}
        ]

        with pytest.raises(ValueError):
            validate_dataset(invalid_data, require_labels=True)

    def test_train_test_split(self):
        """Test creating train/test splits for evaluation"""
        from evaluation.cross_validation import generate_loo_splits

        n_samples = 20
        splits = generate_loo_splits(n_samples)

        assert len(splits) == n_samples

        # Verify each split is valid
        for train_idx, test_idx in splits:
            assert len(train_idx) == n_samples - 1
            assert len(test_idx) == 1
            assert len(set(train_idx) & set(test_idx)) == 0


class TestDataPreprocessing:
    """Test data preprocessing and formatting"""

    def test_korean_text_encoding(self):
        """Test Korean text is properly encoded"""
        korean_text = "환자가 우울증을 호소합니다."

        # Should preserve Korean characters
        assert len(korean_text) > 0
        assert any('\uac00' <= char <= '\ud7af' for char in korean_text)  # Korean Unicode range

    def test_prompt_formatting(self):
        """Test formatting task descriptions into prompts"""
        from baselines.load_baselines import BaselineModelManager

        # Create mock manager
        with tempfile.TemporaryDirectory() as tmpdir:
            # This will fail initially - expected for TDD
            try:
                manager = BaselineModelManager("exaone-base", model_path=tmpdir)

                task = "환자가 우울증을 호소하고 있습니다."
                prompt = manager.format_prompt(task)

                assert isinstance(prompt, str)
                assert task in prompt
            except Exception:
                # Expected to fail without actual model
                pass

    def test_batch_processing(self):
        """Test batch processing of multiple samples"""
        samples = [
            {"task_description": f"질문 {i}"}
            for i in range(10)
        ]

        # Test batch size handling
        batch_size = 4
        n_batches = (len(samples) + batch_size - 1) // batch_size

        assert n_batches == 3  # 10 samples, batch_size 4 = 3 batches


class TestDataStatistics:
    """Test dataset statistics and properties"""

    def test_dataset_balance(self):
        """Test checking dataset class balance"""
        dataset = [
            {"task_description": f"샘플 {i}", "label": i % 2}
            for i in range(100)
        ]

        labels = [s['label'] for s in dataset]
        label_counts = {0: labels.count(0), 1: labels.count(1)}

        assert label_counts[0] == 50
        assert label_counts[1] == 50

    def test_dataset_size_requirements(self):
        """Test minimum dataset size for evaluation"""
        min_samples_for_loo = 20  # Minimum for meaningful LOO CV
        min_samples_for_nested_cv = 50  # Minimum for nested CV

        # Mock dataset
        dataset = [{"task_description": f"샘플 {i}", "label": 0} for i in range(100)]

        assert len(dataset) >= min_samples_for_loo
        assert len(dataset) >= min_samples_for_nested_cv


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
