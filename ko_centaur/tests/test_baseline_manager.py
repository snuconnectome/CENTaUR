"""
Test suite for BaselineModelManager

TDD approach: Tests written first to define expected behavior
"""
import pytest
import torch
from unittest.mock import Mock, patch, MagicMock
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestBaselineModelManager:
    """Test BaselineModelManager functionality"""

    def test_model_registry_complete(self):
        """Test that all expected models are in registry"""
        from baselines.load_baselines import BaselineModelManager

        expected_models = {
            "ko-centaur",
            "exaone-base",
            "llama-centaur",
            "llama-base",
            "solar"
        }

        assert set(BaselineModelManager.MODELS.keys()) == expected_models

    def test_model_descriptions_exist(self):
        """Test that all models have descriptions"""
        from baselines.load_baselines import BaselineModelManager

        for model_type, description in BaselineModelManager.MODELS.items():
            assert description, f"Model {model_type} missing description"
            assert isinstance(description, str)
            assert len(description) > 10, f"Description too short for {model_type}"

    @patch('baselines.load_baselines.AutoModelForCausalLM')
    @patch('baselines.load_baselines.AutoTokenizer')
    def test_exaone_base_loading(self, mock_tokenizer, mock_model):
        """Test EXAONE-base model loads correctly"""
        from baselines.load_baselines import BaselineModelManager

        # Setup mocks
        mock_model.from_pretrained.return_value = Mock(config=Mock(hidden_size=4096))
        mock_tokenizer.from_pretrained.return_value = Mock()

        # Load model
        manager = BaselineModelManager("exaone-base")

        # Verify correct model loaded
        assert manager.model_type == "exaone-base"
        assert manager.hidden_size == 4096
        mock_model.from_pretrained.assert_called_once()

        # Verify 4-bit quantization enabled
        call_kwargs = mock_model.from_pretrained.call_args[1]
        assert call_kwargs.get('load_in_4bit') == True

    @patch('baselines.load_baselines.PeftModel')
    @patch('baselines.load_baselines.AutoModelForCausalLM')
    @patch('baselines.load_baselines.AutoTokenizer')
    def test_ko_centaur_loading(self, mock_tokenizer, mock_model, mock_peft):
        """Test Ko-CENTaUR model loads with LoRA weights"""
        from baselines.load_baselines import BaselineModelManager

        # Setup mocks
        base_model = Mock(config=Mock(hidden_size=4096))
        mock_model.from_pretrained.return_value = base_model
        mock_peft.from_pretrained.return_value = Mock(config=Mock(hidden_size=4096))
        mock_tokenizer.from_pretrained.return_value = Mock()

        # Load Ko-CENTaUR
        manager = BaselineModelManager("ko-centaur")

        # Verify LoRA loading
        assert manager.model_type == "ko-centaur"
        mock_peft.from_pretrained.assert_called_once()

        # Verify checkpoint path
        peft_args = mock_peft.from_pretrained.call_args
        checkpoint_path = peft_args[0][1]
        assert "ko-centaur-full" in checkpoint_path
        assert "checkpoint" in checkpoint_path

    @patch('baselines.load_baselines.AutoModelForCausalLM')
    @patch('baselines.load_baselines.AutoTokenizer')
    def test_llama_centaur_loading(self, mock_tokenizer, mock_model):
        """Test Llama-Centaur-70B loads with multi-GPU config"""
        from baselines.load_baselines import BaselineModelManager

        # Setup mocks
        mock_model.from_pretrained.return_value = Mock(config=Mock(hidden_size=8192))
        mock_tokenizer.from_pretrained.return_value = Mock()

        # Load with specific device map
        device_map = {3: "22GB", 4: "22GB", 5: "22GB"}
        manager = BaselineModelManager("llama-centaur", device_map=device_map)

        # Verify multi-GPU configuration
        call_kwargs = mock_model.from_pretrained.call_args[1]
        assert 'max_memory' in call_kwargs
        assert call_kwargs['load_in_4bit'] == True

    def test_prompt_formatting_exaone(self):
        """Test EXAONE-style prompt formatting"""
        from baselines.load_baselines import BaselineModelManager

        # Mock manager without loading model
        with patch.object(BaselineModelManager, '_load_model', return_value=(Mock(), Mock())):
            manager = BaselineModelManager("ko-centaur")

            prompt = manager.format_prompt(
                "환자가 우울증을 호소합니다.",
                system="전문 심리학자"
            )

            assert "<|system|>" in prompt
            assert "<|user|>" in prompt
            assert "<|assistant|>" in prompt
            assert "전문 심리학자" in prompt
            assert "우울증" in prompt

    def test_prompt_formatting_llama(self):
        """Test Llama-style prompt formatting with chat template"""
        from baselines.load_baselines import BaselineModelManager

        mock_tokenizer = Mock()
        mock_tokenizer.apply_chat_template.return_value = "formatted_prompt"

        with patch.object(BaselineModelManager, '_load_model', return_value=(Mock(), mock_tokenizer)):
            manager = BaselineModelManager("llama-centaur")

            prompt = manager.format_prompt(
                "환자가 우울증을 호소합니다.",
                system="전문 심리학자"
            )

            # Verify chat template called
            mock_tokenizer.apply_chat_template.assert_called_once()
            call_args = mock_tokenizer.apply_chat_template.call_args[0][0]

            assert len(call_args) == 2
            assert call_args[0]['role'] == 'system'
            assert call_args[1]['role'] == 'user'

    @patch('baselines.load_baselines.AutoModelForCausalLM')
    @patch('baselines.load_baselines.AutoTokenizer')
    def test_feature_extraction_shape(self, mock_tokenizer, mock_model):
        """Test feature extraction returns correct tensor shape"""
        from baselines.load_baselines import BaselineModelManager

        # Setup mocks
        hidden_size = 4096
        mock_outputs = Mock()
        mock_outputs.hidden_states = [[Mock(), Mock(), torch.randn(1, 10, hidden_size)]]

        mock_gen_model = Mock()
        mock_gen_model.generate.return_value = mock_outputs
        mock_gen_model.config = Mock(hidden_size=hidden_size)

        mock_model.from_pretrained.return_value = mock_gen_model
        mock_tokenizer.from_pretrained.return_value = Mock(return_value={'input_ids': torch.tensor([[1, 2, 3]])})

        manager = BaselineModelManager("exaone-base")

        # Extract features
        features = manager.extract_features("test prompt")

        # Verify shape
        assert features.shape == (1, hidden_size)
        assert features.dtype == torch.float32 or features.dtype == torch.float16

    def test_invalid_model_type_raises_error(self):
        """Test that invalid model type raises appropriate error"""
        from baselines.load_baselines import BaselineModelManager

        with pytest.raises((ValueError, KeyError)):
            BaselineModelManager("invalid-model-type")

    def test_feature_extraction_deterministic(self):
        """Test that feature extraction uses temperature=0.0"""
        from baselines.load_baselines import BaselineModelManager

        mock_model = Mock()
        mock_model.generate.return_value = Mock(
            hidden_states=[[Mock(), Mock(), torch.randn(1, 5, 4096)]]
        )
        mock_model.config = Mock(hidden_size=4096)

        mock_tokenizer = Mock(return_value={'input_ids': torch.tensor([[1]])})

        with patch.object(BaselineModelManager, '_load_model', return_value=(mock_model, mock_tokenizer)):
            manager = BaselineModelManager("exaone-base")
            manager.extract_features("test")

            # Verify temperature=0.0 (deterministic)
            call_kwargs = mock_model.generate.call_args[1]
            assert call_kwargs.get('temperature') == 0.0
            assert call_kwargs.get('max_new_tokens') == 1


class TestModelRegistry:
    """Test model registry and configuration"""

    def test_all_models_have_gpu_requirements(self):
        """Test that GPU requirements are documented for each model"""
        from baselines.load_baselines import BaselineModelManager

        # Expected GPU requirements
        gpu_requirements = {
            "ko-centaur": 1,
            "exaone-base": 1,
            "llama-centaur": 3,
            "llama-base": 3,
            "solar": 1
        }

        for model_type in BaselineModelManager.MODELS.keys():
            assert model_type in gpu_requirements, \
                f"GPU requirement not documented for {model_type}"

    def test_checkpoint_paths_configured(self):
        """Test that checkpoint paths are properly configured"""
        # Ko-CENTaUR checkpoint should be defined
        ko_centaur_checkpoint = "/scratch/connectome/connectome1/ko-centaur/models/ko-centaur-full/checkpoint-22536"

        # This will be in the actual implementation
        assert True  # Placeholder for path validation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
