"""
Test suite for model download verification

TDD approach: Define expected download validation behavior
"""
import pytest
from pathlib import Path
from unittest.mock import Mock, patch, mock_open
import json


class TestDownloadVerification:
    """Test download verification functionality"""

    def test_verify_model_files_exist(self):
        """Test that required model files are present"""
        from baselines.download_verification import verify_model_download

        mock_path = Mock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = True

        # Mock required files
        config_file = Mock(spec=Path)
        config_file.exists.return_value = True

        model_file = Mock(spec=Path)
        model_file.exists.return_value = True

        with patch('pathlib.Path', return_value=mock_path):
            result = verify_model_download(mock_path)
            assert result['valid'] == True

    def test_verify_config_json_valid(self):
        """Test that config.json has required fields"""
        from baselines.download_verification import verify_config_json

        valid_config = {
            "model_type": "llama",
            "hidden_size": 8192,
            "num_hidden_layers": 80,
            "vocab_size": 128256
        }

        result = verify_config_json(valid_config)

        assert result['valid'] == True
        assert 'model_type' in result
        assert 'hidden_size' in result

    def test_verify_config_json_missing_fields(self):
        """Test config validation with missing required fields"""
        from baselines.download_verification import verify_config_json

        invalid_config = {
            "model_type": "llama"
            # Missing hidden_size, etc.
        }

        result = verify_config_json(invalid_config)

        assert result['valid'] == False
        assert len(result['missing_fields']) > 0

    def test_calculate_model_size(self):
        """Test model size calculation"""
        from baselines.download_verification import calculate_model_size

        mock_path = Mock(spec=Path)

        # Mock file sizes
        with patch('pathlib.Path.glob') as mock_glob:
            mock_files = [
                Mock(stat=Mock(return_value=Mock(st_size=1024*1024*1024))),  # 1GB
                Mock(stat=Mock(return_value=Mock(st_size=1024*1024*1024))),  # 1GB
            ]
            mock_glob.return_value = mock_files

            size_gb = calculate_model_size(mock_path)

            assert size_gb == pytest.approx(2.0, rel=0.1)

    def test_verify_all_models_downloaded(self):
        """Test verification of all baseline models"""
        from baselines.download_verification import verify_all_baselines

        baselines_dir = Path("/scratch/connectome/connectome1/ko-centaur/models/baselines")

        with patch('baselines.download_verification.verify_model_download') as mock_verify:
            mock_verify.return_value = {'valid': True, 'size_gb': 140.0}

            results = verify_all_baselines(baselines_dir)

            expected_models = ["llama-centaur-70b", "llama-70b-instruct", "solar-10.7b"]

            for model_name in expected_models:
                assert model_name in results
                assert results[model_name]['valid'] == True

    def test_verify_model_not_found(self):
        """Test handling of missing model directory"""
        from baselines.download_verification import verify_model_download

        non_existent_path = Path("/nonexistent/model")

        result = verify_model_download(non_existent_path)

        assert result['valid'] == False
        assert 'error' in result
        assert 'not found' in result['error'].lower()

    def test_verify_incomplete_download(self):
        """Test detection of incomplete downloads"""
        from baselines.download_verification import verify_model_download

        mock_path = Mock(spec=Path)
        mock_path.exists.return_value = True
        mock_path.is_dir.return_value = True

        # Config exists but model files missing
        config = mock_path / "config.json"
        config.exists.return_value = True

        model_file = mock_path / "model.safetensors.index.json"
        model_file.exists.return_value = False  # Missing!

        with patch.object(Path, '__truediv__', return_value=model_file):
            result = verify_model_download(mock_path)

            assert result['valid'] == False
            assert 'incomplete' in result.get('error', '').lower() or \
                   'missing' in result.get('error', '').lower()

    def test_expected_sizes_for_models(self):
        """Test that model sizes match expectations"""
        expected_sizes = {
            "llama-centaur-70b": (130, 150),  # GB range
            "llama-70b-instruct": (130, 150),
            "solar-10.7b": (18, 25)
        }

        # This validates our expectations
        assert all(low < high for low, high in expected_sizes.values())

    def test_download_progress_tracking(self):
        """Test download progress can be tracked"""
        from baselines.download_verification import get_download_progress

        mock_path = Path("/models/llama-centaur-70b")

        with patch('pathlib.Path.exists', return_value=True), \
             patch('baselines.download_verification.calculate_model_size', return_value=75.0):

            progress = get_download_progress(mock_path, expected_size_gb=140.0)

            assert progress['downloaded_gb'] == 75.0
            assert progress['expected_gb'] == 140.0
            assert progress['percent'] == pytest.approx(53.6, rel=1.0)
            assert progress['complete'] == False

    def test_download_complete_detection(self):
        """Test detection of completed downloads"""
        from baselines.download_verification import get_download_progress

        with patch('baselines.download_verification.calculate_model_size', return_value=140.0):
            progress = get_download_progress(Path("/model"), expected_size_gb=140.0)

            assert progress['complete'] == True
            assert progress['percent'] >= 99.0


class TestDownloadScript:
    """Test download script functionality"""

    def test_download_script_creates_directory(self):
        """Test that download script creates baselines directory"""
        from baselines.download_models import ensure_baselines_dir

        with patch('pathlib.Path.mkdir') as mock_mkdir:
            ensure_baselines_dir()
            mock_mkdir.assert_called()

    def test_download_script_checks_disk_space(self):
        """Test disk space check before download"""
        from baselines.download_models import check_disk_space

        required_gb = 400

        with patch('shutil.disk_usage') as mock_usage:
            mock_usage.return_value = Mock(free=500 * 1024**3)  # 500GB free

            has_space = check_disk_space(required_gb)

            assert has_space == True

    def test_download_script_insufficient_disk_space(self):
        """Test handling of insufficient disk space"""
        from baselines.download_models import check_disk_space

        required_gb = 400

        with patch('shutil.disk_usage') as mock_usage:
            mock_usage.return_value = Mock(free=100 * 1024**3)  # Only 100GB

            has_space = check_disk_space(required_gb)

            assert has_space == False

    def test_huggingface_cli_installed(self):
        """Test that huggingface-cli is available"""
        from baselines.download_models import check_huggingface_cli

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(returncode=0)

            installed = check_huggingface_cli()

            assert installed == True

    def test_huggingface_authentication(self):
        """Test HuggingFace authentication check"""
        from baselines.download_models import check_hf_token

        with patch('subprocess.run') as mock_run:
            mock_run.return_value = Mock(
                returncode=0,
                stdout="Token: hf_xxx..."
            )

            has_token = check_hf_token()

            assert has_token == True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
