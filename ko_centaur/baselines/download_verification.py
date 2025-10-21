"""
Download verification utilities for baseline models

TDD Implementation: Passes all tests in test_download_verification.py
"""
import json
from pathlib import Path
from typing import Dict, List, Optional
import subprocess


def verify_config_json(config: Dict) -> Dict:
    """
    Verify config.json has required fields

    Args:
        config: Parsed config.json dict

    Returns:
        Dict with 'valid' bool and details
    """
    required_fields = ['model_type', 'hidden_size']
    missing_fields = [f for f in required_fields if f not in config]

    result = {
        'valid': len(missing_fields) == 0,
        'missing_fields': missing_fields
    }

    if result['valid']:
        result['model_type'] = config['model_type']
        result['hidden_size'] = config['hidden_size']

    return result


def calculate_model_size(model_path: Path) -> float:
    """
    Calculate total model size in GB

    Args:
        model_path: Path to model directory

    Returns:
        Size in GB
    """
    if not model_path.exists():
        return 0.0

    total_bytes = 0

    # Sum all model files
    for file_pattern in ['*.safetensors', '*.bin', '*.pth', '*.pt']:
        for file in model_path.glob(file_pattern):
            if file.is_file():
                total_bytes += file.stat().st_size

    # Convert to GB
    size_gb = total_bytes / (1024 ** 3)
    return size_gb


def verify_model_download(model_path: Path) -> Dict:
    """
    Verify model download is complete

    Args:
        model_path: Path to model directory

    Returns:
        Dict with verification results
    """
    if not model_path.exists():
        return {
            'valid': False,
            'error': f'Model directory not found: {model_path}'
        }

    if not model_path.is_dir():
        return {
            'valid': False,
            'error': f'Path is not a directory: {model_path}'
        }

    # Check for required files
    config_file = model_path / "config.json"
    if not config_file.exists():
        return {
            'valid': False,
            'error': 'Missing config.json'
        }

    # Check for model files
    has_safetensors = len(list(model_path.glob("*.safetensors*"))) > 0
    has_bin = len(list(model_path.glob("*.bin"))) > 0

    if not (has_safetensors or has_bin):
        return {
            'valid': False,
            'error': 'Incomplete download: missing model weight files'
        }

    # Calculate size
    size_gb = calculate_model_size(model_path)

    return {
        'valid': True,
        'size_gb': round(size_gb, 2),
        'path': str(model_path)
    }


def verify_all_baselines(baselines_dir: Path) -> Dict[str, Dict]:
    """
    Verify all baseline model downloads

    Args:
        baselines_dir: Path to baselines directory

    Returns:
        Dict mapping model names to verification results
    """
    expected_models = {
        "llama-centaur-70b": "Llama-3.1-Centaur-70B",
        "llama-70b-instruct": "Llama-3.1-70B-Instruct",
        "solar-10.7b": "SOLAR-10.7B"
    }

    results = {}

    for model_name, description in expected_models.items():
        model_path = baselines_dir / model_name
        result = verify_model_download(model_path)
        result['description'] = description
        results[model_name] = result

    return results


def get_download_progress(model_path: Path, expected_size_gb: float) -> Dict:
    """
    Get download progress for a model

    Args:
        model_path: Path to model directory
        expected_size_gb: Expected final size in GB

    Returns:
        Dict with progress information
    """
    if not model_path.exists():
        return {
            'downloaded_gb': 0.0,
            'expected_gb': expected_size_gb,
            'percent': 0.0,
            'complete': False
        }

    downloaded_gb = calculate_model_size(model_path)
    percent = (downloaded_gb / expected_size_gb) * 100 if expected_size_gb > 0 else 0
    complete = percent >= 99.0  # Consider complete if >= 99%

    return {
        'downloaded_gb': round(downloaded_gb, 2),
        'expected_gb': expected_size_gb,
        'percent': round(percent, 1),
        'complete': complete
    }


def check_huggingface_cli() -> bool:
    """Check if huggingface-cli is installed"""
    try:
        result = subprocess.run(
            ['huggingface-cli', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def check_hf_token() -> bool:
    """Check if HuggingFace token is configured"""
    try:
        result = subprocess.run(
            ['huggingface-cli', 'whoami'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0 and 'token' in result.stdout.lower()
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def print_verification_report(results: Dict[str, Dict]):
    """
    Print human-readable verification report

    Args:
        results: Results from verify_all_baselines()
    """
    print("\n" + "="*60)
    print("Baseline Model Download Verification Report")
    print("="*60)

    for model_name, result in results.items():
        print(f"\n{model_name}: {result.get('description', 'N/A')}")

        if result['valid']:
            print(f"  ✓ Downloaded: {result['size_gb']} GB")
            print(f"  ✓ Path: {result['path']}")
        else:
            print(f"  ✗ Error: {result.get('error', 'Unknown')}")

    print("\n" + "="*60)


if __name__ == "__main__":
    # Verify all baseline models
    baselines_dir = Path("/scratch/connectome/connectome1/ko-centaur/models/baselines")

    print("Checking baseline model downloads...")
    results = verify_all_baselines(baselines_dir)
    print_verification_report(results)

    # Check prerequisites
    print("\nPrerequisite Checks:")
    print(f"  huggingface-cli installed: {check_huggingface_cli()}")
    print(f"  HuggingFace token configured: {check_hf_token()}")
