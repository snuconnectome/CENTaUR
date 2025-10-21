"""
Automated baseline model download with validation

TDD Implementation: Passes tests in test_download_verification.py
"""
import subprocess
import shutil
from pathlib import Path
from typing import Tuple
import sys


# Expected model sizes (GB)
EXPECTED_SIZES = {
    "llama-centaur-70b": 140.0,
    "llama-70b-instruct": 140.0,
    "solar-10.7b": 20.0
}

# HuggingFace repository IDs
REPO_IDS = {
    "llama-centaur-70b": "marcelbinz/Llama-3.1-Centaur-70B",
    "llama-70b-instruct": "meta-llama/Llama-3.1-70B-Instruct",
    "solar-10.7b": "upstage/SOLAR-10.7B-Instruct-v1.0"
}

# Models requiring authentication
AUTH_REQUIRED = {
    "llama-70b-instruct": "https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct"
}


def ensure_baselines_dir() -> Path:
    """Create baselines directory if it doesn't exist"""
    baselines_dir = Path("/scratch/connectome/connectome1/ko-centaur/models/baselines")
    baselines_dir.mkdir(parents=True, exist_ok=True)
    return baselines_dir


def check_disk_space(required_gb: float) -> bool:
    """
    Check if sufficient disk space is available

    Args:
        required_gb: Required space in GB

    Returns:
        True if sufficient space available
    """
    baselines_dir = ensure_baselines_dir()
    usage = shutil.disk_usage(baselines_dir)

    free_gb = usage.free / (1024 ** 3)

    print(f"\nDisk Space Check:")
    print(f"  Required: {required_gb} GB")
    print(f"  Available: {free_gb:.1f} GB")

    if free_gb < required_gb:
        print(f"  ✗ Insufficient disk space!")
        return False

    print(f"  ✓ Sufficient disk space")
    return True


def check_huggingface_cli() -> bool:
    """Check if huggingface-cli is installed"""
    try:
        result = subprocess.run(
            ['huggingface-cli', '--version'],
            capture_output=True,
            text=True,
            timeout=5
        )
        installed = result.returncode == 0

        if installed:
            print("✓ huggingface-cli is installed")
        else:
            print("✗ huggingface-cli not found")

        return installed

    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("✗ huggingface-cli not found")
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
        has_token = result.returncode == 0

        if has_token:
            print("✓ HuggingFace token configured")
        else:
            print("✗ HuggingFace token not configured")

        return has_token

    except (FileNotFoundError, subprocess.TimeoutExpired):
        print("✗ Cannot check HuggingFace token")
        return False


def install_huggingface_hub():
    """Install huggingface-hub package"""
    print("\nInstalling huggingface-hub...")

    try:
        subprocess.run(
            [sys.executable, '-m', 'pip', 'install', 'huggingface-hub'],
            check=True
        )
        print("✓ huggingface-hub installed")
        return True

    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install huggingface-hub: {e}")
        return False


def download_model(model_name: str, repo_id: str, local_dir: Path) -> bool:
    """
    Download a model using huggingface-cli

    Args:
        model_name: Model identifier
        repo_id: HuggingFace repository ID
        local_dir: Local download directory

    Returns:
        True if download successful
    """
    print(f"\n{'='*60}")
    print(f"Downloading: {model_name}")
    print(f"Repository: {repo_id}")
    print(f"Destination: {local_dir}")
    print(f"{'='*60}\n")

    # Check if already downloaded
    if local_dir.exists():
        from .download_verification import verify_model_download, get_download_progress

        # Check if complete
        progress = get_download_progress(local_dir, EXPECTED_SIZES[model_name])
        if progress['complete']:
            print(f"✓ {model_name} already downloaded ({progress['downloaded_gb']} GB)")
            return True
        else:
            print(f"  Partial download found: {progress['percent']}%")
            print(f"  Resuming download...")

    # Download command
    cmd = [
        'huggingface-cli', 'download', repo_id,
        '--local-dir', str(local_dir),
        '--local-dir-use-symlinks', 'False',
        '--resume-download'
    ]

    try:
        subprocess.run(cmd, check=True)
        print(f"\n✓ Downloaded {model_name}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"\n✗ Failed to download {model_name}: {e}")
        return False


def verify_prerequisites() -> Tuple[bool, List[str]]:
    """
    Verify all prerequisites are met

    Returns:
        (all_ok, list_of_issues)
    """
    issues = []

    print("\n" + "="*60)
    print("Verifying Prerequisites")
    print("="*60 + "\n")

    # Check disk space (400GB for all models)
    if not check_disk_space(400):
        issues.append("Insufficient disk space")

    # Check huggingface-cli
    if not check_huggingface_cli():
        issues.append("huggingface-cli not installed")

    # Check HF token (needed for some models)
    has_token = check_hf_token()

    all_ok = len(issues) == 0

    if not all_ok:
        print("\n" + "="*60)
        print("Prerequisites Check Failed")
        print("="*60)
        for issue in issues:
            print(f"  ✗ {issue}")

    return all_ok, issues


def download_all_models(models: List[str] = None):
    """
    Download all baseline models

    Args:
        models: List of model names to download (None = all)
    """
    baselines_dir = ensure_baselines_dir()

    # Default to all models
    if models is None:
        models = list(REPO_IDS.keys())

    # Verify prerequisites
    all_ok, issues = verify_prerequisites()

    if not all_ok:
        print("\nCannot proceed with downloads. Please fix issues above.")

        # Offer to install huggingface-hub
        if "huggingface-cli not installed" in issues:
            response = input("\nInstall huggingface-hub now? (y/n): ")
            if response.lower() == 'y':
                if install_huggingface_hub():
                    print("Please re-run this script.")
                return

        return

    # Warn about authentication for protected models
    protected_models = [m for m in models if m in AUTH_REQUIRED]
    if protected_models:
        print("\n" + "="*60)
        print("Authentication Required")
        print("="*60)
        for model in protected_models:
            print(f"\n{model} requires license acceptance:")
            print(f"  1. Visit: {AUTH_REQUIRED[model]}")
            print(f"  2. Click 'Agree and access repository'")
            print(f"  3. Generate token: https://huggingface.co/settings/tokens")
            print(f"  4. Run: huggingface-cli login --token YOUR_TOKEN")

        response = input("\nHave you completed authentication? (y/n): ")
        if response.lower() != 'y':
            print("Please complete authentication first.")
            return

    # Download each model
    results = {}
    for model_name in models:
        repo_id = REPO_IDS[model_name]
        local_dir = baselines_dir / model_name

        success = download_model(model_name, repo_id, local_dir)
        results[model_name] = success

    # Summary
    print("\n" + "="*60)
    print("Download Summary")
    print("="*60)

    successful = [m for m, s in results.items() if s]
    failed = [m for m, s in results.items() if not s]

    print(f"\n✓ Successful: {len(successful)}/{len(models)}")
    for model in successful:
        print(f"  - {model}")

    if failed:
        print(f"\n✗ Failed: {len(failed)}/{len(models)}")
        for model in failed:
            print(f"  - {model}")

    # Verification
    if successful:
        print("\nRunning verification...")
        from .download_verification import verify_all_baselines, print_verification_report

        results = verify_all_baselines(baselines_dir)
        print_verification_report(results)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Download baseline models')
    parser.add_argument(
        '--models',
        nargs='+',
        choices=list(REPO_IDS.keys()),
        help='Specific models to download (default: all)'
    )

    args = parser.parse_args()

    download_all_models(args.models)
