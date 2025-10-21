"""
Feature extraction pipeline for Ko-CENTaUR evaluation

TDD Implementation: Passes tests in test_feature_extraction.py
"""
import torch
import json
from pathlib import Path
from typing import Dict, List, Tuple, Callable, Optional


def extract_features_for_sample(model_manager, sample: Dict) -> torch.Tensor:
    """
    Extract features for a single sample

    Args:
        model_manager: BaselineModelManager instance
        sample: Dict with either:
            - 'text' (choices13k format)
            - 'task_description' with optional 'system_prompt' (Psych-101 format)

    Returns:
        torch.Tensor: Features of shape (1, hidden_size)
    """
    # Build prompt from sample - support both dataset formats
    if 'text' in sample:
        # choices13k format: {'text': str, 'choice': int}
        prompt = sample['text']
    elif 'task_description' in sample:
        # Psych-101 format: {'task_description': str, 'label': int, 'system_prompt': str}
        task_desc = sample['task_description']
        system_prompt = sample.get('system_prompt', '')

        if system_prompt:
            prompt = f"{system_prompt}\n\n{task_desc}"
        else:
            prompt = task_desc
    else:
        raise ValueError(
            f"Sample must have either 'text' or 'task_description' field. "
            f"Found keys: {list(sample.keys())}"
        )

    # Format prompt using model-specific template
    formatted_prompt = model_manager.format_prompt(prompt)

    # Extract features
    features = model_manager.extract_features(formatted_prompt)

    return features


def extract_features_batch(model_manager, samples: List[Dict]) -> torch.Tensor:
    """
    Extract features for batch of samples

    Args:
        model_manager: BaselineModelManager instance
        samples: List of sample dicts

    Returns:
        torch.Tensor: Features of shape (num_samples, hidden_size)
    """
    feature_list = []

    for sample in samples:
        features = extract_features_for_sample(model_manager, sample)
        feature_list.append(features)

    # Stack all features
    batch_features = torch.cat(feature_list, dim=0)

    return batch_features


def save_features(features: torch.Tensor, metadata: Dict, output_path: Path):
    """
    Save features and metadata to .pth file

    Args:
        features: Feature tensor
        metadata: Metadata dict
        output_path: Path to save file
    """
    save_data = {
        'features': features,
        'metadata': metadata
    }

    # Create parent directory if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    torch.save(save_data, output_path)


def load_features(feature_path: Path) -> Tuple[torch.Tensor, Dict]:
    """
    Load features and metadata from .pth file

    Args:
        feature_path: Path to feature file

    Returns:
        Tuple of (features, metadata)
    """
    data = torch.load(feature_path)

    features = data['features']
    metadata = data['metadata']

    return features, metadata


def normalize_features(features: torch.Tensor) -> torch.Tensor:
    """
    Normalize features using z-score normalization

    Args:
        features: Feature tensor

    Returns:
        torch.Tensor: Normalized features with mean≈0, std≈1
    """
    mean = features.mean()
    std = features.std()

    normalized = (features - mean) / (std + 1e-8)

    return normalized


def extract_with_progress(
    model_manager,
    samples: List[Dict],
    progress_callback: Callable[[int, int], None]
) -> torch.Tensor:
    """
    Extract features with progress tracking

    Args:
        model_manager: BaselineModelManager instance
        samples: List of sample dicts
        progress_callback: Callback function(current, total)

    Returns:
        torch.Tensor: Extracted features
    """
    feature_list = []
    total = len(samples)

    for i, sample in enumerate(samples, 1):
        features = extract_features_for_sample(model_manager, sample)
        feature_list.append(features)

        # Call progress callback
        progress_callback(i, total)

    batch_features = torch.cat(feature_list, dim=0)

    return batch_features


def extract_features_cached(
    model_manager,
    samples: List[Dict],
    cache_path: Path
) -> Tuple[torch.Tensor, bool]:
    """
    Extract features with caching support

    Args:
        model_manager: BaselineModelManager instance
        samples: List of sample dicts
        cache_path: Path to cache file

    Returns:
        Tuple of (features, from_cache)
    """
    # Check if cache exists
    if cache_path.exists():
        # Load from cache
        features, _ = load_features(cache_path)
        return features, True

    # Cache miss - extract features
    features = extract_features_batch(model_manager, samples)

    # Save to cache
    metadata = {
        'model_type': getattr(model_manager, 'model_type', 'unknown'),
        'num_samples': len(samples)
    }
    save_features(features, metadata, cache_path)

    return features, False


def extract_parallel(
    model_managers: Dict,
    samples: List[Dict]
) -> Dict[str, torch.Tensor]:
    """
    Extract features in parallel across multiple models

    Args:
        model_managers: Dict mapping model_name -> BaselineModelManager
        samples: List of sample dicts

    Returns:
        Dict mapping model_name -> features
    """
    results = {}

    for model_name, manager in model_managers.items():
        features = extract_features_batch(manager, samples)
        results[model_name] = features

    return results


def validate_feature_dimensions(
    features_dict: Dict[str, torch.Tensor],
    expected_sizes: Optional[Dict[str, int]] = None
):
    """
    Validate feature dimensions across models

    Args:
        features_dict: Dict mapping model_name -> features
        expected_sizes: Optional dict of expected sizes per model

    Raises:
        ValueError: If dimensions are inconsistent with expectations
    """
    if expected_sizes is None:
        # No expectations - just accept any dimensions
        return

    for model_name, expected_size in expected_sizes.items():
        if model_name in features_dict:
            features = features_dict[model_name]
            actual_size = features.shape[1]

            if actual_size != expected_size:
                raise ValueError(
                    f"Model {model_name}: expected size {expected_size}, "
                    f"got {actual_size}"
                )


def load_dataset(dataset_path: Path) -> List[Dict]:
    """
    Load dataset from JSONL file

    Args:
        dataset_path: Path to .jsonl file

    Returns:
        List of sample dicts
    """
    samples = []

    with open(dataset_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                sample = json.loads(line)
                samples.append(sample)

    return samples


def validate_dataset(
    samples: List[Dict],
    require_labels: bool = False
):
    """
    Validate dataset schema

    Args:
        samples: List of sample dicts
        require_labels: Whether to require label field ('label' or 'choice')

    Raises:
        ValueError: If validation fails
    """
    for i, sample in enumerate(samples):
        # Check required fields - support both formats
        has_text = 'text' in sample
        has_task_desc = 'task_description' in sample

        if not has_text and not has_task_desc:
            raise ValueError(
                f"Sample {i}: must have either 'text' or 'task_description'. "
                f"Found keys: {list(sample.keys())}"
            )

        # Check labels if required
        if require_labels:
            has_label = 'label' in sample
            has_choice = 'choice' in sample

            if not has_label and not has_choice:
                raise ValueError(
                    f"Sample {i}: must have either 'label' or 'choice' field when labels required"
                )


def run_extraction_pipeline(
    model_manager,
    samples: List[Dict],
    output_dir: Path
) -> Tuple[torch.Tensor, Dict]:
    """
    Run complete extraction pipeline for one model

    Args:
        model_manager: BaselineModelManager instance
        samples: List of sample dicts
        output_dir: Directory to save features

    Returns:
        Tuple of (features, metadata)
    """
    # Extract features
    features = extract_features_batch(model_manager, samples)

    # Build metadata
    model_type = getattr(model_manager, 'model_type', 'unknown')
    metadata = {
        'model_type': model_type,
        'num_samples': len(samples),
        'hidden_size': features.shape[1]
    }

    # Save features
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{model_type}_features.pth"
    save_features(features, metadata, output_path)

    return features, metadata


def run_extraction_all_models(
    model_managers: Dict,
    samples: List[Dict],
    output_dir: Path
) -> Dict[str, Dict]:
    """
    Run extraction pipeline for all baseline models

    Args:
        model_managers: Dict mapping model_name -> BaselineModelManager
        samples: List of sample dicts
        output_dir: Directory to save features

    Returns:
        Dict mapping model_name -> {'features': tensor, 'metadata': dict}
    """
    results = {}

    for model_name, manager in model_managers.items():
        # Extract features for this model
        features, metadata = run_extraction_pipeline(
            model_manager=manager,
            samples=samples,
            output_dir=output_dir
        )

        results[model_name] = {
            'features': features,
            'metadata': metadata
        }

    return results


if __name__ == "__main__":
    print("Feature extraction pipeline module")
    print("Run tests with: pytest tests/test_feature_extraction.py -v")
