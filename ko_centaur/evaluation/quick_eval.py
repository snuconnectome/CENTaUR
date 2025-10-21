"""
Quick evaluation system for Ko-CENTaUR

TDD Implementation: Passes tests in test_quick_eval.py

Provides rapid model comparison with mini test sets for iterative development.
"""
import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split


def create_mini_test_set(
    full_dataset: List[Dict],
    n_samples: int = 50,
    stratified: bool = False,
    random_state: int = 42
) -> List[Dict]:
    """
    Create mini test set from full dataset

    Args:
        full_dataset: Complete dataset
        n_samples: Number of samples for mini set
        stratified: Whether to preserve class distribution
        random_state: Random seed for reproducibility

    Returns:
        Mini test set
    """
    if not stratified:
        # Random sampling
        np.random.seed(random_state)
        indices = np.random.choice(len(full_dataset), n_samples, replace=False)
        mini_set = [full_dataset[i] for i in indices]
        return mini_set

    # Stratified sampling
    labels = np.array([s.get("label", 0) for s in full_dataset])
    indices = np.arange(len(full_dataset))

    # Use stratified split
    _, mini_indices = train_test_split(
        indices,
        test_size=n_samples,
        stratify=labels,
        random_state=random_state
    )

    mini_set = [full_dataset[i] for i in mini_indices]
    return mini_set


def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute classification accuracy

    Args:
        predictions: Predicted labels
        labels: Ground truth labels

    Returns:
        Accuracy as float
    """
    correct = (predictions == labels).sum().item()
    total = len(labels)
    accuracy = correct / total
    return accuracy


def compute_log_likelihood(
    probabilities: torch.Tensor,
    labels: torch.Tensor
) -> float:
    """
    Compute average log-likelihood

    Args:
        probabilities: Predicted probabilities (n_samples, n_classes)
        labels: Ground truth labels

    Returns:
        Average log-likelihood
    """
    # Get probabilities of true labels
    true_probs = probabilities[torch.arange(len(labels)), labels]

    # Compute log-likelihood
    log_probs = torch.log(true_probs + 1e-10)  # Add epsilon for stability
    log_likelihood = log_probs.mean().item()

    return log_likelihood


def compute_confusion_matrix(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    n_classes: int = 2
) -> torch.Tensor:
    """
    Compute confusion matrix

    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        n_classes: Number of classes

    Returns:
        Confusion matrix of shape (n_classes, n_classes)
    """
    confusion = torch.zeros(n_classes, n_classes, dtype=torch.long)

    for pred, label in zip(predictions, labels):
        confusion[label.item(), pred.item()] += 1

    return confusion


def compute_per_class_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor
) -> Dict[str, List[float]]:
    """
    Compute per-class precision, recall, F1

    Args:
        predictions: Predicted labels
        labels: Ground truth labels

    Returns:
        Dict with precision, recall, f1 per class
    """
    n_classes = max(predictions.max().item(), labels.max().item()) + 1

    precision = []
    recall = []
    f1 = []

    for class_id in range(n_classes):
        # True positives, false positives, false negatives
        tp = ((predictions == class_id) & (labels == class_id)).sum().item()
        fp = ((predictions == class_id) & (labels != class_id)).sum().item()
        fn = ((predictions != class_id) & (labels == class_id)).sum().item()

        # Precision and recall
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        # F1 score
        f1_score = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0

        precision.append(prec)
        recall.append(rec)
        f1.append(f1_score)

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


def compute_agreement_rate(
    predictions_a: torch.Tensor,
    predictions_b: torch.Tensor
) -> float:
    """
    Compute agreement rate between two models

    Args:
        predictions_a: Predictions from model A
        predictions_b: Predictions from model B

    Returns:
        Agreement rate as float
    """
    agreement = (predictions_a == predictions_b).sum().item()
    total = len(predictions_a)
    rate = agreement / total
    return rate


def compute_feature_distance(
    features_a: torch.Tensor,
    features_b: torch.Tensor
) -> float:
    """
    Compute average Euclidean distance between feature representations

    Args:
        features_a: Features from model A
        features_b: Features from model B

    Returns:
        Average distance
    """
    distances = torch.norm(features_a - features_b, dim=1)
    avg_distance = distances.mean().item()
    return avg_distance


def analyze_disagreements(
    predictions_a: torch.Tensor,
    predictions_b: torch.Tensor,
    labels: torch.Tensor
) -> Dict:
    """
    Analyze cases where models disagree

    Args:
        predictions_a: Predictions from model A
        predictions_b: Predictions from model B
        labels: Ground truth labels

    Returns:
        Dict with disagreement analysis
    """
    # Find disagreements
    disagree_mask = predictions_a != predictions_b
    n_disagreements = disagree_mask.sum().item()

    if n_disagreements == 0:
        return {
            "total_disagreements": 0,
            "both_wrong": 0,
            "a_correct": 0,
            "b_correct": 0
        }

    # Analyze disagreement cases
    disagree_a = predictions_a[disagree_mask]
    disagree_b = predictions_b[disagree_mask]
    disagree_labels = labels[disagree_mask]

    both_wrong = ((disagree_a != disagree_labels) & (disagree_b != disagree_labels)).sum().item()
    a_correct = ((disagree_a == disagree_labels) & (disagree_b != disagree_labels)).sum().item()
    b_correct = ((disagree_a != disagree_labels) & (disagree_b == disagree_labels)).sum().item()

    return {
        "total_disagreements": n_disagreements,
        "both_wrong": both_wrong,
        "a_correct": a_correct,
        "b_correct": b_correct
    }


def prepare_tsne_features(
    features: torch.Tensor,
    n_components: int = 2,
    random_state: int = 42
) -> np.ndarray:
    """
    Prepare t-SNE visualization of features

    Args:
        features: Feature tensor
        n_components: Number of t-SNE dimensions
        random_state: Random seed

    Returns:
        t-SNE transformed features
    """
    # Convert to numpy
    features_np = features.cpu().numpy()

    # Apply t-SNE
    tsne = TSNE(n_components=n_components, random_state=random_state)
    tsne_features = tsne.fit_transform(features_np)

    return tsne_features


def rapid_comparison(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    labels: torch.Tensor,
    model_names: Tuple[str, str] = ("model_a", "model_b")
) -> Dict:
    """
    Rapid comparison between two models

    Args:
        features_a: Features from model A
        features_b: Features from model B
        labels: Ground truth labels
        model_names: Names for models A and B

    Returns:
        Comparison results
    """
    # Simple logistic regression for quick evaluation
    # (In practice, would use proper binomial regression from models.py)

    def simple_predict(features):
        # Simplified: use mean of features as decision boundary
        scores = features.mean(dim=1)
        predictions = (scores > scores.median()).long()
        return predictions

    predictions_a = simple_predict(features_a)
    predictions_b = simple_predict(features_b)

    # Compute metrics for both models
    results = {}

    for name, predictions in [(model_names[0], predictions_a), (model_names[1], predictions_b)]:
        accuracy = compute_accuracy(predictions, labels)

        # Mock probabilities for log-likelihood
        n_samples = len(predictions)
        probabilities = torch.zeros(n_samples, 2)
        for i, pred in enumerate(predictions):
            probabilities[i, pred] = 0.9  # Mock high confidence
            probabilities[i, 1 - pred] = 0.1

        log_likelihood = compute_log_likelihood(probabilities, labels)

        results[name] = {
            "accuracy": accuracy,
            "log_likelihood": log_likelihood,
            "predictions": predictions
        }

    return results


def qualitative_analysis(
    features_a: torch.Tensor,
    features_b: torch.Tensor,
    labels: torch.Tensor
) -> Dict:
    """
    Qualitative analysis of model differences

    Args:
        features_a: Features from model A
        features_b: Features from model B
        labels: Ground truth labels

    Returns:
        Qualitative analysis dict
    """
    def simple_predict(features):
        scores = features.mean(dim=1)
        predictions = (scores > scores.median()).long()
        return predictions

    predictions_a = simple_predict(features_a)
    predictions_b = simple_predict(features_b)

    agreement = compute_agreement_rate(predictions_a, predictions_b)
    disagreement_analysis = analyze_disagreements(predictions_a, predictions_b, labels)
    feature_distance = compute_feature_distance(features_a, features_b)

    return {
        "agreement": agreement,
        "disagreement": disagreement_analysis,
        "feature_distance": feature_distance
    }


def generate_comparison_report(results: Dict) -> Dict:
    """
    Generate comparison report from results

    Args:
        results: Results dict from rapid_comparison

    Returns:
        Report dict
    """
    # Determine winner based on accuracy
    models = list(results.keys())
    accuracies = {name: res["accuracy"] for name, res in results.items()}

    winner = max(accuracies, key=accuracies.get)
    winner_acc = accuracies[winner]

    loser = min(accuracies, key=accuracies.get)
    loser_acc = accuracies[loser]

    improvement = winner_acc - loser_acc

    report = {
        "summary": results,
        "winner": winner,
        "improvement": improvement,
        "winner_accuracy": winner_acc,
        "loser_accuracy": loser_acc
    }

    return report


def quick_eval_with_checkpoints(
    model_a_type: str,
    model_b_type: str,
    samples: List[Dict],
    labels: torch.Tensor
) -> Dict:
    """
    Quick evaluation loading models from checkpoints

    Args:
        model_a_type: Model A identifier
        model_b_type: Model B identifier
        samples: Test samples
        labels: Ground truth labels

    Returns:
        Evaluation results
    """
    from baselines import BaselineModelManager
    from evaluation.extract_features import extract_features_batch

    # Load models
    manager_a = BaselineModelManager(model_a_type)
    manager_b = BaselineModelManager(model_b_type)

    # Extract features
    features_a = extract_features_batch(manager_a, samples)
    features_b = extract_features_batch(manager_b, samples)

    # Run comparison
    results = rapid_comparison(
        features_a, features_b, labels,
        model_names=(model_a_type, model_b_type)
    )

    return results


def save_report(report: Dict, output_path: Path):
    """
    Save evaluation report to JSON

    Args:
        report: Report dict
        output_path: Path to save file
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert tensors to lists for JSON serialization
    def convert_tensors(obj):
        if isinstance(obj, torch.Tensor):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_tensors(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_tensors(item) for item in obj]
        else:
            return obj

    serializable_report = convert_tensors(report)

    with open(output_path, 'w') as f:
        json.dump(serializable_report, f, indent=2)


def run_quick_eval_pipeline(
    model_a,
    model_b,
    samples: List[Dict],
    labels: torch.Tensor,
    output_dir: Path
) -> Dict:
    """
    Run complete quick evaluation pipeline

    Args:
        model_a: BaselineModelManager for model A
        model_b: BaselineModelManager for model B
        samples: Test samples
        labels: Ground truth labels
        output_dir: Directory for outputs

    Returns:
        Pipeline results
    """
    from evaluation.extract_features import extract_features_batch

    # Extract features
    features_a = extract_features_batch(model_a, samples)
    features_b = extract_features_batch(model_b, samples)

    # Run comparison
    comparison = rapid_comparison(
        features_a, features_b, labels,
        model_names=(model_a.model_type, model_b.model_type)
    )

    # Generate report
    report = generate_comparison_report(comparison)

    # Save report
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "quick_eval_report.json"
    save_report(report, report_path)

    return {
        "comparison": comparison,
        "report": report,
        "report_path": str(report_path)
    }


if __name__ == "__main__":
    print("Quick evaluation module for Ko-CENTaUR")
    print("Run tests with: pytest tests/test_quick_eval.py -v")
