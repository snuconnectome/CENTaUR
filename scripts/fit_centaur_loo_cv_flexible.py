#!/usr/bin/env python3
"""
100-fold Leave-One-Out Cross-Validation with Binomial Regression
Flexible version accepting custom paths
"""

import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json
import sys

# Add CENTaUR root to path for models.py import
sys.path.insert(0, str(Path(__file__).parent.parent))
from models import BinomialRegression

# GPU setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🔧 Using device: {device}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA capability: {torch.cuda.get_device_capability(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")


def nested_cv_select_alpha(X_train, y_train, alpha_grid, n_folds=11, seed=42):
    """Nested cross-validation to select optimal alpha (L2 regularization)"""
    n = len(X_train)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    fold_size = n // n_folds
    
    alpha_scores = {alpha: [] for alpha in alpha_grid}
    
    for fold_idx in range(n_folds):
        val_start = fold_idx * fold_size
        val_end = val_start + fold_size if fold_idx < n_folds - 1 else n
        
        val_mask = torch.zeros(n, dtype=torch.bool)
        val_mask[indices[val_start:val_end]] = True
        train_mask = ~val_mask
        
        X_inner_train = X_train[train_mask]
        y_inner_train = y_train[train_mask]
        X_val = X_train[val_mask]
        y_val = y_train[val_mask]
        
        for alpha in alpha_grid:
            model = BinomialRegression(num_inputs=X_train.shape[1], alpha=alpha).to(device)
            # Convert binary labels to binomial format
            num_choices = torch.ones(len(X_inner_train), dtype=torch.long).to(device)
            num_B_choices = y_inner_train.long().to(device)
            model.fit(X_inner_train, num_choices, num_B_choices, num_iterations=1000)
            
            with torch.no_grad():
                logits = model(X_val)
                probs_B = torch.sigmoid(logits)
                val_nll = 0.0
                for i in range(len(y_val)):
                    if y_val[i] == 1:
                        val_nll -= torch.log(probs_B[i] + 1e-10)
                    else:
                        val_nll -= torch.log(1 - probs_B[i] + 1e-10)
                val_nll = val_nll.item()
            
            alpha_scores[alpha].append(val_nll)
    
    avg_scores = {alpha: np.mean(scores) for alpha, scores in alpha_scores.items()}
    best_alpha = min(avg_scores, key=avg_scores.get)
    
    return best_alpha, alpha_scores


def run_loo_cv(features_path, output_path, model_name="Model"):
    """Run 100-fold LOO CV with binomial regression"""
    
    print(f"\n{'='*60}")
    print(f"100-Fold LOO CV: {model_name}")
    print(f"{'='*60}\n")
    
    # Load features
    if features_path.endswith('.npz'):
        # FIXED: .npz files are PyTorch format (torch.save), not NumPy
        data = torch.load(features_path)
        # Convert to float32 (features may be bfloat16)
        features = data['features'].float() if isinstance(data['features'], torch.Tensor) else torch.FloatTensor(data['features'])
        labels = data['labels'].float() if isinstance(data['labels'], torch.Tensor) else torch.FloatTensor(data['labels'])
    elif features_path.endswith('.pth'):
        data = torch.load(features_path)
        features = data['features']
        labels = data['labels']
    else:
        raise ValueError(f"Unsupported file format: {features_path}")
    
    # Move to GPU
    features = features.to(device)
    labels = labels.to(device)
    
    print(f"📊 Dataset: {len(features)} samples, {features.shape[1]}-dim features")
    print(f"📂 Features: {features_path}")
    print(f"📝 Output: {output_path}")
    print(f"🔧 Device: {device}\n")
    
    # Alpha grid for L2 regularization
    alpha_grid = [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
    
    # 100-fold LOO CV
    n_folds = 100
    n_samples = len(features)
    fold_size = n_samples // n_folds
    
    print(f"🔄 Running {n_folds}-fold LOO CV...")
    print(f"   Fold size: {fold_size} samples")
    print(f"   Alpha grid: {alpha_grid}\n")
    
    results = []
    test_nlls = []
    test_probs = []
    test_labels = []
    best_alphas = []
    
    for fold_idx in tqdm(range(n_folds), desc="LOO CV"):
        # Split data
        test_start = fold_idx * fold_size
        test_end = test_start + fold_size if fold_idx < n_folds - 1 else n_samples
        
        test_mask = torch.zeros(n_samples, dtype=torch.bool)
        test_mask[test_start:test_end] = True
        train_mask = ~test_mask
        
        X_train = features[train_mask]
        y_train = labels[train_mask]
        X_test = features[test_mask]
        y_test = labels[test_mask]
        
        # Select alpha via nested CV
        best_alpha, _ = nested_cv_select_alpha(X_train, y_train, alpha_grid)
        best_alphas.append(best_alpha)
        
        # Train with best alpha
        model = BinomialRegression(num_inputs=features.shape[1], alpha=best_alpha).to(device)
        # Convert binary labels to binomial format
        num_choices_train = torch.ones(len(X_train), dtype=torch.long).to(device)
        num_B_choices_train = y_train.long().to(device)
        model.fit(X_train, num_choices_train, num_B_choices_train, num_iterations=1000)
        
        # Evaluate on test set
        with torch.no_grad():
            logits = model(X_test)
            probs_B = torch.sigmoid(logits)
            # Calculate NLL manually
            nll = 0.0
            for i in range(len(y_test)):
                if y_test[i] == 1:
                    nll -= torch.log(probs_B[i] + 1e-10).item()
                else:
                    nll -= torch.log(1 - probs_B[i] + 1e-10).item()
        
        test_nlls.append(nll)
        test_probs.append(probs_B.cpu().numpy())
        test_labels.append(y_test.cpu().numpy())
        
        results.append({
            'fold': fold_idx,
            'test_nll': nll,
            'best_alpha': best_alpha,
            'test_size': len(y_test)
        })
    
    # Aggregate results
    avg_nll = np.mean(test_nlls)
    std_nll = np.std(test_nlls)
    total_nll = np.sum(test_nlls)
    
    # Accuracy
    all_probs = np.concatenate(test_probs)
    all_labels = np.concatenate(test_labels)
    predictions = (all_probs > 0.5).astype(float)
    accuracy = np.mean(predictions == all_labels)
    
    # Alpha statistics
    alpha_counts = {alpha: best_alphas.count(alpha) for alpha in alpha_grid}
    
    print(f"\n{'='*60}")
    print(f"📊 RESULTS: {model_name}")
    print(f"{'='*60}")
    print(f"Average NLL per fold: {avg_nll:.4f} ± {std_nll:.4f}")
    print(f"Total NLL: {total_nll:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"\nAlpha selection frequency:")
    for alpha, count in sorted(alpha_counts.items()):
        print(f"  α={alpha:>8.4f}: {count:>3} folds")
    print(f"{'='*60}\n")
    
    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    if output_path.suffix == '.json':
        # Save as JSON
        json_results = {
            'model_name': model_name,
            'mean_nll': float(avg_nll),
            'std_nll': float(std_nll),
            'total_nll': float(total_nll),
            'accuracy': float(accuracy),
            'total_samples': int(n_samples),
            'n_folds': n_folds,
            'alpha_grid': alpha_grid,
            'alpha_counts': {str(k): v for k, v in alpha_counts.items()},
            'fold_results': results
        }
        with open(output_path, 'w') as f:
            json.dump(json_results, f, indent=2)
    else:
        # Save as PyTorch
        torch.save({
            'model_name': model_name,
            'n_samples': len(features),
            'hidden_dim': features.shape[1],
            'alpha_grid': alpha_grid,
            'results': results,
            'test_nlls': test_nlls,
            'test_probs': test_probs,
            'test_labels': test_labels,
            'best_alphas': best_alphas,
            'avg_nll': avg_nll,
            'std_nll': std_nll,
            'total_nll': total_nll,
            'accuracy': accuracy,
            'alpha_counts': alpha_counts
        }, output_path)
    
    print(f"✅ Results saved to: {output_path}\n")
    
    return {
        'avg_nll': avg_nll,
        'std_nll': std_nll,
        'total_nll': total_nll,
        'accuracy': accuracy
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="100-fold LOO CV with Binomial Regression for CENTaUR evaluation"
    )
    parser.add_argument(
        "features_path",
        type=str,
        help="Path to features file (.npz or .pth)"
    )
    parser.add_argument(
        "output_path",
        type=str,
        help="Path to save results (.json or .pth)"
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="Model",
        help="Model name for reporting"
    )
    args = parser.parse_args()
    
    # Run evaluation
    run_loo_cv(args.features_path, args.output_path, args.model_name)
