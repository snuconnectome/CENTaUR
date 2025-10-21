#!/usr/bin/env python3
"""
Quick Evaluation Script for Ko-CENTaUR

Runs 50-sample rapid evaluation to validate pipeline before full evaluation.
Configured for connectome server execution.

Usage:
    python scripts/run_quick_eval.py \
        --checkpoint /scratch/connectome/connectome1/ko-centaur/models/ko_centaur_checkpoint \
        --dataset /scratch/connectome/connectome1/ko-centaur/data/raw/psych101_train.jsonl \
        --baseline exaone-base \
        --n_samples 50 \
        --output_dir /scratch/connectome/connectome1/ko-centaur/results/quick_eval
"""

import argparse
import json
import sys
from pathlib import Path
import torch
from datetime import datetime

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from baselines.load_baselines import BaselineModelManager
from evaluation.extract_features import extract_features_batch
from evaluation.quick_eval import (
    run_quick_eval_pipeline,
    rapid_comparison,
    generate_comparison_report
)


def load_jsonl_dataset(path: str, n_samples: int = None):
    """Load JSONL dataset"""
    samples = []
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if n_samples and i >= n_samples:
                break
            samples.append(json.loads(line))
    return samples


def main():
    parser = argparse.ArgumentParser(description='Ko-CENTaUR Quick Evaluation')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to Ko-CENTaUR checkpoint directory'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        required=True,
        help='Path to Psych-101 JSONL dataset'
    )
    parser.add_argument(
        '--baseline',
        type=str,
        default='exaone-base',
        help='Baseline model name (default: exaone-base)'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=50,
        help='Number of samples for quick evaluation (default: 50)'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='/scratch/connectome/connectome1/ko-centaur/results/quick_eval',
        help='Output directory for results'
    )
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        help='Device to use (cuda or cpu)'
    )

    args = parser.parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("="*70)
    print("Ko-CENTaUR Quick Evaluation")
    print("="*70)
    print(f"Start time: {datetime.now()}")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Dataset: {args.dataset}")
    print(f"Baseline: {args.baseline}")
    print(f"Samples: {args.n_samples}")
    print(f"Output: {output_dir}")
    print()

    # Load dataset
    print(f"[1/5] Loading dataset...")
    try:
        dataset = load_jsonl_dataset(args.dataset, args.n_samples)
        print(f"✓ Loaded {len(dataset)} samples")

        # Check for labels
        if 'label' in dataset[0]:
            labels = torch.tensor([s['label'] for s in dataset])
        elif 'choice' in dataset[0]:
            labels = torch.tensor([s['choice'] for s in dataset])
        else:
            print("⚠ No labels found in dataset, using zeros")
            labels = torch.zeros(len(dataset), dtype=torch.long)

    except Exception as e:
        print(f"✗ Error loading dataset: {e}")
        return 1

    # Load Ko-CENTaUR
    print(f"\n[2/5] Loading Ko-CENTaUR...")
    try:
        kocentaur = BaselineModelManager("ko-centaur")
        print(f"✓ Ko-CENTaUR loaded")
    except Exception as e:
        print(f"✗ Error loading Ko-CENTaUR: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Load baseline
    print(f"\n[3/5] Loading baseline: {args.baseline}...")
    try:
        baseline = BaselineModelManager(args.baseline)
        print(f"✓ Baseline loaded")
    except Exception as e:
        print(f"✗ Error loading baseline: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Extract features
    print(f"\n[4/5] Extracting features...")
    try:
        print(f"  Ko-CENTaUR features...")
        kocentaur_features = extract_features_batch(kocentaur, dataset)
        print(f"  ✓ Ko-CENTaUR: {kocentaur_features.shape}")

        print(f"  Baseline features...")
        baseline_features = extract_features_batch(baseline, dataset)
        print(f"  ✓ Baseline: {baseline_features.shape}")
    except Exception as e:
        print(f"✗ Error extracting features: {e}")
        import traceback
        traceback.print_exc()
        return 1

    # Run comparison
    print(f"\n[5/5] Running rapid comparison...")
    try:
        result = rapid_comparison(
            features_a=kocentaur_features,
            features_b=baseline_features,
            labels=labels,
            model_names=("ko-centaur", args.baseline)
        )

        report = generate_comparison_report(result)

        print(f"\n{'='*70}")
        print("Results")
        print("="*70)
        print(f"Ko-CENTaUR accuracy: {result['ko-centaur']['accuracy']:.3f}")
        print(f"{args.baseline} accuracy: {result[args.baseline]['accuracy']:.3f}")
        print(f"Winner: {report['winner']}")
        print(f"Improvement: {report['improvement']:.3f}")
        print()

        # Save results (convert all numeric values to Python floats for JSON serialization)
        results_path = output_dir / f"quick_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        # Build JSON-serializable output (exclude tensors from predictions)
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'checkpoint': args.checkpoint,
            'dataset': args.dataset,
            'n_samples': args.n_samples,
            'results': {
                'ko-centaur': {
                    'accuracy': float(result['ko-centaur']['accuracy']),
                    'log_likelihood': float(result['ko-centaur']['log_likelihood'])
                },
                args.baseline: {
                    'accuracy': float(result[args.baseline]['accuracy']),
                    'log_likelihood': float(result[args.baseline]['log_likelihood'])
                }
            },
            'report': {
                'winner': report['winner'],
                'improvement': float(report['improvement']),
                'winner_accuracy': float(report['winner_accuracy']),
                'loser_accuracy': float(report['loser_accuracy'])
            }
        }

        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)

        print(f"✓ Results saved to: {results_path}")
        print()
        print("="*70)
        print(f"Quick evaluation complete!")
        print(f"End time: {datetime.now()}")
        print("="*70)

        return 0

    except Exception as e:
        print(f"✗ Error during comparison: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
