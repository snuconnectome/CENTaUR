#!/usr/bin/env python3
"""
Create stratified 1000-sample evaluation dataset from choices13k

Implements stratified random sampling by problem difficulty to ensure:
- Representative distribution across difficulty levels
- 45-55% class balance (Choice A vs B)
- Quality validation and comprehensive reporting

Usage:
    python scripts/create_choices13k_1000.py \
        --problems /tmp/choices13k/c13k_problems.json \
        --selections /tmp/choices13k/c13k_selections.csv \
        --output data/choices13k_1000.jsonl \
        --seed 42
"""

import argparse
import json
import csv
import random
import numpy as np
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple


def format_gamble(outcomes: List[List[float]]) -> str:
    """
    Format gamble outcomes as natural language

    Args:
        outcomes: List of [probability, payout] pairs

    Returns:
        String like "95% chance of $26, 5% chance of $-1"
    """
    parts = []
    for prob, payout in outcomes:
        prob_pct = int(prob * 100)
        if payout >= 0:
            parts.append(f"{prob_pct}% chance of ${int(payout)}")
        else:
            parts.append(f"{prob_pct}% chance of $-{int(abs(payout))}")

    return ", ".join(parts)


def calculate_expected_value(outcomes: List[List[float]]) -> float:
    """Calculate expected value of a gamble"""
    return sum(prob * payout for prob, payout in outcomes)


def calculate_difficulty(problem_data: Dict) -> float:
    """
    Calculate problem difficulty as absolute difference in expected values

    Larger difference = easier decision (one option clearly dominates)
    Smaller difference = harder decision (options are close)

    Args:
        problem_data: Dict with "A" and "B" keys containing outcome lists

    Returns:
        Absolute difference in expected values
    """
    ev_a = calculate_expected_value(problem_data["A"])
    ev_b = calculate_expected_value(problem_data["B"])
    return abs(ev_a - ev_b)


def create_prompt(problem_data: Dict) -> str:
    """Create risky choice prompt from problem data"""
    option_a = format_gamble(problem_data["A"])
    option_b = format_gamble(problem_data["B"])

    prompt = (
        f"Which option would you choose? "
        f"Option A: {option_a}. "
        f"Option B: {option_b}. "
        f"Machine chose:"
    )

    return prompt


def stratified_sample(
    problems: Dict,
    selections: Dict,
    n_samples: int = 1000,
    n_strata: int = 4,
    seed: int = 42
) -> List[Tuple[str, Dict, Dict]]:
    """
    Perform stratified random sampling by problem difficulty

    Args:
        problems: Dict mapping problem_id to problem data
        selections: Dict mapping problem_id to selection data
        n_samples: Target number of samples (default: 1000)
        n_strata: Number of difficulty strata (default: 4 quartiles)
        seed: Random seed for reproducibility

    Returns:
        List of (problem_id, problem_data, selection_data) tuples
    """
    random.seed(seed)
    np.random.seed(seed)

    # Calculate difficulty for each problem
    print(f"\n[1/4] Calculating difficulty metrics...")
    problem_difficulties = []
    for problem_id in selections.keys():
        if problem_id not in problems:
            continue
        difficulty = calculate_difficulty(problems[problem_id])
        problem_difficulties.append((problem_id, difficulty))

    print(f"  ✓ Computed difficulty for {len(problem_difficulties)} problems")

    # Sort by difficulty
    problem_difficulties.sort(key=lambda x: x[1])

    # Create difficulty quartiles
    print(f"\n[2/4] Creating {n_strata} difficulty strata...")
    difficulties = [d for _, d in problem_difficulties]
    quartile_boundaries = np.percentile(difficulties, np.linspace(0, 100, n_strata + 1))

    print(f"  Difficulty boundaries: {[f'${b:.2f}' for b in quartile_boundaries]}")

    # Assign problems to strata
    strata = [[] for _ in range(n_strata)]
    for problem_id, difficulty in problem_difficulties:
        # Determine which stratum this problem belongs to
        stratum_idx = min(
            n_strata - 1,
            int((difficulty - quartile_boundaries[0]) /
                (quartile_boundaries[-1] - quartile_boundaries[0] + 1e-10) * n_strata)
        )
        strata[stratum_idx].append(problem_id)

    for i, stratum in enumerate(strata):
        print(f"  Stratum {i+1}: {len(stratum)} problems")

    # Sample from each stratum
    print(f"\n[3/4] Sampling {n_samples} problems (stratified)...")
    samples_per_stratum = n_samples // n_strata
    sampled_ids = []

    for i, stratum in enumerate(strata):
        # Sample without replacement
        stratum_sample = random.sample(stratum, min(samples_per_stratum, len(stratum)))
        sampled_ids.extend(stratum_sample)
        print(f"  Stratum {i+1}: sampled {len(stratum_sample)}/{len(stratum)}")

    # If we're short, sample remaining from all strata
    if len(sampled_ids) < n_samples:
        remaining = n_samples - len(sampled_ids)
        all_unsampled = [pid for stratum in strata for pid in stratum if pid not in sampled_ids]
        additional = random.sample(all_unsampled, min(remaining, len(all_unsampled)))
        sampled_ids.extend(additional)
        print(f"  Additional: sampled {len(additional)} to reach {n_samples}")

    # Create final sample list with all data
    samples = []
    for problem_id in sampled_ids:
        samples.append((
            problem_id,
            problems[problem_id],
            selections[problem_id]
        ))

    return samples


def validate_sample(samples: List[Tuple], target_balance: Tuple[float, float] = (0.45, 0.55)):
    """
    Validate sample quality

    Args:
        samples: List of (problem_id, problem_data, selection_data) tuples
        target_balance: (min, max) class balance range

    Returns:
        Dict with validation metrics
    """
    print(f"\n[4/4] Validating sample quality...")

    # Count choices
    choices = [selection_data['choice'] for _, _, selection_data in samples]
    choice_counts = Counter(choices)

    choice_0_pct = choice_counts[0] / len(samples)
    choice_1_pct = choice_counts[1] / len(samples)

    # Check balance
    balance_ok = (
        target_balance[0] <= choice_0_pct <= target_balance[1] and
        target_balance[0] <= choice_1_pct <= target_balance[1]
    )

    # Calculate difficulty distribution
    difficulties = [
        calculate_difficulty(problem_data)
        for _, problem_data, _ in samples
    ]

    metrics = {
        'n_samples': len(samples),
        'choice_0_count': choice_counts[0],
        'choice_1_count': choice_counts[1],
        'choice_0_pct': choice_0_pct * 100,
        'choice_1_pct': choice_1_pct * 100,
        'balance_ok': balance_ok,
        'difficulty_mean': np.mean(difficulties),
        'difficulty_std': np.std(difficulties),
        'difficulty_min': np.min(difficulties),
        'difficulty_25': np.percentile(difficulties, 25),
        'difficulty_50': np.percentile(difficulties, 50),
        'difficulty_75': np.percentile(difficulties, 75),
        'difficulty_max': np.max(difficulties)
    }

    # Print validation report
    print(f"\n{'='*60}")
    print(f"SAMPLE VALIDATION REPORT")
    print(f"{'='*60}")
    print(f"\nSample Size:")
    print(f"  Total: {metrics['n_samples']} problems")

    print(f"\nClass Balance:")
    print(f"  Choice A (0): {metrics['choice_0_count']} ({metrics['choice_0_pct']:.1f}%)")
    print(f"  Choice B (1): {metrics['choice_1_count']} ({metrics['choice_1_pct']:.1f}%)")
    print(f"  Target range: {target_balance[0]*100:.0f}%-{target_balance[1]*100:.0f}%")

    if metrics['balance_ok']:
        print(f"  ✅ PASS: Class balance within target range")
    else:
        print(f"  ❌ FAIL: Class balance outside target range")

    print(f"\nDifficulty Distribution:")
    print(f"  Mean: ${metrics['difficulty_mean']:.2f}")
    print(f"  Std:  ${metrics['difficulty_std']:.2f}")
    print(f"  Min:  ${metrics['difficulty_min']:.2f}")
    print(f"  Q1:   ${metrics['difficulty_25']:.2f}")
    print(f"  Q2:   ${metrics['difficulty_50']:.2f}")
    print(f"  Q3:   ${metrics['difficulty_75']:.2f}")
    print(f"  Max:  ${metrics['difficulty_max']:.2f}")

    print(f"\n{'='*60}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description='Create stratified 1000-sample dataset')
    parser.add_argument(
        '--problems',
        type=str,
        required=True,
        help='Path to c13k_problems.json'
    )
    parser.add_argument(
        '--selections',
        type=str,
        required=True,
        help='Path to c13k_selections.csv'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Output JSONL file path'
    )
    parser.add_argument(
        '--n_samples',
        type=int,
        default=1000,
        help='Target number of samples (default: 1000)'
    )
    parser.add_argument(
        '--n_strata',
        type=int,
        default=4,
        help='Number of difficulty strata (default: 4 quartiles)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )

    args = parser.parse_args()

    print(f"{'='*60}")
    print(f"CREATE CHOICES13K 1000-SAMPLE DATASET")
    print(f"{'='*60}")
    print(f"\nParameters:")
    print(f"  Target samples: {args.n_samples}")
    print(f"  Difficulty strata: {args.n_strata}")
    print(f"  Random seed: {args.seed}")
    print(f"  Output: {args.output}")

    # Load problems
    print(f"\nLoading problems from {args.problems}")
    with open(args.problems, 'r') as f:
        problems = json.load(f)
    print(f"✓ Loaded {len(problems)} problems")

    # Load selections
    print(f"\nLoading selections from {args.selections}")
    selections = {}
    with open(args.selections, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            problem_id = row['Problem']
            feedback = row['Feedback'] == 'True'
            brate = float(row['bRate'])

            # Use majority human choice (bRate > 0.5 means B chosen more)
            choice = 1 if brate > 0.5 else 0

            # Store unique problems only
            if problem_id not in selections:
                selections[problem_id] = {
                    'choice': choice,
                    'brate': brate,
                    'feedback': feedback
                }

    print(f"✓ Loaded {len(selections)} unique problems with selection data")

    # Perform stratified sampling
    samples = stratified_sample(
        problems=problems,
        selections=selections,
        n_samples=args.n_samples,
        n_strata=args.n_strata,
        seed=args.seed
    )

    # Validate sample quality
    metrics = validate_sample(samples)

    # Write to JSONL
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nWriting samples to {output_path}...")
    with open(output_path, 'w', encoding='utf-8') as f:
        for problem_id, problem_data, selection_data in samples:
            prompt = create_prompt(problem_data)
            entry = {
                'text': prompt,
                'choice': selection_data['choice']
            }
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"✓ Wrote {len(samples)} samples to {output_path}")

    # Save validation metrics
    metrics_path = output_path.with_suffix('.metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"✓ Saved validation metrics to {metrics_path}")

    print(f"\n{'='*60}")
    print(f"COMPLETE")
    print(f"{'='*60}")

    if not metrics['balance_ok']:
        print(f"\n⚠️  WARNING: Class balance outside target range!")
        print(f"   Consider adjusting sampling strategy or seed.")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
