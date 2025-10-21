#!/usr/bin/env python3
"""
Convert choices13k dataset to Ko-CENTaUR evaluation format

Converts probability-payout gamble descriptions to JSONL format:
{"text": "prompt", "choice": 0/1}

Usage:
    python scripts/convert_choices13k.py \
        --problems /tmp/choices13k/c13k_problems.json \
        --selections /tmp/choices13k/c13k_selections.csv \
        --output data/choices13k.jsonl \
        --n_samples 100
"""

import argparse
import json
import csv
from pathlib import Path


def format_gamble(outcomes):
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


def create_prompt(problem_data):
    """
    Create risky choice prompt from problem data

    Args:
        problem_data: Dict with "A" and "B" keys containing outcome lists

    Returns:
        Formatted prompt string
    """
    option_a = format_gamble(problem_data["A"])
    option_b = format_gamble(problem_data["B"])

    prompt = (
        f"Which option would you choose? "
        f"Option A: {option_a}. "
        f"Option B: {option_b}. "
        f"Machine chose:"
    )

    return prompt


def main():
    parser = argparse.ArgumentParser(description='Convert choices13k to JSONL')
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
        default=None,
        help='Number of samples to convert (default: all)'
    )
    parser.add_argument(
        '--feedback_only',
        action='store_true',
        help='Only include problems with feedback (Feedback=True)'
    )

    args = parser.parse_args()

    # Load problems
    print(f"Loading problems from {args.problems}")
    with open(args.problems, 'r') as f:
        problems = json.load(f)

    print(f"✓ Loaded {len(problems)} problems")

    # Load selections
    print(f"Loading selections from {args.selections}")
    selections = {}
    with open(args.selections, 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            problem_id = row['Problem']
            feedback = row['Feedback'] == 'True'
            brate = float(row['bRate'])

            # Skip if feedback_only and no feedback
            if args.feedback_only and not feedback:
                continue

            # Use majority human choice (bRate > 0.5 means B chosen more)
            choice = 1 if brate > 0.5 else 0

            # Store unique problems only (some problems appear multiple times)
            if problem_id not in selections:
                selections[problem_id] = {
                    'choice': choice,
                    'brate': brate,
                    'feedback': feedback
                }

    print(f"✓ Loaded {len(selections)} unique problems with selection data")

    # Create JSONL entries
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nConverting to JSONL format...")
    converted_count = 0
    choice_0_count = 0
    choice_1_count = 0

    with open(output_path, 'w', encoding='utf-8') as f:
        for problem_id, selection_data in sorted(selections.items(), key=lambda x: int(x[0])):
            # Check if problem exists in problems dict
            if problem_id not in problems:
                continue

            # Create prompt
            prompt = create_prompt(problems[problem_id])

            # Create JSONL entry
            entry = {
                'text': prompt,
                'choice': selection_data['choice']
            }

            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

            converted_count += 1
            if selection_data['choice'] == 0:
                choice_0_count += 1
            else:
                choice_1_count += 1

            # Limit samples if requested
            if args.n_samples and converted_count >= args.n_samples:
                break

    print(f"✓ Converted {converted_count} problems")
    print(f"  Choice A (0): {choice_0_count} ({choice_0_count/converted_count*100:.1f}%)")
    print(f"  Choice B (1): {choice_1_count} ({choice_1_count/converted_count*100:.1f}%)")
    print(f"\nOutput saved to: {output_path}")


if __name__ == "__main__":
    main()
