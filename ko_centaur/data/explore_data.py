#!/usr/bin/env python3
"""Explore Psych-101 dataset structure in detail"""

import json
import pandas as pd
from collections import Counter

print("=" * 60)
print("Psych-101 Dataset Exploration")
print("=" * 60)

data_path = "/scratch/connectome/connectome1/ko-centaur/data/psych101_train.jsonl"

# Load data
print(f"\nLoading data from {data_path}...")
data = []
with open(data_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))

print(f"✅ Loaded {len(data)} samples")

# Basic statistics
print(f"\n" + "=" * 60)
print("Basic Statistics")
print("=" * 60)

experiments = [d['experiment'] for d in data]
participants = [d['participant'] for d in data]
text_lengths = [len(d['text']) for d in data]

print(f"\nUnique experiments: {len(set(experiments))}")
print(f"Unique participants: {len(set(participants))}")
print(f"\nText length statistics:")
print(f"  Min: {min(text_lengths)} chars")
print(f"  Max: {max(text_lengths)} chars")
print(f"  Mean: {sum(text_lengths)/len(text_lengths):.1f} chars")
print(f"  Median: {sorted(text_lengths)[len(text_lengths)//2]} chars")

# Top experiments
print(f"\n" + "=" * 60)
print("Top 10 Experiments (by sample count)")
print("=" * 60)

exp_counts = Counter(experiments)
for exp, count in exp_counts.most_common(10):
    print(f"  {exp}: {count} samples")

# Show examples from different experiments
print(f"\n" + "=" * 60)
print("Sample Examples")
print("=" * 60)

shown_exps = set()
for sample in data[:100]:
    exp = sample['experiment']
    if exp not in shown_exps and len(shown_exps) < 3:
        shown_exps.add(exp)
        text_preview = sample['text'][:300] + "..." if len(sample['text']) > 300 else sample['text']
        print(f"\n[Experiment: {exp}]")
        print(f"[Participant: {sample['participant']}]")
        print(f"Text:\n{text_preview}")
        print("-" * 60)

print(f"\n" + "=" * 60)
print("✅ Exploration complete!")
print("=" * 60)
