#!/usr/bin/env python3
"""
Preprocess Psych-101 dataset for EXAONE fine-tuning
Converts raw text into instruction-following format
"""

import json
import os
from tqdm import tqdm

print("=" * 60)
print("Psych-101 Data Preprocessing for EXAONE")
print("=" * 60)

# Paths
data_dir = "/scratch/connectome/connectome1/ko-centaur/data"
input_path = f"{data_dir}/psych101_train.jsonl"
output_path = f"{data_dir}/psych101_exaone_train.jsonl"

# Load data
print(f"\nLoading data from {input_path}...")
data = []
with open(input_path, 'r') as f:
    for line in f:
        data.append(json.loads(line))

print(f"✅ Loaded {len(data)} samples")

# Convert to EXAONE chat format
print(f"\n" + "=" * 60)
print("Converting to EXAONE instruction format...")
print("=" * 60)

processed_data = []

for sample in tqdm(data):
    # Extract components
    text = sample['text']
    experiment = sample['experiment']
    participant = sample['participant']
    
    # Create instruction-following format
    # System: Define the task
    # User: Provide the experimental context
    # Assistant: Generate cognitive response
    
    instruction = {
        "messages": [
            {
                "role": "system",
                "content": "You are a cognitive model that predicts human behavior in psychological experiments. Given an experimental scenario, predict how a human participant would respond."
            },
            {
                "role": "user", 
                "content": text
            }
        ],
        "metadata": {
            "experiment": experiment,
            "participant": participant
        }
    }
    
    processed_data.append(instruction)

# Save processed data
print(f"\nSaving processed data to {output_path}...")
with open(output_path, 'w') as f:
    for item in processed_data:
        f.write(json.dumps(item) + '\n')

print(f"✅ Saved {len(processed_data)} processed samples")

# Show example
print(f"\n" + "=" * 60)
print("Example Processed Sample")
print("=" * 60)
print(json.dumps(processed_data[0], indent=2))

print(f"\n" + "=" * 60)
print("✅ Preprocessing complete!")
print("=" * 60)

# Statistics
total_chars = sum(len(json.dumps(item)) for item in processed_data)
avg_chars = total_chars / len(processed_data)
print(f"\nDataset statistics:")
print(f"  Total samples: {len(processed_data)}")
print(f"  Average sample size: {avg_chars:.1f} characters")
print(f"  Total dataset size: {total_chars / 1024 / 1024:.1f} MB")
