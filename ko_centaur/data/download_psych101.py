#!/usr/bin/env python3
"""Download and explore Psych-101 dataset"""

from datasets import load_dataset
import os

print("=" * 60)
print("Downloading Psych-101 Dataset")
print("=" * 60)

# Set cache directory
cache_dir = os.environ.get("HF_HOME", "/scratch/connectome/connectome1/ko-centaur/cache")
data_dir = "/scratch/connectome/connectome1/ko-centaur/data"
os.makedirs(data_dir, exist_ok=True)

print(f"\nCache directory: {cache_dir}")
print(f"Data directory: {data_dir}")

try:
    print("\nLoading Psych-101 dataset from HuggingFace...")
    dataset = load_dataset("marcelbinz/Psych-101", cache_dir=cache_dir)
    
    print(f"\n✅ Dataset loaded successfully")
    print(f"\nDataset structure:")
    print(dataset)
    
    # Explore the dataset
    print(f"\n" + "=" * 60)
    print("Dataset Statistics")
    print("=" * 60)
    
    for split in dataset.keys():
        print(f"\n{split.upper()} split:")
        print(f"  Total samples: {len(dataset[split])}")
        print(f"  Features: {dataset[split].features}")
        
        # Show first example
        if len(dataset[split]) > 0:
            example = dataset[split][0]
            print(f"\n  First example:")
            for key, value in example.items():
                if isinstance(value, str):
                    preview = value[:200] + "..." if len(value) > 200 else value
                    print(f"    {key}: {preview}")
                else:
                    print(f"    {key}: {value}")
    
    # Save dataset to disk
    print(f"\n" + "=" * 60)
    print("Saving dataset to disk...")
    print("=" * 60)
    
    for split in dataset.keys():
        save_path = f"{data_dir}/psych101_{split}.jsonl"
        dataset[split].to_json(save_path)
        print(f"✅ Saved {split} to {save_path}")
    
    print(f"\n" + "=" * 60)
    print("✅ All operations completed successfully!")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)
