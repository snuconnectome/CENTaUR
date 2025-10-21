#!/usr/bin/env python3
"""
Analyze extracted features to find the root cause of identical predictions
"""
import torch
import numpy as np

print("=" * 60)
print("Feature Analysis")
print("=" * 60)

# Load features
ko_features = torch.load('results/choices13k_100/features/ko_centaur_features.pth', weights_only=False)
ex_features = torch.load('results/choices13k_100/features/exaone-base_features.pth', weights_only=False)

print(f"\nKo-CENTaUR Features:")
print(f"  Shape: {ko_features.shape}")
print(f"  Mean: {ko_features.mean():.6f}")
print(f"  Std: {ko_features.std():.6f}")
print(f"  Min: {ko_features.min():.6f}")
print(f"  Max: {ko_features.max():.6f}")

print(f"\nEXAONE-base Features:")
print(f"  Shape: {ex_features.shape}")
print(f"  Mean: {ex_features.mean():.6f}")
print(f"  Std: {ex_features.std():.6f}")
print(f"  Min: {ex_features.min():.6f}")
print(f"  Max: {ex_features.max():.6f}")

# Check if features are identical
feature_diff = (ko_features - ex_features).abs().max()
print(f"\n최대 Feature 차이: {feature_diff:.6f}")

if feature_diff < 1e-6:
    print("❌ Features가 거의 동일합니다!")
else:
    print("✓ Features가 다릅니다")

# Check if features have variance
ko_var = ko_features.var(dim=0).mean()
ex_var = ex_features.var(dim=0).mean()

print(f"\nFeature Variance (across samples):")
print(f"  Ko-CENTaUR 평균: {ko_var:.6f}")
print(f"  EXAONE-base 평균: {ex_var:.6f}")

# Check for constant features
ko_constant = (ko_features.std(dim=0) < 1e-6).sum()
ex_constant = (ex_features.std(dim=0) < 1e-6).sum()

print(f"\nConstant Features (std < 1e-6):")
print(f"  Ko-CENTaUR: {ko_constant}/4096")
print(f"  EXAONE-base: {ex_constant}/4096")

# Sample-wise analysis: are all samples identical?
ko_sample_std = ko_features.std(dim=1)
ex_sample_std = ex_features.std(dim=1)

print(f"\nSample-wise Feature Std:")
print(f"  Ko-CENTaUR (평균): {ko_sample_std.mean():.6f}")
print(f"  EXAONE-base (평균): {ex_sample_std.mean():.6f}")

# Check if all samples have identical features
ko_all_same = (ko_features - ko_features[0]).abs().max(dim=1)[0].max()
ex_all_same = (ex_features - ex_features[0]).abs().max(dim=1)[0].max()

print(f"\n모든 샘플이 동일한가?")
print(f"  Ko-CENTaUR max diff from first sample: {ko_all_same:.6f}")
print(f"  EXAONE-base max diff from first sample: {ex_all_same:.6f}")

if ko_all_same < 1e-3:
    print("  ❌ Ko-CENTaUR: 모든 샘플의 features가 거의 동일!")
if ex_all_same < 1e-3:
    print("  ❌ EXAONE-base: 모든 샘플의 features가 거의 동일!")

print("\n" + "=" * 60)
