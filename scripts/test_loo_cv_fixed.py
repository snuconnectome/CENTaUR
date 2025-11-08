#!/usr/bin/env python3
"""
Quick LOO CV test on corrected features
Tests if features with 95% similarity are still predictive
"""

import sys
sys.path.append('/scratch/connectome/connectome1/ko-centaur')

from fit_centaur_loo_cv import run_loo_cv

# Test corrected features
features_path = "/scratch/connectome/connectome1/ko-centaur/data/features/centaur_features_qwen25_FIXED_FULL.pth"
output_path = "/scratch/connectome/connectome1/ko-centaur/data/results/loo_cv_results_qwen25_FIXED_test.pth"
model_name = "Qwen2.5-32B-QLoRA (FIXED extraction)"

print("\n" + "="*80)
print("TESTING CORRECTED FEATURE EXTRACTION")
print("="*80)
print(f"\nFeature file: {features_path}")
print(f"Average pairwise similarity: 95.25%")
print(f"\nGoal: Test if high-similarity features can still predict better than random")
print(f"      Random baseline NLL: 0.6931")
print(f"      Previous buggy NLL:  0.8560")
print(f"      Target: NLL < 0.69 (better than random)")
print("="*80 + "\n")

# Run LOO CV
results = run_loo_cv(features_path, output_path, model_name)

# Print comparison
print("\n" + "="*80)
print("COMPARISON WITH BASELINES")
print("="*80)
print(f"Random baseline:          NLL = 0.6931")
print(f"Previous buggy features:  NLL = 0.8560")
print(f"Corrected features:       NLL = {results['avg_nll']:.4f}")
print("\nConclusion:")
if results['avg_nll'] < 0.69:
    print("  ✅ CORRECTED features BEAT random baseline!")
    print("  The high similarity is acceptable - features are informative")
elif results['avg_nll'] < 0.86:
    print("  ⚠️  Better than buggy features but worse than random")
    print("  Methodology fix helped but model needs investigation")
else:
    print("  ❌ Still worse than buggy features")
    print("  Further investigation needed")
print("="*80 + "\n")
