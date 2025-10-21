#!/usr/bin/env python3
"""
Analyze model predictions to check for suspicious patterns
"""
import torch
import numpy as np
import sys

def analyze_predictions(results_path):
    # Load both model results
    results = torch.load(results_path, weights_only=False)

    kocentaur_preds = [fold['test_prediction'] for fold in results['ko-centaur']['fold_results']]
    exaone_preds = [fold['test_prediction'] for fold in results['exaone-base']['fold_results']]

    print("=" * 60)
    print("예측 비교 분석")
    print("=" * 60)

    # Check if predictions are identical
    identical_count = sum(1 for k, e in zip(kocentaur_preds, exaone_preds) if k == e)
    print(f"\n동일한 예측 수: {identical_count}/100")
    print(f"동일 비율: {identical_count/100*100:.1f}%")

    # Check prediction distribution
    ko_pred_0 = kocentaur_preds.count(0)
    ko_pred_1 = kocentaur_preds.count(1)
    ex_pred_0 = exaone_preds.count(0)
    ex_pred_1 = exaone_preds.count(1)

    print(f"\nKo-CENTaUR 예측 분포:")
    print(f"  Choice A (0): {ko_pred_0}/100 ({ko_pred_0}%)")
    print(f"  Choice B (1): {ko_pred_1}/100 ({ko_pred_1}%)")

    print(f"\nEXAONE-base 예측 분포:")
    print(f"  Choice A (0): {ex_pred_0}/100 ({ex_pred_0}%)")
    print(f"  Choice B (1): {ex_pred_1}/100 ({ex_pred_1}%)")

    # Check ground truth distribution
    ko_true = [fold['test_true_label'] for fold in results['ko-centaur']['fold_results']]
    true_0 = ko_true.count(0)
    true_1 = ko_true.count(1)

    print(f"\n실제 Ground Truth 분포:")
    print(f"  Choice A (0): {true_0}/100 ({true_0}%)")
    print(f"  Choice B (1): {true_1}/100 ({true_1}%)")

    # Critical check: Are models always predicting majority class?
    print(f"\n{'='*60}")
    print("⚠️  CRITICAL FINDING:")
    print(f"{'='*60}")

    if ko_pred_1 == 100 or ex_pred_1 == 100:
        print("❌ 모델이 ALWAYS Choice B를 예측하고 있습니다!")
        print("   → Majority class prediction (trivial baseline)")
    elif ko_pred_0 == 100 or ex_pred_0 == 100:
        print("❌ 모델이 ALWAYS Choice A를 예측하고 있습니다!")
        print("   → Minority class prediction")
    else:
        print("✓ 모델이 다양한 예측을 하고 있습니다")
        print(f"   Ko-CENTaUR: {ko_pred_0}A / {ko_pred_1}B")
        print(f"   EXAONE-base: {ex_pred_0}A / {ex_pred_1}B")

    if identical_count == 100:
        print("\n❌ 두 모델의 예측이 100% 동일합니다!")
        print("   → 모델들이 동일한 체크포인트를 사용하거나")
        print("   → Fine-tuning이 제대로 작동하지 않았을 가능성")

    print(f"{'='*60}")

    # Show first 10 predictions for inspection
    print("\n처음 10개 예측 샘플:")
    print(f"{'Fold':<6} {'True':<6} {'Ko-CENTaUR':<12} {'EXAONE':<8}")
    print("-" * 40)
    for i in range(min(10, len(kocentaur_preds))):
        true_label = ko_true[i]
        ko_pred = kocentaur_preds[i]
        ex_pred = exaone_preds[i]
        match_symbol = "✓" if ko_pred == true_label else "✗"
        print(f"{i:<6} {true_label:<6} {ko_pred:<12} {ex_pred:<8} {match_symbol}")

if __name__ == "__main__":
    results_path = sys.argv[1] if len(sys.argv) > 1 else "results/choices13k_100/all_results.pth"
    analyze_predictions(results_path)
