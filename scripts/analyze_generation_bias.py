#!/usr/bin/env python3
"""
Generation Bias 분석 스크립트
모델이 특정 선택지를 선호하는 경향을 분석
"""

import argparse
import json
import numpy as np
import torch
from pathlib import Path
from typing import Dict, Tuple
from collections import Counter


def load_features(features_path: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Feature 파일 로드"""
    data = torch.load(features_path, map_location='cpu')

    if isinstance(data, dict):
        features = data.get('features', data.get('hidden_states'))
        labels = data.get('labels', data.get('actions'))
    else:
        # NPZ 파일인 경우
        features = torch.from_numpy(data['features'])
        labels = torch.from_numpy(data['labels'])

    return features, labels


def compute_choice_distribution(labels: torch.Tensor) -> Dict[str, float]:
    """선택 분포 계산"""
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    unique, counts = np.unique(labels_np, return_counts=True)

    total = len(labels_np)
    distribution = {
        f'choice_{int(choice)}': count / total
        for choice, count in zip(unique, counts)
    }

    return distribution


def compute_generation_bias_score(labels: torch.Tensor) -> float:
    """Generation bias 점수 계산 (0 = 완벽한 균형, 1 = 완전한 편향)"""
    labels_np = labels.numpy() if isinstance(labels, torch.Tensor) else labels
    counts = Counter(labels_np)

    # 가장 흔한 선택의 비율
    most_common_ratio = max(counts.values()) / len(labels_np)

    # Bias score: 0.5 (완벽한 균형)에서 얼마나 벗어났는지
    bias_score = abs(most_common_ratio - 0.5) * 2  # 0~1 범위로 정규화

    return bias_score


def analyze_model(features_path: Path, model_name: str) -> Dict:
    """개별 모델 분석"""
    try:
        features, labels = load_features(features_path)

        distribution = compute_choice_distribution(labels)
        bias_score = compute_generation_bias_score(labels)

        return {
            'model_name': model_name,
            'n_samples': len(labels),
            'choice_distribution': distribution,
            'bias_score': bias_score,
            'most_frequent_choice': int(labels.mode()[0].item()),
            'status': 'success'
        }
    except Exception as e:
        return {
            'model_name': model_name,
            'status': 'error',
            'error': str(e)
        }


def format_bias_report(results: Dict[str, dict]) -> str:
    """Generation bias 리포트 생성"""
    report = []

    report.append("# Generation Bias 분석")
    report.append("")
    report.append("모델이 특정 선택지를 선호하는 경향 분석")
    report.append("")

    # 전체 요약
    report.append("## 전체 요약")
    report.append("")

    successful_results = [r for r in results.values() if r['status'] == 'success']

    if not successful_results:
        report.append("❌ 분석 가능한 결과가 없습니다.")
        return "\n".join(report)

    # Bias 점수 테이블
    report.append("| 모델 | Bias Score | Choice 0 | Choice 1 | 선호 선택 | N Samples |")
    report.append("|------|------------|----------|----------|-----------|-----------|")

    sorted_results = sorted(successful_results, key=lambda x: x['bias_score'])

    for result in sorted_results:
        model_name = result['model_name'].replace('_', ' ').title()
        bias_score = result['bias_score']
        dist = result['choice_distribution']

        choice_0_pct = dist.get('choice_0', 0.0) * 100
        choice_1_pct = dist.get('choice_1', 0.0) * 100
        most_freq = result['most_frequent_choice']
        n_samples = result['n_samples']

        report.append(
            f"| {model_name} | {bias_score:.4f} | "
            f"{choice_0_pct:.1f}% | {choice_1_pct:.1f}% | "
            f"Choice {most_freq} | {n_samples} |"
        )

    report.append("")

    # Bias 해석 가이드
    report.append("## Bias Score 해석")
    report.append("")
    report.append("- **0.00 - 0.10**: 매우 균형적 (Low bias)")
    report.append("- **0.10 - 0.30**: 약간 편향 (Moderate bias)")
    report.append("- **0.30 - 0.50**: 중간 편향 (High bias)")
    report.append("- **0.50 - 1.00**: 강한 편향 (Very high bias)")
    report.append("")

    # Base vs Fine-tuned 비교
    report.append("## Base vs Fine-tuned 비교")
    report.append("")

    base_results = {k: v for k, v in results.items() if 'base' in k and v['status'] == 'success'}
    ft_results = {k: v for k, v in results.items() if 'finetuned' in k and v['status'] == 'success'}

    if base_results:
        avg_base_bias = np.mean([r['bias_score'] for r in base_results.values()])
        report.append(f"**Base 모델 평균 Bias**: {avg_base_bias:.4f}")

    if ft_results:
        avg_ft_bias = np.mean([r['bias_score'] for r in ft_results.values()])
        report.append(f"**Fine-tuned 모델 평균 Bias**: {avg_ft_bias:.4f}")

    if base_results and ft_results:
        bias_change = avg_ft_bias - avg_base_bias
        report.append(f"**Fine-tuning으로 인한 변화**: {bias_change:+.4f}")

        if bias_change > 0.05:
            report.append("  → Fine-tuning이 편향을 증가시킴")
        elif bias_change < -0.05:
            report.append("  → Fine-tuning이 편향을 감소시킨")
        else:
            report.append("  → Fine-tuning의 편향 영향 미미")

    report.append("")

    # 주요 발견사항
    report.append("## 주요 발견사항")
    report.append("")

    least_biased = sorted_results[0]
    most_biased = sorted_results[-1]

    report.append(f"1. **가장 균형적인 모델**: {least_biased['model_name']} (Bias = {least_biased['bias_score']:.4f})")
    report.append(f"2. **가장 편향된 모델**: {most_biased['model_name']} (Bias = {most_biased['bias_score']:.4f})")

    # 전체 평균
    avg_bias = np.mean([r['bias_score'] for r in successful_results])
    report.append(f"3. **전체 평균 Bias**: {avg_bias:.4f}")

    report.append("")

    # 권장사항
    report.append("## 권장사항")
    report.append("")

    if avg_bias > 0.3:
        report.append("- ⚠️ Generation bias가 높음 → 프롬프트 엔지니어링 필요")
        report.append("- 고려사항: 질문 표현 방식, 선택지 순서, 맥락 정보")
    else:
        report.append("- ✅ Generation bias가 적정 수준")
        report.append("- 현재 프롬프트 방식이 효과적")

    report.append("")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="Generation Bias 분석")
    parser.add_argument("--results_dir", type=str, default="data/results/",
                        help="결과 디렉토리 (JSON 파일 참조용)")
    parser.add_argument("--features_dir", type=str, default="data/features/",
                        help="Feature 디렉토리")
    parser.add_argument("--output", type=str, default="reports/generation_bias_analysis.md",
                        help="출력 리포트 경로")
    args = parser.parse_args()

    features_dir = Path(args.features_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== Generation Bias 분석 ===")
    print(f"Feature 디렉토리: {features_dir}")

    # 모든 feature 파일 분석
    results = {}

    for features_path in features_dir.glob("*_features.pth"):
        model_name = features_path.stem.replace("_features", "")
        print(f"분석 중: {model_name}...")

        result = analyze_model(features_path, model_name)
        results[model_name] = result

        if result['status'] == 'success':
            print(f"  Bias Score: {result['bias_score']:.4f}")
        else:
            print(f"  ❌ 에러: {result['error']}")

    # 리포트 생성
    report = format_bias_report(results)

    # 저장
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"\n리포트 저장: {output_path}")

    # 콘솔 요약
    print("\n=== 요약 ===")
    successful = [r for r in results.values() if r['status'] == 'success']
    if successful:
        for result in sorted(successful, key=lambda x: x['bias_score']):
            print(f"{result['model_name']:20s}: Bias = {result['bias_score']:.4f}")


if __name__ == "__main__":
    main()
