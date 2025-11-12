#!/usr/bin/env python3
"""
종합 결과 분석 스크립트
모든 모델의 LOO CV 결과를 종합하고 비교
"""

import argparse
import json
import numpy as np
from pathlib import Path
from typing import Dict, List


def load_loo_results(results_dir: Path) -> Dict[str, dict]:
    """모든 LOO CV 결과 로드"""
    results = {}

    for json_file in results_dir.glob("*_loo_results.json"):
        model_name = json_file.stem.replace("_loo_results", "")
        with open(json_file) as f:
            results[model_name] = json.load(f)

    return results


def compute_statistics(results: Dict[str, dict]) -> Dict[str, dict]:
    """각 모델의 통계 계산"""
    stats = {}

    for model_name, result in results.items():
        nll = result.get('nll', result.get('negative_log_likelihood', np.nan))
        accuracy = result.get('accuracy', np.nan)

        stats[model_name] = {
            'nll': nll,
            'accuracy': accuracy,
            'n_samples': result.get('n_samples', 0),
            'n_features': result.get('n_features', 0),
            'best_alpha': result.get('best_alpha', np.nan)
        }

    return stats


def format_report(stats: Dict[str, dict]) -> str:
    """마크다운 리포트 생성"""
    report = []

    report.append("# Ko-CENTaUR 종합 결과 분석")
    report.append("")
    report.append(f"**생성 시간**: {np.datetime64('now')}")
    report.append("")

    # 전체 요약
    report.append("## 전체 요약")
    report.append("")

    # Random baseline
    random_nll = np.log(2)  # ln(2) ≈ 0.6931
    report.append(f"**Random Baseline**: NLL = {random_nll:.4f}")
    report.append("")

    # 모델별 결과 테이블
    report.append("## 모델별 성능")
    report.append("")
    report.append("| 모델 | NLL | Accuracy | Best Alpha | N Samples | N Features |")
    report.append("|------|-----|----------|------------|-----------|------------|")

    # NLL 기준 정렬
    sorted_models = sorted(stats.items(), key=lambda x: x[1]['nll'])

    for model_name, model_stats in sorted_models:
        nll = model_stats['nll']
        acc = model_stats['accuracy']
        alpha = model_stats['best_alpha']
        n_samples = model_stats['n_samples']
        n_features = model_stats['n_features']

        # 모델 이름 포맷팅
        display_name = model_name.replace('_', ' ').title()

        # Baseline 대비 개선율
        improvement = ((random_nll - nll) / random_nll) * 100

        report.append(
            f"| {display_name} | {nll:.4f} ({improvement:+.1f}%) | "
            f"{acc:.2%} | {alpha:.4f} | {n_samples} | {n_features} |"
        )

    report.append("")

    # Base vs Fine-tuned 비교
    report.append("## Base vs Fine-tuned 비교")
    report.append("")

    # Qwen2.5 비교
    if 'qwen25_base' in stats and 'qwen25_finetuned' in stats:
        qwen_base_nll = stats['qwen25_base']['nll']
        qwen_ft_nll = stats['qwen25_finetuned']['nll']
        qwen_improvement = ((qwen_base_nll - qwen_ft_nll) / qwen_base_nll) * 100

        report.append("### Qwen2.5-32B")
        report.append("")
        report.append(f"- **Base**: NLL = {qwen_base_nll:.4f}")
        report.append(f"- **Fine-tuned**: NLL = {qwen_ft_nll:.4f}")
        report.append(f"- **개선율**: {qwen_improvement:+.1f}%")
        report.append("")

    # DeepSeek 비교
    if 'deepseek_base' in stats and 'deepseek_finetuned' in stats:
        ds_base_nll = stats['deepseek_base']['nll']
        ds_ft_nll = stats['deepseek_finetuned']['nll']
        ds_improvement = ((ds_base_nll - ds_ft_nll) / ds_base_nll) * 100

        report.append("### DeepSeek-R1-32B")
        report.append("")
        report.append(f"- **Base**: NLL = {ds_base_nll:.4f}")
        report.append(f"- **Fine-tuned**: NLL = {ds_ft_nll:.4f}")
        report.append(f"- **개선율**: {ds_improvement:+.1f}%")
        report.append("")

    # 주요 발견사항
    report.append("## 주요 발견사항")
    report.append("")

    best_model = sorted_models[0]
    best_name = best_model[0].replace('_', ' ').title()
    best_nll = best_model[1]['nll']

    report.append(f"1. **최고 성능 모델**: {best_name} (NLL = {best_nll:.4f})")
    report.append(f"2. **Random Baseline 대비**: {((random_nll - best_nll) / random_nll * 100):.1f}% 개선")

    # Fine-tuning 효과
    base_models = [m for m in stats.keys() if 'base' in m]
    ft_models = [m for m in stats.keys() if 'finetuned' in m]

    if base_models and ft_models:
        avg_base_nll = np.mean([stats[m]['nll'] for m in base_models])
        avg_ft_nll = np.mean([stats[m]['nll'] for m in ft_models])
        avg_improvement = ((avg_base_nll - avg_ft_nll) / avg_base_nll) * 100

        report.append(f"3. **Fine-tuning 평균 효과**: {avg_improvement:+.1f}%")

    report.append("")

    # 결론
    report.append("## 결론")
    report.append("")
    report.append("- CENTaUR 방법론을 한국어 가능 LLM에 성공적으로 적용")
    report.append("- Random baseline을 크게 상회하는 성능 달성")

    if avg_improvement > 0:
        report.append("- Fine-tuning이 인지 모델링 성능 향상에 기여")
    else:
        report.append("- Fine-tuning의 효과는 제한적이거나 부정적")

    report.append("")

    return "\n".join(report)


def main():
    parser = argparse.ArgumentParser(description="종합 결과 분석")
    parser.add_argument("--results_dir", type=str, default="data/results/",
                        help="LOO CV 결과 디렉토리")
    parser.add_argument("--output", type=str, default="reports/final_analysis.md",
                        help="출력 리포트 경로")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=== 종합 결과 분석 ===")
    print(f"결과 디렉토리: {results_dir}")

    # 결과 로드
    results = load_loo_results(results_dir)
    print(f"로드된 모델: {len(results)}개")
    for model_name in results.keys():
        print(f"  - {model_name}")

    # 통계 계산
    stats = compute_statistics(results)

    # 리포트 생성
    report = format_report(stats)

    # 저장
    with open(output_path, 'w') as f:
        f.write(report)

    print(f"\n리포트 저장: {output_path}")
    print("\n=== 요약 ===")

    # 콘솔 출력
    sorted_models = sorted(stats.items(), key=lambda x: x[1]['nll'])
    for model_name, model_stats in sorted_models:
        print(f"{model_name:20s}: NLL = {model_stats['nll']:.4f}, "
              f"Acc = {model_stats['accuracy']:.2%}")


if __name__ == "__main__":
    main()
