#!/usr/bin/env python3
"""
결과 시각화 스크립트
LOO CV 결과를 그래프로 시각화
"""

import argparse
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List


# 한글 폰트 설정 (선택적)
try:
    plt.rcParams['font.family'] = 'NanumGothic'
except:
    # 한글 폰트가 없으면 기본 폰트 사용
    pass

plt.rcParams['axes.unicode_minus'] = False  # 마이너스 기호 깨짐 방지


def load_results(results_dir: Path) -> Dict[str, dict]:
    """모든 결과 로드"""
    results = {}

    for json_file in results_dir.glob("*_loo_results.json"):
        model_name = json_file.stem.replace("_loo_results", "")
        with open(json_file) as f:
            results[model_name] = json.load(f)

    return results


def plot_nll_comparison(results: Dict[str, dict], output_dir: Path):
    """NLL 비교 막대 그래프"""
    # 데이터 준비
    models = []
    nlls = []
    colors = []

    color_map = {
        'base': '#3498db',      # 파란색
        'finetuned': '#e74c3c'  # 빨간색
    }

    sorted_items = sorted(results.items(), key=lambda x: x[1].get('nll', np.inf))

    for model_name, result in sorted_items:
        nll = result.get('nll', result.get('negative_log_likelihood', np.nan))

        display_name = model_name.replace('_', '\n')
        models.append(display_name)
        nlls.append(nll)

        # 색상 결정
        if 'finetuned' in model_name:
            colors.append(color_map['finetuned'])
        else:
            colors.append(color_map['base'])

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(range(len(models)), nlls, color=colors, alpha=0.8, edgecolor='black')

    # Random baseline 라인
    random_nll = np.log(2)
    ax.axhline(y=random_nll, color='gray', linestyle='--', linewidth=2,
               label=f'Random Baseline (NLL={random_nll:.4f})')

    # 레이블
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Negative Log-Likelihood (NLL)', fontsize=12, fontweight='bold')
    ax.set_title('Ko-CENTaUR Model Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=10)

    # 그리드
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    ax.legend(fontsize=10)

    # 값 표시
    for i, (bar, nll) in enumerate(zip(bars, nlls)):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{nll:.4f}',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'nll_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ NLL 비교 그래프 저장: {output_dir / 'nll_comparison.png'}")


def plot_base_vs_finetuned(results: Dict[str, dict], output_dir: Path):
    """Base vs Fine-tuned 비교 그래프"""
    # 데이터 준비
    comparisons = []

    # Qwen2.5
    if 'qwen25_base' in results and 'qwen25_finetuned' in results:
        comparisons.append({
            'model': 'Qwen2.5-32B',
            'base_nll': results['qwen25_base'].get('nll', np.nan),
            'finetuned_nll': results['qwen25_finetuned'].get('nll', np.nan)
        })

    # DeepSeek
    if 'deepseek_base' in results and 'deepseek_finetuned' in results:
        comparisons.append({
            'model': 'DeepSeek-R1-32B',
            'base_nll': results['deepseek_base'].get('nll', np.nan),
            'finetuned_nll': results['deepseek_finetuned'].get('nll', np.nan)
        })

    if not comparisons:
        print("⚠️ Base vs Fine-tuned 비교 데이터 없음")
        return

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(comparisons))
    width = 0.35

    base_nlls = [c['base_nll'] for c in comparisons]
    ft_nlls = [c['finetuned_nll'] for c in comparisons]

    bars1 = ax.bar(x - width/2, base_nlls, width, label='Base',
                   color='#3498db', alpha=0.8, edgecolor='black')
    bars2 = ax.bar(x + width/2, ft_nlls, width, label='Fine-tuned',
                   color='#e74c3c', alpha=0.8, edgecolor='black')

    # 레이블
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Negative Log-Likelihood (NLL)', fontsize=12, fontweight='bold')
    ax.set_title('Base vs Fine-tuned Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([c['model'] for c in comparisons], fontsize=11)
    ax.legend(fontsize=11)

    # 그리드
    ax.grid(axis='y', alpha=0.3, linestyle=':')

    # 값 표시
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{height:.4f}',
                    ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'base_vs_finetuned.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ Base vs Fine-tuned 그래프 저장: {output_dir / 'base_vs_finetuned.png'}")


def plot_improvement_percentage(results: Dict[str, dict], output_dir: Path):
    """개선율 그래프"""
    # 데이터 준비
    random_nll = np.log(2)

    models = []
    improvements = []
    colors = []

    sorted_items = sorted(results.items(), key=lambda x: x[1].get('nll', np.inf))

    for model_name, result in sorted_items:
        nll = result.get('nll', result.get('negative_log_likelihood', np.nan))

        if np.isnan(nll):
            continue

        improvement = ((random_nll - nll) / random_nll) * 100

        display_name = model_name.replace('_', '\n')
        models.append(display_name)
        improvements.append(improvement)

        # 색상: 개선이면 녹색, 악화면 빨간색
        if improvement > 0:
            colors.append('#2ecc71')  # 녹색
        else:
            colors.append('#e74c3c')  # 빨간색

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(range(len(models)), improvements, color=colors, alpha=0.8, edgecolor='black')

    # Zero 라인
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    # 레이블
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Improvement over Random Baseline (%)', fontsize=12, fontweight='bold')
    ax.set_title('Performance Improvement', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=10)

    # 그리드
    ax.grid(axis='y', alpha=0.3, linestyle=':')

    # 값 표시
    for bar, improvement in zip(bars, improvements):
        height = bar.get_height()
        va = 'bottom' if height > 0 else 'top'
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{improvement:+.1f}%',
                ha='center', va=va, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'improvement_percentage.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 개선율 그래프 저장: {output_dir / 'improvement_percentage.png'}")


def plot_accuracy_comparison(results: Dict[str, dict], output_dir: Path):
    """정확도 비교 그래프"""
    # 데이터 준비
    models = []
    accuracies = []

    sorted_items = sorted(results.items(),
                          key=lambda x: x[1].get('accuracy', 0),
                          reverse=True)

    for model_name, result in sorted_items:
        acc = result.get('accuracy', np.nan)

        if np.isnan(acc):
            continue

        display_name = model_name.replace('_', '\n')
        models.append(display_name)
        accuracies.append(acc * 100)  # 퍼센트로 변환

    if not accuracies:
        print("⚠️ 정확도 데이터 없음")
        return

    # 그래프 생성
    fig, ax = plt.subplots(figsize=(10, 6))

    bars = ax.bar(range(len(models)), accuracies,
                  color='#9b59b6', alpha=0.8, edgecolor='black')

    # Random baseline (50%)
    ax.axhline(y=50, color='gray', linestyle='--', linewidth=2,
               label='Random Baseline (50%)')

    # 레이블
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Accuracy (%)', fontsize=12, fontweight='bold')
    ax.set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, fontsize=10)
    ax.set_ylim([0, 100])

    # 그리드
    ax.grid(axis='y', alpha=0.3, linestyle=':')
    ax.legend(fontsize=10)

    # 값 표시
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width() / 2., height,
                f'{acc:.1f}%',
                ha='center', va='bottom', fontsize=9)

    plt.tight_layout()
    plt.savefig(output_dir / 'accuracy_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

    print(f"✅ 정확도 비교 그래프 저장: {output_dir / 'accuracy_comparison.png'}")


def main():
    parser = argparse.ArgumentParser(description="결과 시각화")
    parser.add_argument("--results_dir", type=str, default="data/results/",
                        help="LOO CV 결과 디렉토리")
    parser.add_argument("--output_dir", type=str, default="figures/",
                        help="출력 디렉토리")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=== 결과 시각화 ===")
    print(f"결과 디렉토리: {results_dir}")
    print(f"출력 디렉토리: {output_dir}")

    # 결과 로드
    results = load_results(results_dir)
    print(f"로드된 모델: {len(results)}개")

    if not results:
        print("❌ 결과 파일이 없습니다.")
        return

    # 그래프 생성
    print("\n그래프 생성 중...")

    plot_nll_comparison(results, output_dir)
    plot_base_vs_finetuned(results, output_dir)
    plot_improvement_percentage(results, output_dir)
    plot_accuracy_comparison(results, output_dir)

    print("\n✅ 모든 그래프 생성 완료")
    print(f"출력 위치: {output_dir}")


if __name__ == "__main__":
    main()
