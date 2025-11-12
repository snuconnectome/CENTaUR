#!/usr/bin/env python3
# CENTaUR 실험 결과 리포트 생성

import json
import numpy as np
from pathlib import Path

def generate_report():
    print("=" * 60)
    print("CENTaUR dgx-spark 실험 결과 리포트")
    print("=" * 60)

    # Baseline NLL values
    random_nll = 120000
    llama65b_nll = 30000

    # Qwen2.5 results
    print("\n1. Qwen2.5-32B-Instruct Base Model")
    try:
        with open('outputs/qwen25_base_nll_results.json') as f:
            qwen_results = json.load(f)

        qwen_nll = qwen_results['mean_nll']
        qwen_std = qwen_results['std_nll']

        print(f"   Mean NLL: {qwen_nll:.2f} ± {qwen_std:.2f}")
        print(f"   vs Random: {((random_nll - qwen_nll) / random_nll * 100):.1f}% improvement")
        print(f"   vs LLaMA-65B: {((llama65b_nll - qwen_nll) / llama65b_nll * 100):.1f}% {'improvement' if qwen_nll < llama65b_nll else 'degradation'}")
    except FileNotFoundError:
        print("   ⚠️  결과 파일 없음")
        qwen_nll = None

    # DeepSeek results
    print("\n2. DeepSeek-R1-Distill-Qwen-32B Base Model")
    try:
        with open('outputs/deepseek_base_nll_results.json') as f:
            deepseek_results = json.load(f)

        deepseek_nll = deepseek_results['mean_nll']
        deepseek_std = deepseek_results['std_nll']

        print(f"   Mean NLL: {deepseek_nll:.2f} ± {deepseek_std:.2f}")
        print(f"   vs Random: {((random_nll - deepseek_nll) / random_nll * 100):.1f}% improvement")
        print(f"   vs LLaMA-65B: {((llama65b_nll - deepseek_nll) / llama65b_nll * 100):.1f}% {'improvement' if deepseek_nll < llama65b_nll else 'degradation'}")
    except FileNotFoundError:
        print("   ⚠️  결과 파일 없음")
        deepseek_nll = None

    # Model comparison
    if qwen_nll and deepseek_nll:
        print("\n3. 모델 비교")
        better_model = "Qwen2.5" if qwen_nll < deepseek_nll else "DeepSeek"
        diff = abs(qwen_nll - deepseek_nll)
        print(f"   Best model: {better_model}")
        print(f"   NLL difference: {diff:.2f}")

    # Feature extraction files
    print("\n4. 생성된 Feature 파일")
    for feature_file in Path('outputs').glob('*_features.npz'):
        data = np.load(feature_file)
        print(f"   {feature_file.name}:")
        print(f"      Features: {data['features'].shape}")
        print(f"      Labels: {data['labels'].shape}")
        print(f"      File size: {feature_file.stat().st_size / 1024 / 1024:.2f} MB")

    print("\n" + "=" * 60)
    print("실험 완료")
    print("=" * 60)

if __name__ == "__main__":
    generate_report()
