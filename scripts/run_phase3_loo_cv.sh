#!/bin/bash
# Phase 3: LOO Cross-Validation
# 모든 모델 (Base 5개 + Fine-tuned 2개)에 대해 100-fold LOO CV 실행

set -e

echo "Phase 3: LOO Cross-Validation 시작"
echo "시간: $(date)"
echo ""

mkdir -p data/results

# 모델 리스트 (Base 5개 + Fine-tuned 2개)
models=(
    "qwen25_base"
    "qwen25_finetuned"
    "deepseek_base"
    "deepseek_finetuned"
    "exaone35_base"
    "gpt_oss_20b_base"
    "kimi_k2_base"
)

# 각 모델에 대해 LOO CV 실행
for model in "${models[@]}"; do
    echo ""
    echo "=== $model LOO CV ==="

    if [ -f "data/results/${model}_loo_results.json" ]; then
        echo "✅ ${model} LOO CV 이미 완료"
        continue
    fi

    if [ ! -f "data/features/${model}_features.pth" ]; then
        echo "❌ ${model} features 없음, 건너뜀"
        continue
    fi

    echo "⏳ ${model} LOO CV 실행 중..."

    python scripts/fit_centaur_loo_cv.py \
        --features "data/features/${model}_features.pth" \
        --output "data/results/${model}_loo_results.json" \
        --model_name "$model" \
        2>&1 | tee "logs/${model}_loo_cv.log"

    echo "✅ ${model} LOO CV 완료"
done

echo ""
echo "Phase 3 완료: 모든 LOO CV 완료"
echo "시간: $(date)"
echo ""
echo "결과 파일:"
ls -lh data/results/
