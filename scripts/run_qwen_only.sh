#!/bin/bash
# Qwen2.5만 Feature Extraction 및 평가
# DeepSeek은 건너뛰기

set -e

echo "==========================================="
echo "Qwen2.5 Feature Extraction & Evaluation"
echo "==========================================="
echo "시간: $(date)"
echo ""

mkdir -p data/features
mkdir -p data/results

# Qwen2.5 Fine-tuned 모델 확인
if [ ! -d "models/qwen25-32b-qlora" ]; then
    echo "❌ Qwen2.5 Fine-tuned 모델 없음"
    exit 1
fi

echo "✅ Qwen2.5 Fine-tuned 모델 확인됨"
echo ""

# Phase 2: Feature Extraction (Qwen2.5만)
echo "=== Phase 2: Qwen2.5 Feature Extraction ==="

# dgx-venv Python 경로
PYTHON="/home/juke/git/CENTaUR/dgx-venv/bin/python"

# Qwen2.5 Base (NO quantization - GB10 compatibility)
if [ ! -f "data/features/qwen25_base_features.pth" ]; then
    echo "⏳ Qwen2.5 Base Feature Extraction (bf16, no quantization)..."
    $PYTHON scripts/extract_centaur_features.py \
        --model qwen25-base \
        --dataset ko_centaur/data/choices13k_100.jsonl \
        --output data/features/qwen25_base_features.pth \
        --no-quantization
    echo "✅ Qwen2.5 Base 완료"
else
    echo "✅ Qwen2.5 Base features 이미 존재"
fi

# Qwen2.5 Fine-tuned (NO quantization - GB10 compatibility)
if [ ! -f "data/features/qwen25_finetuned_features.pth" ]; then
    echo "⏳ Qwen2.5 Fine-tuned Feature Extraction (bf16, no quantization)..."
    $PYTHON scripts/extract_centaur_features.py \
        --model qwen25 \
        --dataset ko_centaur/data/choices13k_100.jsonl \
        --output data/features/qwen25_finetuned_features.pth \
        --no-quantization
    echo "✅ Qwen2.5 Fine-tuned 완료"
else
    echo "✅ Qwen2.5 Fine-tuned features 이미 존재"
fi

echo ""
echo "=== Phase 3: LOO Cross-Validation ==="

# Qwen2.5 Base LOO CV
if [ ! -f "data/results/qwen25_base_loo_results.json" ]; then
    echo "⏳ Qwen2.5 Base LOO CV..."
    $PYTHON scripts/fit_centaur_loo_cv.py \
        --features data/features/qwen25_base_features.pth \
        --output data/results/qwen25_base_loo_results.json
    echo "✅ Qwen2.5 Base LOO CV 완료"
else
    echo "✅ Qwen2.5 Base LOO CV 결과 이미 존재"
fi

# Qwen2.5 Fine-tuned LOO CV
if [ ! -f "data/results/qwen25_finetuned_loo_results.json" ]; then
    echo "⏳ Qwen2.5 Fine-tuned LOO CV..."
    $PYTHON scripts/fit_centaur_loo_cv.py \
        --features data/features/qwen25_finetuned_features.pth \
        --output data/results/qwen25_finetuned_loo_results.json
    echo "✅ Qwen2.5 Fine-tuned LOO CV 완료"
else
    echo "✅ Qwen2.5 Fine-tuned LOO CV 결과 이미 존재"
fi

echo ""
echo "==========================================="
echo "✅ Qwen2.5 평가 완료!"
echo "==========================================="
echo "시간: $(date)"
echo ""
echo "결과 파일:"
ls -lh data/features/qwen25*.pth
ls -lh data/results/qwen25*.json
