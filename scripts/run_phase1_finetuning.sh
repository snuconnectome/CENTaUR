#!/bin/bash
# Phase 1: Fine-tuning
# Qwen2.5-32B와 DeepSeek-R1-32B를 QLoRA로 훈련

set -e

echo "Phase 1: Fine-tuning 시작"
echo "시간: $(date)"
echo ""

# Qwen2.5-32B (이미 완료)
if [ -d "models/qwen25-32b-qlora" ]; then
    echo "✅ Qwen2.5-32B Fine-tuning 이미 완료"
else
    echo "⏳ Qwen2.5-32B Fine-tuning 시작..."
    ./scripts/train_qwen25_ngc.sh
    echo "✅ Qwen2.5-32B Fine-tuning 완료"
fi

echo ""

# DeepSeek-R1-32B
if [ -d "models/deepseek-r1-qlora" ]; then
    echo "✅ DeepSeek-R1 Fine-tuning 이미 완료"
else
    echo "⏳ DeepSeek-R1 Fine-tuning 시작..."

    # 스크립트가 없으면 생성
    if [ ! -f "scripts/train_deepseek_r1_ngc.sh" ]; then
        echo "DeepSeek-R1 훈련 스크립트 생성 중..."
        # Qwen 스크립트를 복사하고 수정
        cp scripts/train_qwen25_ngc.sh scripts/train_deepseek_r1_ngc.sh
        sed -i 's/qwen25/deepseek-r1/g' scripts/train_deepseek_r1_ngc.sh
        sed -i 's/Qwen2.5/DeepSeek-R1/g' scripts/train_deepseek_r1_ngc.sh
        chmod +x scripts/train_deepseek_r1_ngc.sh
    fi

    ./scripts/train_deepseek_r1_ngc.sh
    echo "✅ DeepSeek-R1 Fine-tuning 완료"
fi

echo ""
echo "Phase 1 완료: 모든 Fine-tuning 작업 완료"
echo "시간: $(date)"
