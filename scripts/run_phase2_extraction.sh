#!/bin/bash
# Phase 2: Feature Extraction
# 모든 모델 (Base 5개 + Fine-tuned 2개)에 대해 hidden states 추출

set -e

echo "Phase 2: Feature Extraction 시작"
echo "시간: $(date)"
echo ""

mkdir -p data/features

# ============================================================
# Base 모델 (5개)
# ============================================================

echo "=== Base 모델 Feature Extraction ==="
echo ""

# 1. Qwen2.5-32B Base
if [ -f "data/features/qwen25_base_features.pth" ]; then
    echo "✅ Qwen2.5 Base features 이미 존재"
else
    echo "⏳ Qwen2.5 Base Feature Extraction..."
    if [ -f "outputs/qwen25_base_features.npz" ]; then
        # NPZ를 PTH로 변환
        ./ngc-python -c "
import numpy as np
import torch
data = np.load('outputs/qwen25_base_features.npz')
torch.save({
    'features': torch.from_numpy(data['features']),
    'labels': torch.from_numpy(data['labels'])
}, 'data/features/qwen25_base_features.pth')
"
        echo "✅ Qwen2.5 Base 변환 완료"
    else
        ./ngc-python scripts/extract_centaur_features.py \
            --model qwen25-base \
            --dataset ko_centaur/data/choices13k_100.jsonl \
            --output data/features/qwen25_base_features.pth
        echo "✅ Qwen2.5 Base 완료"
    fi
fi

echo ""

# 2. DeepSeek-R1 Base
if [ -f "data/features/deepseek_base_features.pth" ]; then
    echo "✅ DeepSeek Base features 이미 존재"
else
    echo "⏳ DeepSeek Base Feature Extraction..."
    if [ -f "outputs/deepseek_base_features.npz" ]; then
        ./ngc-python -c "
import numpy as np
import torch
data = np.load('outputs/deepseek_base_features.npz')
torch.save({
    'features': torch.from_numpy(data['features']),
    'labels': torch.from_numpy(data['labels'])
}, 'data/features/deepseek_base_features.pth')
"
        echo "✅ DeepSeek Base 변환 완료"
    else
        ./ngc-python scripts/extract_centaur_features.py \
            --model deepseek-base \
            --dataset ko_centaur/data/choices13k_100.jsonl \
            --output data/features/deepseek_base_features.pth
        echo "✅ DeepSeek Base 완료"
    fi
fi

echo ""

# 3. EXAONE-3.5-32B Base
if [ -f "data/features/exaone35_base_features.pth" ]; then
    echo "✅ EXAONE-3.5 Base features 이미 존재"
else
    echo "⏳ EXAONE-3.5 Base Feature Extraction..."
    if [ -f "outputs/exaone35_features.npz" ]; then
        ./ngc-python -c "
import numpy as np
import torch
data = np.load('outputs/exaone35_features.npz')
torch.save({
    'features': torch.from_numpy(data['features']),
    'labels': torch.from_numpy(data['labels'])
}, 'data/features/exaone35_base_features.pth')
"
        echo "✅ EXAONE-3.5 Base 변환 완료"
    else
        ./ngc-python scripts/extract_centaur_features.py \
            --model exaone35-base \
            --dataset ko_centaur/data/choices13k_100.jsonl \
            --output data/features/exaone35_base_features.pth
        echo "✅ EXAONE-3.5 Base 완료"
    fi
fi

echo ""

# 4. GPT-OSS-20B Base (새로 추가)
if [ -f "data/features/gpt_oss_20b_base_features.pth" ]; then
    echo "✅ GPT-OSS-20B Base features 이미 존재"
else
    echo "⏳ GPT-OSS-20B Base Feature Extraction..."
    if [ -f "outputs/gpt_oss_base_features.npz" ]; then
        ./ngc-python -c "
import numpy as np
import torch
data = np.load('outputs/gpt_oss_base_features.npz')
torch.save({
    'features': torch.from_numpy(data['features']),
    'labels': torch.from_numpy(data['labels'])
}, 'data/features/gpt_oss_20b_base_features.pth')
"
        echo "✅ GPT-OSS-20B Base 변환 완료"
    else
        ./ngc-python scripts/extract_centaur_features.py \
            --model gpt-oss-base \
            --dataset ko_centaur/data/choices13k_100.jsonl \
            --output data/features/gpt_oss_20b_base_features.pth
        echo "✅ GPT-OSS-20B Base 완료"
    fi
fi

echo ""

# 5. Kimi-K2-Instruct Base (새로 추가)
if [ -f "data/features/kimi_k2_base_features.pth" ]; then
    echo "✅ Kimi-K2 Base features 이미 존재"
else
    echo "⏳ Kimi-K2 Base Feature Extraction..."
    echo "⚠️  주의: Kimi-K2는 1T MoE 모델로 매우 큽니다. 4-bit quantization 사용."
    if [ -f "outputs/kimi_k2_features.npz" ]; then
        ./ngc-python -c "
import numpy as np
import torch
data = np.load('outputs/kimi_k2_features.npz')
torch.save({
    'features': torch.from_numpy(data['features']),
    'labels': torch.from_numpy(data['labels'])
}, 'data/features/kimi_k2_base_features.pth')
"
        echo "✅ Kimi-K2 Base 변환 완료"
    else
        ./ngc-python scripts/extract_centaur_features.py \
            --model kimi-k2 \
            --dataset ko_centaur/data/choices13k_100.jsonl \
            --output data/features/kimi_k2_base_features.pth
        echo "✅ Kimi-K2 Base 완료"
    fi
fi

echo ""

# ============================================================
# Fine-tuned 모델 (2개)
# ============================================================

echo "=== Fine-tuned 모델 Feature Extraction ==="
echo ""

# 1. Qwen2.5 Fine-tuned
if [ -f "data/features/qwen25_finetuned_features.pth" ]; then
    echo "✅ Qwen2.5 Fine-tuned features 이미 존재"
else
    if [ ! -d "models/qwen25-32b-qlora" ]; then
        echo "❌ Qwen2.5 Fine-tuned 모델 없음, Phase 1 먼저 실행하세요"
    else
        echo "⏳ Qwen2.5 Fine-tuned Feature Extraction..."
        ./ngc-python scripts/extract_centaur_features.py \
            --model qwen25 \
            --dataset ko_centaur/data/choices13k_100.jsonl \
            --output data/features/qwen25_finetuned_features.pth
        echo "✅ Qwen2.5 Fine-tuned 완료"
    fi
fi

echo ""

# 2. DeepSeek Fine-tuned
if [ -f "data/features/deepseek_finetuned_features.pth" ]; then
    echo "✅ DeepSeek Fine-tuned features 이미 존재"
else
    if [ ! -d "models/deepseek-r1-qlora" ]; then
        echo "❌ DeepSeek Fine-tuned 모델 없음, Phase 1 먼저 실행하세요"
    else
        echo "⏳ DeepSeek Fine-tuned Feature Extraction..."
        ./ngc-python scripts/extract_centaur_features.py \
            --model deepseek \
            --dataset ko_centaur/data/choices13k_100.jsonl \
            --output data/features/deepseek_finetuned_features.pth
        echo "✅ DeepSeek Fine-tuned 완료"
    fi
fi

echo ""
echo "Phase 2 완료: 모든 Feature Extraction 완료"
echo "시간: $(date)"
echo ""
echo "추출된 특징:"
ls -lh data/features/
