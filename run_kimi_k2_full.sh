#!/bin/bash
# Kimi K2 Full Feature Extraction

set -e
cd ~/git/CENTaUR
source venv/bin/activate

mkdir -p outputs logs

echo "=== Kimi K2 Feature Extraction 시작 ==="
date

echo "⚠️  주의: Kimi K2는 매우 큰 모델입니다 (1T 파라미터, 활성 32B)"
echo "   - INT4 양자화 사용 (기본값)"
echo "   - 예상 GPU 메모리: ~16GB (활성 파라미터만)"
echo "   - 다운로드 시간이 오래 걸릴 수 있습니다"
echo ""

# Feature extraction 실행
python scripts/extract_centaur_features.py \
    --model kimi-k2 \
    --dataset ko_centaur/data/choices13k_100.jsonl \
    --output outputs/kimi_k2_features.npz \
    2>&1 | tee logs/kimi_k2_extraction.log

echo "=== Feature extraction 완료 ==="
date

# 결과 확인
python -c "
import numpy as np
data = np.load('outputs/kimi_k2_features.npz')
print(f'Features shape: {data[\"features\"].shape}')
print(f'Labels shape: {data[\"labels\"].shape}')
print(f'Total samples: {len(data[\"labels\"])}')
"

