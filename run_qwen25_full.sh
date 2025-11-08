#!/bin/bash
# Qwen2.5-32B Full Feature Extraction

set -e
cd ~/git/CENTaUR
source venv/bin/activate

mkdir -p outputs logs

echo "=== Qwen2.5-32B Feature Extraction 시작 ==="
date

# 올바른 데이터셋 경로 사용
python scripts/extract_centaur_features.py \
    --model qwen25-base \
    --dataset ko_centaur/data/choices13k_100.jsonl \
    --output outputs/qwen25_base_features.npz \
    2>&1 | tee logs/qwen25_base_extraction.log

echo "=== Feature extraction 완료 ==="
date

# 결과 확인
python -c "
import numpy as np
data = np.load('outputs/qwen25_base_features.npz')
print(f'Features shape: {data[\"features\"].shape}')
print(f'Labels shape: {data[\"labels\"].shape}')
print(f'Total samples: {len(data[\"labels\"])}')
"
