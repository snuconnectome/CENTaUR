#!/bin/bash
# DeepSeek-R1-Distill-Qwen-32B Full Feature Extraction

set -e
cd ~/git/CENTaUR
source venv/bin/activate

mkdir -p outputs logs

echo "=== DeepSeek-R1 Feature Extraction 시작 ==="
date

python scripts/extract_centaur_features.py \
    --model deepseek-base \
    --dataset ko_centaur/data/choices13k_100.jsonl \
    --output outputs/deepseek_base_features.npz \
    2>&1 | tee logs/deepseek_base_extraction.log

echo "=== Feature extraction 완료 ==="
date

# 결과 확인
python -c "
import numpy as np
data = np.load('outputs/deepseek_base_features.npz')
print(f'Features shape: {data[\"features\"].shape}')
print(f'Labels shape: {data[\"labels\"].shape}')
print(f'Total samples: {len(data[\"labels\"])}')
"
