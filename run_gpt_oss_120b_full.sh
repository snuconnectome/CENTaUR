#!/bin/bash
# GPT-OSS-120B Full Feature Extraction (requires model download first)

set -e
cd ~/git/CENTaUR
source venv/bin/activate

mkdir -p outputs logs

echo "=== GPT-OSS-120B Feature Extraction 시작 ==="
date

python scripts/extract_centaur_features.py \
    --model gpt-oss-120b \
    --dataset ko_centaur/data/choices13k_100.jsonl \
    --output outputs/gpt_oss_120b_features.npz \
    2>&1 | tee logs/gpt_oss_120b_extraction.log

echo "=== Feature extraction 완료 ==="
date

# 결과 확인
python -c "
import numpy as np
data = np.load('outputs/gpt_oss_120b_features.npz')
print(f'Features shape: {data[\"features\"].shape}')
print(f'Labels shape: {data[\"labels\"].shape}')
print(f'Total samples: {len(data[\"labels\"])}')
"
