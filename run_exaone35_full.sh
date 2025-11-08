#!/bin/bash
# EXAONE-3.5-32B Full Feature Extraction

set -e
cd ~/git/CENTaUR
source venv/bin/activate

mkdir -p outputs logs

echo "=== EXAONE-3.5-32B Feature Extraction 시작 ==="
date

# Feature extraction 실행
python scripts/extract_centaur_features.py \
    --model exaone35-base \
    --dataset ko_centaur/data/choices13k_100.jsonl \
    --output outputs/exaone35_features.npz \
    2>&1 | tee logs/exaone35_extraction.log

echo "=== Feature extraction 완료 ==="
date

# 결과 확인
python -c "
import numpy as np
data = np.load('outputs/exaone35_features.npz')
print(f'Features shape: {data[\"features\"].shape}')
print(f'Labels shape: {data[\"labels\"].shape}')
print(f'Total samples: {len(data[\"labels\"])}')
"
