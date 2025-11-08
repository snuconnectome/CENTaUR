#!/bin/bash
# GPT-OSS-20B Full Feature Extraction

set -e
cd ~/git/CENTaUR
source venv/bin/activate

mkdir -p outputs logs

echo "=== GPT-OSS-20B Feature Extraction 시작 ==="
date

# 모델 다운로드 확인
MODEL_PATH="/home/connectome/connectome1/models/gpt-oss-20b"
if [ ! -d "$MODEL_PATH" ]; then
    echo "⚠️  GPT-OSS-20B 모델이 다운로드되지 않았습니다."
    echo "다운로드 스크립트 실행:"
    echo "  sbatch ko_centaur/scripts/download_gpt_oss_optimized.sh"
    echo ""
    echo "또는 수동 다운로드:"
    echo "  hf download openai/gpt-oss-20b --local-dir $MODEL_PATH"
    exit 1
fi

echo "✓ 모델 경로 확인: $MODEL_PATH"

# Feature extraction 실행
python scripts/extract_centaur_features.py \
    --model gpt-oss \
    --dataset ko_centaur/data/choices13k_100.jsonl \
    --output outputs/gpt_oss_features.npz \
    2>&1 | tee logs/gpt_oss_extraction.log

echo "=== Feature extraction 완료 ==="
date

# 결과 확인
python -c "
import numpy as np
data = np.load('outputs/gpt_oss_features.npz')
print(f'Features shape: {data[\"features\"].shape}')
print(f'Labels shape: {data[\"labels\"].shape}')
print(f'Total samples: {len(data[\"labels\"])}')
"

