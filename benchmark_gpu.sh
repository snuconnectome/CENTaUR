#!/bin/bash
# CENTaUR GPU 벤치마크 스크립트

set -e
cd ~/git/CENTaUR
source venv/bin/activate

echo "=== CENTaUR GPU 벤치마크 시작 ==="
date

# 출력 디렉토리 생성
mkdir -p outputs logs

# Test 1: 소규모 샘플 (10개)
echo -e "\n[Test 1] 10 samples 처리 시간 측정"
time python scripts/extract_centaur_features.py \
    --model qwen25-base \
    --dataset ko_centaur/data/choices13k_100.jsonl \
    --n_samples 10 \
    --output outputs/benchmark_10samples.npz

# Test 2: 중규모 샘플 (100개)
echo -e "\n[Test 2] 100 samples 처리 시간 측정"
time python scripts/extract_centaur_features.py \
    --model qwen25-base \
    --dataset ko_centaur/data/choices13k_100.jsonl \
    --n_samples 100 \
    --output outputs/benchmark_100samples.npz

# GPU 정보 출력
echo -e "\n[GPU Info]"
nvidia-smi --query-gpu=name,memory.used,memory.total,utilization.gpu --format=csv

echo -e "\n=== 벤치마크 완료 ==="
echo "결과 파일:"
ls -lh outputs/benchmark_*.npz 2>/dev/null || echo "No benchmark files yet"
date
