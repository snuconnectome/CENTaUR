#!/bin/bash
# Qwen2.5 LOO Cross-Validation

set -e
cd ~/git/CENTaUR
source venv/bin/activate

mkdir -p outputs logs

echo "=== Qwen2.5 LOO CV 시작 ==="
date

python scripts/fit_centaur_loo_cv_flexible.py \
    outputs/qwen25_base_features.npz \
    outputs/qwen25_base_nll_results.json \
    --model_name "Qwen2.5-32B-Instruct-Base" \
    2>&1 | tee logs/qwen25_loo_cv.log

echo "=== LOO CV 완료 ==="
date

# NLL 결과 확인
python -c "
import json
with open('outputs/qwen25_base_nll_results.json') as f:
    results = json.load(f)
print(f'Mean NLL: {results[\"mean_nll\"]:.2f}')
print(f'Std NLL: {results[\"std_nll\"]:.2f}')
print(f'Total samples: {results[\"total_samples\"]}')
"
