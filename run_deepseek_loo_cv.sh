#!/bin/bash
# DeepSeek LOO Cross-Validation

set -e
cd ~/git/CENTaUR
source venv/bin/activate

mkdir -p outputs logs

echo "=== DeepSeek LOO CV 시작 ==="
date

python scripts/fit_centaur_loo_cv_flexible.py \
    outputs/deepseek_base_features.npz \
    outputs/deepseek_base_nll_results.json \
    --model_name "DeepSeek-R1-Distill-Qwen-32B-Base" \
    2>&1 | tee logs/deepseek_loo_cv.log

echo "=== LOO CV 완료 ==="
date

# NLL 결과 확인
python -c "
import json
with open('outputs/deepseek_base_nll_results.json') as f:
    results = json.load(f)
print(f'Mean NLL: {results[\"mean_nll\"]:.2f}')
print(f'Std NLL: {results[\"std_nll\"]:.2f}')
print(f'Total samples: {results[\"total_samples\"]}')
"
