#!/bin/bash
# EXAONE-3.5-32B LOO Cross-Validation

set -e
cd ~/git/CENTaUR
source venv/bin/activate

mkdir -p outputs logs

echo "=== EXAONE-3.5-32B LOO CV 시작 ==="
date

# Feature 파일 확인
if [ ! -f "outputs/exaone35_features.npz" ]; then
    echo "❌ Feature 파일이 없습니다: outputs/exaone35_features.npz"
    echo "먼저 feature extraction을 실행하세요:"
    echo "  ./run_exaone35_full.sh"
    exit 1
fi

python scripts/fit_centaur_loo_cv_flexible.py \
    outputs/exaone35_features.npz \
    outputs/exaone35_nll_results.json \
    --model_name "EXAONE-3.5-32B" \
    2>&1 | tee logs/exaone35_loo_cv.log

echo "=== LOO CV 완료 ==="
date

# NLL 결과 확인
python -c "
import json
with open('outputs/exaone35_nll_results.json') as f:
    results = json.load(f)
print(f'Mean NLL: {results[\"mean_nll\"]:.2f}')
print(f'Std NLL: {results[\"std_nll\"]:.2f}')
print(f'Total samples: {results[\"total_samples\"]}')
"

