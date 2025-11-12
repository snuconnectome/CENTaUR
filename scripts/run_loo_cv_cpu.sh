#!/bin/bash
# CPU로 LOO CV 실행 (CUDA 호환성 문제 해결)

set -e
cd ~/git/CENTaUR
source venv/bin/activate

mkdir -p outputs logs

MODEL_NAME=${1:-"Qwen2.5-32B-Base"}
FEATURES_FILE=${2:-"outputs/qwen25_base_features.npz"}
OUTPUT_FILE=${3:-"outputs/qwen25_base_nll_results.json"}

echo "=== LOO CV (CPU) 시작 ==="
echo "Model: $MODEL_NAME"
echo "Features: $FEATURES_FILE"
echo "Output: $OUTPUT_FILE"
date

# CPU로 강제 실행
CUDA_VISIBLE_DEVICES="" python scripts/fit_centaur_loo_cv_flexible.py \
    "$FEATURES_FILE" \
    "$OUTPUT_FILE" \
    --model_name "$MODEL_NAME" \
    2>&1 | tee logs/${MODEL_NAME//[^a-zA-Z0-9]/_}_loo_cv_cpu.log

echo "=== LOO CV 완료 ==="
date

# 결과 확인
if [ -f "$OUTPUT_FILE" ]; then
    python -c "
import json
with open('$OUTPUT_FILE') as f:
    results = json.load(f)
print(f'Mean NLL: {results[\"mean_nll\"]:.4f}')
print(f'Std NLL: {results[\"std_nll\"]:.4f}')
print(f'Total samples: {results[\"total_samples\"]}')
"
fi

