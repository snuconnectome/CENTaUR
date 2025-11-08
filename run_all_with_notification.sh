#!/bin/bash
# 전체 실험 자동 실행 및 완료 알림

set -e
cd ~/git/CENTaUR

echo "=== CENTaUR 전체 실험 시작 ==="
date

# Phase 1: Benchmark
echo -e "\n### Phase 1: Benchmark ###"
./benchmark_gpu.sh

# Phase 2: Feature Extraction
echo -e "\n### Phase 2: Qwen2.5 Feature Extraction ###"
./run_qwen25_full.sh

echo -e "\n### Phase 2: DeepSeek Feature Extraction ###"
./run_deepseek_full.sh

# Phase 3: LOO CV
echo -e "\n### Phase 3: Qwen2.5 LOO CV ###"
./run_qwen25_loo_cv.sh

echo -e "\n### Phase 3: DeepSeek LOO CV ###"
./run_deepseek_loo_cv.sh

# Phase 4: Report & Notification
echo -e "\n### Phase 4: Final Report & Notification ###"
python generate_report.py

# 완료 알림
./notify_completion.sh

echo -e "\n=== 모든 실험 완료 ==="
date
