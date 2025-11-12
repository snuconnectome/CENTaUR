#!/bin/bash
# EXAONE-3.5-32B Feature Extraction 성능 측정 스크립트

set -e

echo "=========================================="
echo "EXAONE-3.5-32B Feature Extraction 성능 측정"
echo "=========================================="
echo ""

# GPU 모니터링 시작 (백그라운드)
echo "1. GPU 모니터링 시작..."
nvidia-smi dmon -s u -c 1000 > gpu_monitor.log 2>&1 &
MONITOR_PID=$!
sleep 2

# 컨테이너에서 Feature Extraction 실행
echo "2. Feature Extraction 실행 중..."
START_TIME=$(date +%s)

echo "462773" | sudo -S docker run --rm --gpus=all \
  -v /home/juke/git/CENTaUR:/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:24.08-py3 \
  python scripts/extract_centaur_features.py \
    --model exaone35-base \
    --dataset ko_centaur/data/choices13k_100.jsonl \
    --output outputs/exaone35_features_ngc.npz \
    2>&1 | tee extraction_log.txt

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

# 모니터링 종료
kill $MONITOR_PID 2>/dev/null || true
sleep 1

# 결과 분석
echo ""
echo "3. 성능 분석:"
echo "   총 소요 시간: ${ELAPSED}초 ($(($ELAPSED / 60))분 $(($ELAPSED % 60))초)"
echo ""

if [ -f gpu_monitor.log ]; then
    echo "   GPU 사용률 통계:"
    awk '{if ($2 != "" && $2 != "gpu") print $2}' gpu_monitor.log | \
      awk '{sum+=$1; count++; if ($1>max) max=$1} END {if(count>0) printf "     평균: %.1f%%, 최대: %.1f%%\n", sum/count, max}'
fi

echo ""
echo "=========================================="
echo "측정 완료"
echo "=========================================="

