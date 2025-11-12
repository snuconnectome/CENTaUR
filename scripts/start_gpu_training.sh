#!/bin/bash
# GPU 학습 및 분석 작업 시작 스크립트

set -e
cd ~/git/CENTaUR
source venv/bin/activate

mkdir -p outputs logs profiles

echo "=========================================="
echo "CENTaUR GPU 학습 및 분석 작업 시작"
echo "=========================================="
echo "시작 시간: $(date)"
echo ""

# GPU 상태 확인
echo "=== GPU 상태 ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
echo ""

# 1. EXAONE-3.5-32B Feature Extraction (GPU 프로파일링 포함)
echo "=== 1. EXAONE-3.5-32B Feature Extraction 시작 ==="
if [ -f "outputs/exaone35_features.npz" ]; then
    echo "⚠️  Feature 파일이 이미 존재합니다. 건너뜁니다."
else
    echo "프로파일링과 함께 실행 중..."
    bash scripts/run_with_profiling.sh exaone35_feature_extraction \
        'source venv/bin/activate && python scripts/extract_centaur_features.py --model exaone35-base --dataset ko_centaur/data/choices13k_100.jsonl --output outputs/exaone35_features.npz' \
        > logs/exaone35_extraction_profiled.log 2>&1 &
    EXAONE_PID=$!
    echo "✅ EXAONE Feature Extraction 시작 (PID: $EXAONE_PID)"
fi
echo ""

# 2. Qwen2.5-32B Base LOO CV (CPU 모드)
echo "=== 2. Qwen2.5-32B Base LOO CV 시작 ==="
if [ -f "outputs/qwen25_base_nll_results.json" ]; then
    echo "⚠️  결과 파일이 이미 존재합니다. 건너뜁니다."
else
    echo "CPU 모드로 실행 중..."
    bash scripts/run_loo_cv_cpu.sh 'Qwen2.5-32B-Base' \
        outputs/qwen25_base_features.npz \
        outputs/qwen25_base_nll_results.json \
        > logs/qwen25_base_loo_cv_cpu.log 2>&1 &
    QWEN_PID=$!
    echo "✅ Qwen2.5 LOO CV 시작 (PID: $QWEN_PID)"
fi
echo ""

# 3. DeepSeek-R1 Base LOO CV (CPU 모드)
echo "=== 3. DeepSeek-R1 Base LOO CV 시작 ==="
if [ -f "outputs/deepseek_base_nll_results.json" ]; then
    echo "⚠️  결과 파일이 이미 존재합니다. 건너뜁니다."
else
    echo "CPU 모드로 실행 중..."
    bash scripts/run_loo_cv_cpu.sh 'DeepSeek-R1-32B-Base' \
        outputs/deepseek_base_features.npz \
        outputs/deepseek_base_nll_results.json \
        > logs/deepseek_base_loo_cv_cpu.log 2>&1 &
    DEEPSEEK_PID=$!
    echo "✅ DeepSeek LOO CV 시작 (PID: $DEEPSEEK_PID)"
fi
echo ""

# 4. 모니터링 시작
echo "=== 4. 모니터링 시작 ==="
echo "10분마다 진행상황을 이메일로 전송합니다."
echo ""

# 모니터링 스크립트 실행 (백그라운드)
python3 scripts/monitor_and_email.py > logs/monitor.log 2>&1 &
MONITOR_PID=$!
echo "✅ 모니터링 시작 (PID: $MONITOR_PID)"
echo ""

# PID 저장
echo "$EXAONE_PID" > logs/exaone_pid.txt 2>/dev/null || true
echo "$QWEN_PID" > logs/qwen_pid.txt 2>/dev/null || true
echo "$DEEPSEEK_PID" > logs/deepseek_pid.txt 2>/dev/null || true
echo "$MONITOR_PID" > logs/monitor_pid.txt

echo "=========================================="
echo "모든 작업이 시작되었습니다!"
echo "=========================================="
echo ""
echo "작업 확인:"
echo "  ps aux | grep -E 'extract_centaur|fit_centaur_loo_cv|monitor_and_email'"
echo ""
echo "로그 확인:"
echo "  tail -f logs/monitor.log"
echo "  tail -f logs/exaone35_extraction_profiled.log"
echo "  tail -f logs/qwen25_base_loo_cv_cpu.log"
echo ""
echo "모니터링 중지:"
echo "  kill \$(cat logs/monitor_pid.txt)"
echo ""

