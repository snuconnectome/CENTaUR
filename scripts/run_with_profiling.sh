#!/bin/bash
# GPU 프로파일링과 함께 작업을 실행하는 스크립트

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# 인자 확인
if [ $# -lt 2 ]; then
    echo "Usage: $0 <job_name> <command> [args...]"
    echo "Example: $0 qwen25_loo_cv 'python scripts/fit_centaur_loo_cv_flexible.py outputs/qwen25_base_features.npz outputs/qwen25_base_nll_results.json'"
    exit 1
fi

JOB_NAME=$1
shift
COMMAND="$@"

PROFILE_OUTPUT="${PROJECT_DIR}/profiles/${JOB_NAME}.nsys-rep"
PROFILE_SQLITE="${PROJECT_DIR}/profiles/${JOB_NAME}.sqlite"
LOG_FILE="${PROJECT_DIR}/logs/${JOB_NAME}_profiled.log"

# 디렉토리 생성
mkdir -p "${PROJECT_DIR}/profiles"
mkdir -p "${PROJECT_DIR}/logs"

echo "=========================================="
echo "GPU Profiling Job: $JOB_NAME"
echo "=========================================="
echo "Command: $COMMAND"
echo "Profile output: $PROFILE_OUTPUT"
echo "Log file: $LOG_FILE"
echo ""

# GPU 상태 확인
echo "=== GPU Status Before ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits
echo ""

# 프로파일링과 함께 실행
echo "Starting profiled execution..."
echo "Start time: $(date)"

/usr/local/bin/nsys profile \
    --trace=cuda,nvtx \
    --output="$PROFILE_OUTPUT" \
    --force-overwrite=true \
    --capture-range=none \
    bash -c "$COMMAND" 2>&1 | tee "$LOG_FILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

# GPU 상태 확인
echo ""
echo "=== GPU Status After ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total --format=csv,noheader,nounits
echo ""

# 프로파일 리포트 생성
if [ -f "$PROFILE_OUTPUT" ]; then
    echo "=== Generating Profile Reports ==="
    
    # SQLite export
    /usr/local/bin/nsys export --type=sqlite --output="$PROFILE_SQLITE" --force-overwrite=true "$PROFILE_OUTPUT" 2>&1 | tail -3
    
    # CUDA API Summary
    echo ""
    echo "--- CUDA API Summary ---"
    /usr/local/bin/nsys stats --report cuda_api_sum "$PROFILE_OUTPUT" 2>&1 | grep -A 20 "CUDA API Summary" | head -15
    
    # GPU Memory Summary (if available)
    echo ""
    echo "--- GPU Memory Summary ---"
    /usr/local/bin/nsys stats --report cuda_gpu_mem_size_sum "$PROFILE_OUTPUT" 2>&1 | head -20
    
    echo ""
    echo "Profile files:"
    ls -lh "$PROFILE_OUTPUT" "$PROFILE_SQLITE" 2>/dev/null || true
fi

if [ $EXIT_CODE -eq 0 ]; then
    echo ""
    echo "✅ Job completed successfully!"
else
    echo ""
    echo "❌ Job failed with exit code $EXIT_CODE"
    echo "Check log: $LOG_FILE"
fi

exit $EXIT_CODE

