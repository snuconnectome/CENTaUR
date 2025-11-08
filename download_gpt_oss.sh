#!/bin/bash
# GPT-OSS-20B 다운로드 스크립트 (로컬 실행용)

set -e

echo "=========================================="
echo "GPT-OSS-20B 다운로드"
echo "=========================================="
echo ""

# 모델 디렉토리 설정
MODEL_DIR="${HOME}/models/gpt-oss-20b"
mkdir -p "$MODEL_DIR"

echo "모델 저장 경로: $MODEL_DIR"
echo ""

# huggingface-cli 확인
if ! command -v huggingface-cli &> /dev/null; then
    echo "❌ huggingface-cli가 설치되지 않았습니다."
    echo "설치 중..."
    pip install -U "huggingface_hub[cli]"
fi

echo "✓ huggingface-cli 확인 완료"
echo ""

# 다운로드 시작
echo "=========================================="
echo "다운로드 시작..."
echo "=========================================="
echo ""

huggingface-cli download openai/gpt-oss-20b \
    --exclude "original/*" "metal/*" \
    --local-dir "$MODEL_DIR" \
    --resume-download

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ 다운로드 완료"
    echo ""
    echo "다운로드 위치: $MODEL_DIR"
    echo "크기: $(du -sh $MODEL_DIR | cut -f1)"
    echo ""
    echo "다음 단계:"
    echo "  ./run_gpt_oss_full.sh  # Feature extraction 실행"
else
    echo "❌ 다운로드 실패 (exit code: $EXIT_CODE)"
fi
echo "=========================================="

exit $EXIT_CODE

