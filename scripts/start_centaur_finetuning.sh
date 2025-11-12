#!/bin/bash
# CENTaUR Fine-tuning 시작 스크립트 (GPU 프로파일링 포함)

set -e
cd ~/git/CENTaUR
source venv/bin/activate

mkdir -p models logs profiles outputs

echo "=========================================="
echo "CENTaUR Fine-tuning 시작"
echo "=========================================="
echo "시작 시간: $(date)"
echo ""

# GPU 상태 확인
echo "=== GPU 상태 ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader,nounits
echo ""

# 모델 선택
MODEL=${1:-"qwen25"}  # 기본값: qwen25

case $MODEL in
    qwen25|qwen)
        echo "=== Qwen2.5-32B QLoRA Fine-tuning 시작 ==="
        CONFIG_FILE="$(pwd)/ko_centaur/configs/training_qwen25_32b_qlora.yaml"
        TRAIN_SCRIPT="$(pwd)/ko_centaur/training/train_qwen25_32b_qlora.py"
        MODEL_NAME="qwen25-32b-qlora"
        ;;
    exaone|exaone35)
        echo "=== EXAONE-3.5-7.8B QLoRA Fine-tuning 시작 ==="
        CONFIG_FILE="ko_centaur/configs/train_exaone_qlora.yaml"
        TRAIN_SCRIPT="ko_centaur/training/train_exaone_qlora.py"
        MODEL_NAME="exaone-qlora"
        ;;
    *)
        echo "❌ 알 수 없는 모델: $MODEL"
        echo "사용 가능한 모델: qwen25, exaone"
        exit 1
        ;;
esac

# 설정 파일 확인
if [ ! -f "$CONFIG_FILE" ]; then
    echo "⚠️  설정 파일이 없습니다: $CONFIG_FILE"
    echo "기본 설정으로 진행합니다..."
    CONFIG_FILE=""
fi

# 데이터셋 확인
DATASET="ko_centaur/data/choices13k_100.jsonl"
if [ ! -f "$DATASET" ]; then
    echo "❌ 데이터셋 파일이 없습니다: $DATASET"
    exit 1
fi

echo "모델: $MODEL_NAME"
echo "스크립트: $TRAIN_SCRIPT"
echo "데이터셋: $DATASET"
echo ""

# GPU 프로파일링과 함께 fine-tuning 시작
PROFILE_OUTPUT="profiles/${MODEL_NAME}_finetuning.nsys-rep"
LOG_FILE="logs/${MODEL_NAME}_finetuning.log"
OUTPUT_DIR="models/${MODEL_NAME}"

echo "=== Fine-tuning 시작 (GPU 프로파일링 포함) ==="
echo "프로파일 출력: $PROFILE_OUTPUT"
echo "로그 파일: $LOG_FILE"
echo "출력 디렉토리: $OUTPUT_DIR"
echo ""

# 프로파일링과 함께 실행
if [ -n "$CONFIG_FILE" ]; then
    /usr/local/bin/nsys profile \
        --trace=cuda,nvtx \
        --output="$PROFILE_OUTPUT" \
        --force-overwrite=true \
        --capture-range=none \
        bash -c "source venv/bin/activate && python3 $TRAIN_SCRIPT --config $CONFIG_FILE" \
        > "$LOG_FILE" 2>&1 &
else
    # 기본 설정으로 실행 (설정 파일 없을 때)
    /usr/local/bin/nsys profile \
        --trace=cuda,nvtx \
        --output="$PROFILE_OUTPUT" \
        --force-overwrite=true \
        --capture-range=none \
        bash -c "source venv/bin/activate && python3 $TRAIN_SCRIPT --dataset $DATASET --output_dir $OUTPUT_DIR" \
        > "$LOG_FILE" 2>&1 &
fi

TRAIN_PID=$!
echo "$TRAIN_PID" > "logs/${MODEL_NAME}_train_pid.txt"

echo "✅ Fine-tuning 시작 완료!"
echo "  PID: $TRAIN_PID"
echo "  로그: tail -f $LOG_FILE"
echo "  프로파일: $PROFILE_OUTPUT"
echo ""

# 모니터링 정보
echo "=== 모니터링 ==="
echo "진행 상황 확인:"
echo "  tail -f $LOG_FILE"
echo ""
echo "GPU 사용률 확인:"
echo "  watch -n 1 nvidia-smi"
echo ""
echo "프로세스 확인:"
echo "  ps -p $TRAIN_PID"
echo ""

echo "=========================================="
echo "Fine-tuning이 백그라운드에서 실행 중입니다"
echo "=========================================="

