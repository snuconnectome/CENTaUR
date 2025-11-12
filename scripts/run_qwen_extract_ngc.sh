#!/bin/bash
# Qwen2.5 Feature Extraction using NGC Container with GB10

set -e

echo "==========================================="
echo "Qwen2.5 Feature Extraction (NGC Container)"
echo "==========================================="
echo "시간: $(date)"
echo ""

# NGC Container with GPU override (25.01 has newer transformers)
CONTAINER="nvcr.io/nvidia/pytorch:25.01-py3"

# Qwen2.5 Base
if [ ! -f "data/features/qwen25_base_features.pth" ]; then
    echo "⏳ Qwen2.5 Base Feature Extraction (NGC)..."

    docker run --rm --gpus all \
        --entrypoint /bin/bash \
        -v /home/juke/git/CENTaUR:/workspace \
        -w /workspace \
        -e NVIDIA_DISABLE_REQUIRE=1 \
        $CONTAINER \
        -c "
            pip install -q --upgrade transformers && \
            pip install -q bitsandbytes peft datasets accelerate && \
            python scripts/extract_centaur_features.py \
                --model qwen25-base \
                --dataset ko_centaur/data/choices13k_100.jsonl \
                --output data/features/qwen25_base_features.pth \
                --no-quantization
        " 2>&1 | tee logs/ngc_qwen_base_$(date +%Y%m%d_%H%M%S).log

    echo "✅ Qwen2.5 Base 완료"
else
    echo "✅ Qwen2.5 Base features 이미 존재"
fi

# Qwen2.5 Fine-tuned
if [ ! -f "data/features/qwen25_finetuned_features.pth" ]; then
    echo "⏳ Qwen2.5 Fine-tuned Feature Extraction (NGC)..."

    docker run --rm --gpus all \
        --entrypoint /bin/bash \
        -v /home/juke/git/CENTaUR:/workspace \
        -w /workspace \
        -e NVIDIA_DISABLE_REQUIRE=1 \
        $CONTAINER \
        -c "
            pip install -q --upgrade transformers && \
            pip install -q bitsandbytes peft datasets accelerate && \
            python scripts/extract_centaur_features.py \
                --model qwen25 \
                --dataset ko_centaur/data/choices13k_100.jsonl \
                --output data/features/qwen25_finetuned_features.pth \
                --no-quantization
        " 2>&1 | tee logs/ngc_qwen_finetuned_$(date +%Y%m%d_%H%M%S).log

    echo "✅ Qwen2.5 Fine-tuned 완료"
else
    echo "✅ Qwen2.5 Fine-tuned features 이미 존재"
fi

echo ""
echo "==========================================="
echo "✅ NGC Feature Extraction 완료!"
echo "==========================================="
