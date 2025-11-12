#!/bin/bash
# Ko-CENTaUR 전체 파이프라인 자동 실행
# 모든 단계를 순차적으로 실행합니다

set -e

PROJECT_DIR="/home/juke/git/CENTaUR"
cd "$PROJECT_DIR"

echo "=========================================="
echo "Ko-CENTaUR 전체 파이프라인 시작"
echo "=========================================="
echo "시작 시간: $(date)"
echo ""

# 로그 디렉토리 생성
mkdir -p logs data/features data/results reports figures

# Phase 1: Fine-tuning
echo "=== Phase 1: Fine-tuning ==="
./scripts/run_phase1_finetuning.sh

# Phase 2: Feature Extraction
echo ""
echo "=== Phase 2: Feature Extraction ==="
./scripts/run_phase2_extraction.sh

# Phase 3: LOO Cross-Validation
echo ""
echo "=== Phase 3: LOO Cross-Validation ==="
./scripts/run_phase3_loo_cv.sh

# Phase 4: Analysis
echo ""
echo "=== Phase 4: Analysis & Reporting ==="
./scripts/run_phase4_analysis.sh

echo ""
echo "=========================================="
echo "전체 파이프라인 완료!"
echo "종료 시간: $(date)"
echo "=========================================="
echo ""
echo "결과 확인:"
echo "  - 특징: data/features/"
echo "  - LOO CV 결과: data/results/"
echo "  - 리포트: reports/"
echo "  - 그래프: figures/"
