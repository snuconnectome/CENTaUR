#!/bin/bash
# Phase 4: Analysis & Reporting
# 모든 LOO CV 결과를 종합하고 분석

set -e

echo "Phase 4: Analysis & Reporting 시작"
echo "시간: $(date)"
echo ""

mkdir -p reports figures

# 결과 파일 확인
echo "=== 결과 파일 확인 ==="
if [ -z "$(ls -A data/results/*.json 2>/dev/null)" ]; then
    echo "❌ LOO CV 결과가 없습니다. Phase 3를 먼저 실행하세요."
    exit 1
fi

echo "발견된 결과 파일:"
ls -lh data/results/*.json

echo ""
echo "=== 결과 종합 분석 ==="
python scripts/analyze_all_results.py \
    --results_dir data/results/ \
    --output reports/final_analysis.md

echo ""
echo "=== Generation Bias 분석 ==="
python scripts/analyze_generation_bias.py \
    --results_dir data/results/ \
    --features_dir data/features/ \
    --output reports/generation_bias_analysis.md

echo ""
echo "=== 결과 시각화 ==="
python scripts/visualize_results.py \
    --results_dir data/results/ \
    --output_dir figures/

echo ""
echo "Phase 4 완료: 모든 분석 완료"
echo "시간: $(date)"
echo ""
echo "생성된 리포트:"
ls -lh reports/
echo ""
echo "생성된 그래프:"
ls -lh figures/
