#!/bin/bash
# CENTaUR 실험 모니터링 및 완료 알림

RESULTS_FILE="outputs/deepseek_base_nll_results.json"
CHECK_INTERVAL=300  # 5분마다 확인
EMAIL="cha.jiook@gmail.com"

echo "=== CENTaUR 실험 모니터 시작 ==="
echo "결과 파일 대기 중: $RESULTS_FILE"
echo "체크 간격: ${CHECK_INTERVAL}초"

while true; do
    if [ -f "$RESULTS_FILE" ]; then
        echo ""
        echo "✅ 실험 완료 감지!"
        date
        
        # 결과 출력
        echo ""
        echo "=== 최종 결과 ==="
        python generate_report.py
        
        # 이메일 전송 시도 (시스템에 mail 명령어가 있는 경우)
        if command -v mail &> /dev/null; then
            python generate_report.py | mail -s "CENTaUR 실험 완료" $EMAIL
            echo "✉️  이메일 전송 완료: $EMAIL"
        else
            echo "⚠️  mail 명령어 없음 - 이메일 전송 불가"
            echo "📋 결과는 ~/git/CENTaUR/outputs/ 에 저장되었습니다"
        fi
        
        # 완료 파일 생성
        echo "실험 완료 시각: $(date)" > EXPERIMENT_COMPLETED.txt
        cat outputs/qwen25_base_nll_results.json >> EXPERIMENT_COMPLETED.txt
        cat outputs/deepseek_base_nll_results.json >> EXPERIMENT_COMPLETED.txt
        
        break
    else
        echo "대기 중... ($(date))"
        sleep $CHECK_INTERVAL
    fi
done

echo ""
echo "=== 모니터링 종료 ==="
