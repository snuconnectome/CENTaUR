#!/bin/bash
# 실험 완료 후 실행할 알림 스크립트

EMAIL="cha.jiook@gmail.com"
SUBJECT="[CENTaUR] dgx-spark 실험 완료"

# 결과 리포트 생성
REPORT=$(python generate_report.py)

# 메시지 구성
MESSAGE="
CENTaUR 실험이 완료되었습니다.

서버: dgx-spark
완료 시각: $(date)
실험 디렉토리: ~/git/CENTaUR

$REPORT

---
자동 생성 메시지
"

# 이메일 전송 시도
if command -v mail &> /dev/null; then
    echo "$MESSAGE" | mail -s "$SUBJECT" $EMAIL
    echo "✉️  이메일 전송 완료: $EMAIL"
elif command -v sendmail &> /dev/null; then
    echo -e "Subject: $SUBJECT\n\n$MESSAGE" | sendmail $EMAIL
    echo "✉️  sendmail로 전송 완료: $EMAIL"
else
    echo "⚠️  이메일 클라이언트 없음"
    echo "📋 결과를 파일로 저장합니다: EXPERIMENT_RESULTS.txt"
    echo "$MESSAGE" > EXPERIMENT_RESULTS.txt
    cat EXPERIMENT_RESULTS.txt
fi
