#!/bin/bash
# SMTP 설정 후 모니터링 재시작 스크립트

cd ~/git/CENTaUR

# 환경변수 설정
export SMTP_PASSWORD="abcdefghijklmnop"
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USER="cha.jiook@gmail.com"
export SMTP_FROM="cha.jiook@gmail.com"

echo "=========================================="
echo "모니터링 재시작 (SMTP 설정 포함)"
echo "=========================================="
echo ""

# 기존 모니터링 종료
if [ -f logs/monitor_pid.txt ]; then
    OLD_PID=$(cat logs/monitor_pid.txt)
    if ps -p $OLD_PID > /dev/null 2>&1; then
        echo "기존 모니터링 종료 중 (PID: $OLD_PID)..."
        kill $OLD_PID 2>/dev/null
        sleep 2
    fi
fi

# 모든 monitor_and_email 프로세스 종료
pkill -f monitor_and_email.py 2>/dev/null
sleep 1

echo "환경변수 확인:"
echo "  SMTP_PASSWORD: ${SMTP_PASSWORD:0:4}****${SMTP_PASSWORD: -4}"
echo "  SMTP_SERVER: $SMTP_SERVER"
echo "  SMTP_USER: $SMTP_USER"
echo ""

# 모니터링 재시작 (환경변수 포함)
echo "모니터링 시작 중..."
nohup env \
    SMTP_PASSWORD="$SMTP_PASSWORD" \
    SMTP_SERVER="$SMTP_SERVER" \
    SMTP_PORT="$SMTP_PORT" \
    SMTP_USER="$SMTP_USER" \
    SMTP_FROM="$SMTP_FROM" \
    python3 scripts/monitor_and_email.py > logs/monitor.log 2>&1 &

NEW_PID=$!
echo $NEW_PID > logs/monitor_pid.txt

sleep 2

# 프로세스 확인
if ps -p $NEW_PID > /dev/null 2>&1; then
    echo "✅ 모니터링 시작 완료 (PID: $NEW_PID)"
    echo ""
    echo "테스트 이메일 전송 중..."
    
    # 테스트 이메일
    python3 << EOF
import os
import sys
os.environ['SMTP_PASSWORD'] = '$SMTP_PASSWORD'
os.environ['SMTP_SERVER'] = '$SMTP_SERVER'
os.environ['SMTP_PORT'] = '$SMTP_PORT'
os.environ['SMTP_USER'] = '$SMTP_USER'
os.environ['SMTP_FROM'] = '$SMTP_FROM'
sys.path.insert(0, 'scripts')
from monitor_and_email import send_email

result = send_email(
    '[CENTaUR] 모니터링 시작',
    '모니터링이 시작되었습니다!\n\n이제 10분마다 진행상황 리포트를 받으실 수 있습니다.\n\n설정된 SMTP: $SMTP_SERVER'
)

if result:
    print('✅ 테스트 이메일 전송 성공!')
    print('   cha.jiook@gmail.com으로 확인해주세요.')
else:
    print('⚠️  이메일 전송 실패 (파일로 저장됨)')
    print('   logs/email_*.txt 파일 확인')
EOF
    
    echo ""
    echo "=========================================="
    echo "설정 완료!"
    echo "=========================================="
    echo ""
    echo "다음 이메일: 약 10분 후"
    echo "로그 확인: tail -f logs/monitor.log"
    echo ""
else
    echo "❌ 모니터링 시작 실패"
    echo "로그 확인: tail -20 logs/monitor.log"
    exit 1
fi

