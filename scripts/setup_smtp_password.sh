#!/bin/bash
# SMTP_PASSWORD 설정 스크립트

echo "=========================================="
echo "SMTP_PASSWORD 설정 가이드"
echo "=========================================="
echo ""

# 1단계: Gmail 앱 비밀번호 생성 안내
echo "1단계: Gmail 앱 비밀번호 생성"
echo "----------------------------------------"
echo "다음 링크에서 앱 비밀번호를 생성하세요:"
echo "  https://myaccount.google.com/apppasswords"
echo ""
echo "생성 방법:"
echo "  1. '앱 선택' → '기타(맞춤 이름)' 선택"
echo "  2. 이름 입력: 'CENTaUR Monitor'"
echo "  3. '만들기' 클릭"
echo "  4. 16자리 비밀번호 복사 (예: abcd efgh ijkl mnop)"
echo ""
echo "⚠️  주의: 2단계 인증이 활성화되어 있어야 합니다!"
echo "   https://myaccount.google.com/security"
echo ""

# 비밀번호 입력 받기
read -sp "생성한 16자리 앱 비밀번호를 입력하세요 (공백 제거): " APP_PASSWORD
echo ""

if [ -z "$APP_PASSWORD" ]; then
    echo "❌ 비밀번호가 입력되지 않았습니다."
    exit 1
fi

# 공백 제거
APP_PASSWORD=$(echo "$APP_PASSWORD" | tr -d ' ')

if [ ${#APP_PASSWORD} -ne 16 ]; then
    echo "⚠️  경고: 비밀번호가 16자리가 아닙니다 (현재: ${#APP_PASSWORD}자)"
    read -p "계속하시겠습니까? (y/n): " CONTINUE
    if [ "$CONTINUE" != "y" ]; then
        exit 1
    fi
fi

echo ""
echo "2단계: 환경변수 설정"
echo "----------------------------------------"

# 현재 세션에 설정
export SMTP_PASSWORD="$APP_PASSWORD"
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USER="cha.jiook@gmail.com"
export SMTP_FROM="cha.jiook@gmail.com"

echo "✅ 현재 세션에 환경변수 설정 완료"
echo ""

# 영구 설정 여부 확인
read -p "영구 설정 (~/.bashrc에 추가)도 하시겠습니까? (y/n): " PERMANENT

if [ "$PERMANENT" = "y" ]; then
    # 기존 설정 제거
    sed -i '/^export SMTP_PASSWORD=/d' ~/.bashrc
    sed -i '/^export SMTP_SERVER=/d' ~/.bashrc
    sed -i '/^export SMTP_PORT=/d' ~/.bashrc
    sed -i '/^export SMTP_USER=/d' ~/.bashrc
    sed -i '/^export SMTP_FROM=/d' ~/.bashrc
    
    # 새 설정 추가
    echo "" >> ~/.bashrc
    echo "# CENTaUR Email Settings" >> ~/.bashrc
    echo "export SMTP_PASSWORD=\"$APP_PASSWORD\"" >> ~/.bashrc
    echo "export SMTP_SERVER=\"smtp.gmail.com\"" >> ~/.bashrc
    echo "export SMTP_PORT=\"587\"" >> ~/.bashrc
    echo "export SMTP_USER=\"cha.jiook@gmail.com\"" >> ~/.bashrc
    echo "export SMTP_FROM=\"cha.jiook@gmail.com\"" >> ~/.bashrc
    
    echo "✅ ~/.bashrc에 영구 설정 추가 완료"
    echo ""
fi

# 테스트
echo "3단계: 이메일 테스트"
echo "----------------------------------------"
read -p "테스트 이메일을 전송하시겠습니까? (y/n): " TEST_EMAIL

if [ "$TEST_EMAIL" = "y" ]; then
    cd ~/git/CENTaUR
    python3 << EOF
import os
import sys
sys.path.insert(0, 'scripts')
from monitor_and_email import send_email

result = send_email(
    '[CENTaUR] SMTP 설정 테스트',
    '이메일 설정이 성공적으로 완료되었습니다!\n\n이제 10분마다 진행상황 리포트를 받으실 수 있습니다.'
)

if result:
    print("✅ 테스트 이메일 전송 성공!")
else:
    print("❌ 테스트 이메일 전송 실패 (로그 확인 필요)")
EOF
    echo ""
fi

echo "=========================================="
echo "설정 완료!"
echo "=========================================="
echo ""
echo "다음 단계:"
echo "  1. 모니터링 재시작 (환경변수 적용):"
echo "     cd ~/git/CENTaUR"
echo "     kill \$(cat logs/monitor_pid.txt) 2>/dev/null"
echo "     source ~/.bashrc"
echo "     python3 scripts/monitor_and_email.py > logs/monitor.log 2>&1 &"
echo "     echo \$! > logs/monitor_pid.txt"
echo ""
echo "  2. 또는 전체 작업 재시작:"
echo "     bash scripts/start_gpu_training.sh"
echo ""

