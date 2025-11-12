#!/bin/bash
# SMTP 이메일 설정 스크립트

echo "=== CENTaUR 모니터링 이메일 설정 ==="
echo ""

# 기본값
DEFAULT_EMAIL="cha.jiook@gmail.com"
DEFAULT_SMTP="smtp.gmail.com"
DEFAULT_PORT="587"

read -p "수신 이메일 주소 [$DEFAULT_EMAIL]: " EMAIL
EMAIL=${EMAIL:-$DEFAULT_EMAIL}

read -p "SMTP 서버 [$DEFAULT_SMTP]: " SMTP_SERVER
SMTP_SERVER=${SMTP_SERVER:-$DEFAULT_SMTP}

read -p "SMTP 포트 [$DEFAULT_PORT]: " SMTP_PORT
SMTP_PORT=${SMTP_PORT:-$DEFAULT_PORT}

read -p "SMTP 사용자명 (보통 이메일 주소) [$EMAIL]: " SMTP_USER
SMTP_USER=${SMTP_USER:-$EMAIL}

echo ""
echo "⚠️  비밀번호는 환경변수로만 설정하세요 (보안상 화면에 표시되지 않습니다)"
read -sp "SMTP 비밀번호 (또는 Gmail 앱 비밀번호): " SMTP_PASSWORD
echo ""

# 환경변수 설정
export SMTP_SERVER
export SMTP_PORT
export SMTP_USER
export SMTP_PASSWORD
export SMTP_FROM="$SMTP_USER"

# 테스트 이메일 발송
echo ""
echo "테스트 이메일 발송 중..."
python3 << EOF
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import os

try:
    msg = MIMEMultipart()
    msg['From'] = os.getenv('SMTP_FROM')
    msg['To'] = '$EMAIL'
    msg['Subject'] = '[CENTaUR] 이메일 설정 테스트'
    
    body = '이메일 설정이 성공적으로 완료되었습니다!\n\n모니터링 스크립트가 작업 완료 시 이메일을 발송합니다.'
    msg.attach(MIMEText(body, 'plain', 'utf-8'))
    
    server = smtplib.SMTP('$SMTP_SERVER', $SMTP_PORT)
    server.starttls()
    server.login('$SMTP_USER', os.getenv('SMTP_PASSWORD'))
    server.send_message(msg)
    server.quit()
    
    print("✅ 테스트 이메일 발송 성공!")
    print("📧 $EMAIL 로 이메일을 확인하세요.")
except Exception as e:
    print(f"❌ 이메일 발송 실패: {e}")
    print("💡 Gmail 사용 시: 앱 비밀번호를 생성하여 사용하세요")
    print("   https://myaccount.google.com/apppasswords")
EOF

# 환경변수 영구 설정 안내
echo ""
echo "=== 환경변수 영구 설정 ==="
echo "다음 명령어를 ~/.bashrc 또는 ~/.zshrc에 추가하세요:"
echo ""
echo "export SMTP_SERVER=\"$SMTP_SERVER\""
echo "export SMTP_PORT=\"$SMTP_PORT\""
echo "export SMTP_USER=\"$SMTP_USER\""
echo "export SMTP_PASSWORD=\"your_app_password\"  # 실제 비밀번호로 변경"
echo "export SMTP_FROM=\"$SMTP_USER\""
echo ""

