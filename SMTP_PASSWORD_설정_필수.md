# ⚠️ SMTP_PASSWORD 설정 필요

## 현재 상태

모니터링은 실행 중이지만, **실제 Gmail 앱 비밀번호**가 필요합니다.

입력하신 `abcdefghijklmnop`는 예시 비밀번호입니다. 실제 Gmail 앱 비밀번호를 생성해야 합니다.

## 🔑 실제 Gmail 앱 비밀번호 생성 방법

### 1단계: 2단계 인증 확인
- https://myaccount.google.com/security
- "2단계 인증"이 **활성화**되어 있어야 합니다

### 2단계: 앱 비밀번호 생성
1. https://myaccount.google.com/apppasswords 접속
2. "앱 선택" → **"기타(맞춤 이름)"** 선택
3. 이름 입력: `CENTaUR Monitor`
4. "만들기" 클릭
5. **16자리 비밀번호 복사** (예: `abcd efgh ijkl mnop`)

### 3단계: 서버에 설정

**방법 1: 자동 설정 스크립트**
```bash
ssh dgx-spark
cd ~/git/CENTaUR
bash scripts/setup_smtp_password.sh
# 생성한 실제 비밀번호 입력
```

**방법 2: 수동 설정**
```bash
ssh dgx-spark
cd ~/git/CENTaUR

# 실제 비밀번호로 교체 (공백 제거!)
export SMTP_PASSWORD="실제생성한16자리비밀번호"

# 모니터링 재시작
bash scripts/restart_monitor_with_smtp.sh
```

## ✅ 설정 확인

설정 후 테스트:
```bash
cd ~/git/CENTaUR
python3 << 'EOF'
import os
import sys
os.environ['SMTP_PASSWORD'] = '실제비밀번호'  # 여기에 실제 비밀번호 입력
os.environ['SMTP_SERVER'] = 'smtp.gmail.com'
os.environ['SMTP_PORT'] = '587'
os.environ['SMTP_USER'] = 'cha.jiook@gmail.com'
os.environ['SMTP_FROM'] = 'cha.jiook@gmail.com'
sys.path.insert(0, 'scripts')
from monitor_and_email import send_email
result = send_email('테스트', '이메일 테스트입니다.')
print('✅ 성공!' if result else '❌ 실패')
EOF
```

## 📧 현재 상태

- ✅ 모니터링 실행 중 (PID: 965572)
- ❌ 이메일 전송 실패 (잘못된 비밀번호)
- 📁 대신 파일로 저장됨: `logs/email_*.txt`

**실제 Gmail 앱 비밀번호를 설정하면 이메일로 전송됩니다!**

