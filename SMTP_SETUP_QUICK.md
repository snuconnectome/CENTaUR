# SMTP_PASSWORD 빠른 설정 가이드

## 🚀 빠른 설정 (3단계)

### 1단계: Gmail 앱 비밀번호 생성

1. **2단계 인증 확인/활성화**
   - https://myaccount.google.com/security
   - "2단계 인증" 활성화 (없다면)

2. **앱 비밀번호 생성**
   - https://myaccount.google.com/apppasswords
   - "앱 선택" → **"기타(맞춤 이름)"** 선택
   - 이름: `CENTaUR Monitor`
   - "만들기" 클릭
   - **16자리 비밀번호 복사** (예: `abcd efgh ijkl mnop`)

### 2단계: 서버에서 설정

**방법 A: 자동 설정 스크립트 (권장)**
```bash
ssh dgx-spark
cd ~/git/CENTaUR
bash scripts/setup_smtp_password.sh
```
스크립트가 안내에 따라 진행합니다.

**방법 B: 수동 설정**
```bash
ssh dgx-spark
cd ~/git/CENTaUR

# 현재 세션에만 설정 (임시)
export SMTP_PASSWORD="your_16_digit_password"  # 공백 제거!
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USER="cha.jiook@gmail.com"
export SMTP_FROM="cha.jiook@gmail.com"

# 영구 설정 (선택사항)
echo 'export SMTP_PASSWORD="your_16_digit_password"' >> ~/.bashrc
echo 'export SMTP_SERVER="smtp.gmail.com"' >> ~/.bashrc
echo 'export SMTP_PORT="587"' >> ~/.bashrc
echo 'export SMTP_USER="cha.jiook@gmail.com"' >> ~/.bashrc
echo 'export SMTP_FROM="cha.jiook@gmail.com"' >> ~/.bashrc
source ~/.bashrc
```

### 3단계: 모니터링 재시작

```bash
cd ~/git/CENTaUR

# 기존 모니터링 중지
kill $(cat logs/monitor_pid.txt) 2>/dev/null

# 환경변수 로드 (영구 설정했다면)
source ~/.bashrc

# 모니터링 재시작
python3 scripts/monitor_and_email.py > logs/monitor.log 2>&1 &
echo $! > logs/monitor_pid.txt

# 확인
tail -f logs/monitor.log
```

## ✅ 확인 방법

```bash
# 환경변수 확인
echo $SMTP_PASSWORD  # 비어있으면 안됨!

# 모니터링 로그 확인
tail -20 logs/monitor.log
# "✅ Email sent" 메시지가 보여야 함

# 수동 테스트
python3 -c "
import os
import sys
sys.path.insert(0, 'scripts')
from monitor_and_email import send_email
send_email('테스트', '이메일 테스트입니다.')
"
```

## ⚠️ 주의사항

1. **공백 제거**: Gmail에서 복사한 비밀번호는 `abcd efgh ijkl mnop` 형식인데, 사용할 때는 공백을 제거해야 합니다: `abcdefghijklmnop`

2. **16자리 확인**: 정확히 16자리여야 합니다.

3. **2단계 인증 필수**: 앱 비밀번호를 생성하려면 2단계 인증이 활성화되어 있어야 합니다.

4. **보안**: 앱 비밀번호는 일반 비밀번호와 다르며, 특정 앱에서만 사용 가능합니다. 안전하게 관리하세요.

## 🔧 문제 해결

### 이메일이 오지 않는 경우

1. **환경변수 확인**
   ```bash
   echo $SMTP_PASSWORD
   # 비어있으면 설정이 안된 것
   ```

2. **로그 확인**
   ```bash
   tail -50 logs/monitor.log
   # 오류 메시지 확인
   ```

3. **Gmail 보안 설정 확인**
   - "보안 수준이 낮은 앱의 액세스" 허용 (필요시)
   - 또는 앱 비밀번호 재생성

### "Authentication failed" 오류

- 앱 비밀번호가 올바른지 확인 (공백 제거)
- 2단계 인증이 활성화되어 있는지 확인
- 최근 생성한 앱 비밀번호인지 확인

