# 이메일 알림 설정 가이드

## 📧 Gmail 앱 비밀번호 설정

모니터링 스크립트가 10분마다 진행상황을 이메일로 전송하려면 Gmail 앱 비밀번호가 필요합니다.

### 1단계: Google 계정 2단계 인증 활성화

1. https://myaccount.google.com/security 접속
2. "2단계 인증" 섹션 찾기
3. 2단계 인증 활성화 (아직 안 했다면)

### 2단계: 앱 비밀번호 생성

1. https://myaccount.google.com/apppasswords 접속
2. "앱 선택" → "기타(맞춤 이름)" 선택
3. 이름 입력: "CENTaUR Monitor"
4. "만들기" 클릭
5. **16자리 비밀번호 복사** (예: `abcd efgh ijkl mnop`)

### 3단계: 서버에 환경변수 설정

서버에 SSH 접속 후:

```bash
ssh dgx-spark
cd ~/git/CENTaUR

# 환경변수 설정 (현재 세션용)
export SMTP_PASSWORD="your_16_digit_app_password"  # 공백 제거하고 입력
export SMTP_SERVER="smtp.gmail.com"
export SMTP_PORT="587"
export SMTP_USER="cha.jiook@gmail.com"
export SMTP_FROM="cha.jiook@gmail.com"

# 영구 설정 (선택사항)
echo 'export SMTP_PASSWORD="your_16_digit_app_password"' >> ~/.bashrc
echo 'export SMTP_SERVER="smtp.gmail.com"' >> ~/.bashrc
echo 'export SMTP_PORT="587"' >> ~/.bashrc
echo 'export SMTP_USER="cha.jiook@gmail.com"' >> ~/.bashrc
echo 'export SMTP_FROM="cha.jiook@gmail.com"' >> ~/.bashrc
source ~/.bashrc
```

### 4단계: 모니터링 재시작

환경변수 설정 후 모니터링을 재시작:

```bash
# 기존 모니터링 중지
kill $(cat logs/monitor_pid.txt)

# 새로 시작 (환경변수 적용됨)
cd ~/git/CENTaUR
source ~/.bashrc  # 환경변수 로드
python3 scripts/monitor_and_email.py > logs/monitor.log 2>&1 &
echo $! > logs/monitor_pid.txt
```

## 📬 이메일 수신 확인

설정이 완료되면:
- **10분마다** `cha.jiook@gmail.com`으로 진행상황 리포트가 전송됩니다
- 첫 번째 이메일은 약 10분 후 도착합니다

## 🔍 문제 해결

### 이메일이 오지 않는 경우

1. **환경변수 확인**
   ```bash
   echo $SMTP_PASSWORD  # 비어있으면 안됨
   ```

2. **로그 확인**
   ```bash
   tail -50 logs/monitor.log
   ```
   - "✅ Email sent" 메시지가 보여야 함
   - 오류가 있으면 확인

3. **수동 테스트**
   ```bash
   python3 -c "
   import os
   from scripts.monitor_and_email import send_email
   send_email('테스트', '이메일 테스트입니다.')
   "
   ```

### 앱 비밀번호가 작동하지 않는 경우

- 공백 제거 확인: `abcd efgh ijkl mnop` → `abcdefghijklmnop`
- 2단계 인증이 활성화되어 있는지 확인
- 최근 생성한 앱 비밀번호인지 확인 (오래된 것은 만료될 수 있음)

## 📝 참고사항

- **비밀번호 없이 실행**: 이메일 대신 `logs/email_*.txt` 파일로 저장됨
- **보안**: 앱 비밀번호는 일반 비밀번호와 다르며, 특정 앱에서만 사용 가능
- **알림 주기**: 기본 10분 (600초), `monitor_and_email.py`에서 변경 가능

