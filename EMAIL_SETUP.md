# CENTaUR 실험 모니터링 이메일 설정 가이드

## 이메일 발송 방법

모니터링 스크립트는 다음 순서로 이메일 발송을 시도합니다:

1. **mail 명령어** (시스템 기본)
2. **sendmail 명령어** (시스템 기본)
3. **SMTP (Python)** ← **권장 방법**
4. **파일 저장** (수동 발송)

---

## SMTP 설정 방법

### Gmail 사용 시

1. **앱 비밀번호 생성**
   - Google 계정 설정 → 보안 → 2단계 인증 활성화
   - 앱 비밀번호 생성: https://myaccount.google.com/apppasswords
   - 생성된 16자리 비밀번호 복사

2. **환경변수 설정**
   ```bash
   export SMTP_SERVER="smtp.gmail.com"
   export SMTP_PORT="587"
   export SMTP_USER="cha.jiook@gmail.com"
   export SMTP_PASSWORD="your_16_digit_app_password"
   export SMTP_FROM="cha.jiook@gmail.com"
   ```

3. **영구 설정 (선택사항)**
   ```bash
   # ~/.bashrc 또는 ~/.zshrc에 추가
   echo 'export SMTP_PASSWORD="your_app_password"' >> ~/.bashrc
   source ~/.bashrc
   ```

### 다른 SMTP 서버 사용 시

```bash
# 예: 네이버 메일
export SMTP_SERVER="smtp.naver.com"
export SMTP_PORT="587"
export SMTP_USER="your_email@naver.com"
export SMTP_PASSWORD="your_password"
export SMTP_FROM="your_email@naver.com"

# 예: Outlook/Hotmail
export SMTP_SERVER="smtp-mail.outlook.com"
export SMTP_PORT="587"
export SMTP_USER="your_email@outlook.com"
export SMTP_PASSWORD="your_password"
export SMTP_FROM="your_email@outlook.com"
```

---

## 테스트 방법

```bash
# 환경변수 설정 후
source venv/bin/activate
python -c "
from monitor_experiments import send_email_simple
send_email_simple('테스트', '이메일 발송 테스트입니다.')
"
```

---

## 보안 주의사항

⚠️ **앱 비밀번호는 절대 코드에 하드코딩하지 마세요!**

- 환경변수 사용 권장
- `.env` 파일 사용 시 `.gitignore`에 추가
- 파일 권한 설정: `chmod 600 .env`

---

## 문제 해결

### "SMTP 인증 실패" 오류
- Gmail: 앱 비밀번호 사용 확인 (일반 비밀번호 X)
- 2단계 인증 활성화 확인
- "보안 수준이 낮은 앱의 액세스" 허용 (구식 방법, 비권장)

### "연결 시간 초과" 오류
- 방화벽/네트워크 확인
- SMTP 포트 (587, 465) 접근 가능 여부 확인

### 이메일이 스팸으로 분류되는 경우
- 발신자 주소를 신뢰할 수 있는 주소로 설정
- 이메일 제목/본문에 스팸 키워드 제거

---

**마지막 업데이트**: 2025-11-08

