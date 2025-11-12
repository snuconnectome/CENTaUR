# Cursor 설치 가이드 (dgx-spark)

## ⚠️ 중요 사항

Cursor는 주로 **데스크톱 애플리케이션**이므로 서버에 직접 설치하는 것은 제한적입니다.

## 🎯 권장 방법: 원격 개발 설정

### 방법 1: 로컬 Cursor에서 원격 개발 (권장)

1. **로컬 머신에 Cursor 설치**
   - https://cursor.sh 에서 다운로드
   - macOS/Windows/Linux 데스크톱 버전 설치

2. **SSH 확장 설치**
   - Cursor에서 "Remote - SSH" 확장 설치
   - 또는 Cursor는 기본적으로 SSH 연결 지원

3. **원격 서버 연결**
   - Cursor에서 `Cmd/Ctrl + Shift + P`
   - "Remote-SSH: Connect to Host" 선택
   - `dgx-spark` 입력 또는 SSH 설정 추가

### 방법 2: VS Code Server 설치 (서버에 직접 설치)

```bash
# SSH로 dgx-spark 접속
ssh dgx-spark

# VS Code Server 설치
cd ~
mkdir -p ~/.vscode-server
curl -fsSL https://code.visualstudio.com/sha/download?build=stable&os=cli-alpine-x64 -o /tmp/vscode-cli.tar.gz
tar -xzf /tmp/vscode-cli.tar.gz -C ~/.vscode-server/
```

### 방법 3: Cursor 직접 설치 (GUI 환경 필요)

서버에 X11 포워딩이나 GUI 환경이 있는 경우:

```bash
# Ubuntu/Debian
wget https://cursor.sh/download/linux -O cursor.deb
sudo dpkg -i cursor.deb
sudo apt-get install -f  # 의존성 해결
```

## 📋 현재 시스템 정보

- **OS**: Ubuntu 24.04.3 LTS
- **아키텍처**: aarch64 (ARM64)
- **서버**: dgx-spark (원격 서버)

## 🔧 설치 확인

```bash
# Cursor 확인
which cursor
cursor --version

# VS Code Server 확인
which code
code --version
```

## 💡 추천 방법

**가장 권장하는 방법은 방법 1 (로컬 Cursor + 원격 개발)**입니다:
- 서버 리소스 절약
- 로컬 Cursor의 모든 기능 사용 가능
- 원격 파일 편집 및 실행 가능

## 📚 참고 자료

- Cursor 공식 사이트: https://cursor.sh
- VS Code Remote SSH: https://code.visualstudio.com/docs/remote/ssh

