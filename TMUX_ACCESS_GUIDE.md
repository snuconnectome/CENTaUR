# tmux 세션 접속 및 문제 해결 가이드

## 🔌 접속 방법

### 1. SSH 접속
```bash
ssh dgx-spark
```

### 2. tmux 세션 확인
```bash
tmux ls
```

현재 세션:
- `centaur`: CENTaUR 작업 세션
- `business`: 비즈니스 관련 세션
- `coin`: 코인 관련 세션

### 3. centaur 세션에 attach
```bash
tmux attach -t centaur
```

또는 간단히:
```bash
tmux a -t centaur
```

### 4. 세션에서 나가기 (detach)
```
Ctrl+b, d
```
(먼저 Ctrl+b를 누르고, 그 다음 d를 누름)

## 🔧 현재 문제 해결

### bitsandbytes 라이브러리 오류
**오류 메시지:**
```
OSError: libnvJitLink.so.12: cannot open shared object file
```

### 해결 방법

#### 방법 1: 환경 변수 설정 (임시)
```bash
export CUDA_HOME=/usr/local/cuda-13.0
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH
```

#### 방법 2: 심볼릭 링크 생성
```bash
sudo ln -s /usr/local/cuda-13.0/targets/sbsa-linux/lib/libnvJitLink.so.13 /usr/local/cuda-13.0/targets/sbsa-linux/lib/libnvJitLink.so.12
```

#### 방법 3: .bashrc에 영구 설정
```bash
echo 'export CUDA_HOME=/usr/local/cuda-13.0' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
```

## 📊 현재 실행 중인 작업 확인

### 프로세스 확인
```bash
ps aux | grep train_qwen25
ps aux | grep extract_centaur
ps aux | grep monitor_and_email
```

### 로그 확인
```bash
# Fine-tuning 로그
tail -f ~/git/CENTaUR/logs/qwen25-32b-qlora_finetuning.log

# 모니터링 로그
tail -f ~/git/CENTaUR/logs/monitor.log
```

### GPU 상태 확인
```bash
nvidia-smi
watch -n 1 nvidia-smi
```

## 🚀 Fine-tuning 재시작

### 1. 기존 프로세스 종료
```bash
cd ~/git/CENTaUR
pkill -f train_qwen25
```

### 2. 환경 변수 설정 후 재시작
```bash
cd ~/git/CENTaUR
source venv/bin/activate

# CUDA 라이브러리 경로 설정
export CUDA_HOME=/usr/local/cuda-13.0
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

# Fine-tuning 시작
bash scripts/start_centaur_finetuning.sh qwen25
```

## 📁 주요 디렉토리

- 작업 디렉토리: `~/git/CENTaUR`
- 로그: `~/git/CENTaUR/logs/`
- 모델: `~/git/CENTaUR/models/`
- 프로파일: `~/git/CENTaUR/profiles/`

## 💡 유용한 명령어

### tmux 명령어
- `Ctrl+b, c`: 새 창 생성
- `Ctrl+b, n`: 다음 창
- `Ctrl+b, p`: 이전 창
- `Ctrl+b, %`: 세로 분할
- `Ctrl+b, "`: 가로 분할
- `Ctrl+b, d`: 세션 detach

### 작업 관리
```bash
# 백그라운드 실행
nohup command > log.txt 2>&1 &

# 프로세스 확인
ps aux | grep python

# 프로세스 종료
kill <PID>
pkill -f <pattern>
```

## ⚠️ 주의사항

1. **CUDA 버전 불일치**: PyTorch는 CUDA 12.4를 사용하지만 시스템에는 CUDA 13.0이 설치되어 있음
2. **GPU 호환성**: NVIDIA GB10 (sm_121)는 현재 PyTorch와 호환되지 않음
3. **bitsandbytes**: CUDA 12.x 라이브러리를 찾지만 시스템에는 CUDA 13.0만 있음

## 🔍 문제 진단

### bitsandbytes 진단
```bash
cd ~/git/CENTaUR
source venv/bin/activate
python -m bitsandbytes
```

### CUDA 라이브러리 확인
```bash
ldconfig -p | grep nvJitLink
find /usr/local/cuda* -name 'libnvJitLink.so*'
```

### PyTorch CUDA 확인
```bash
python -c "import torch; print(torch.__version__); print(torch.version.cuda)"
```

