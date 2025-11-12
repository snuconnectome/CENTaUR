# NGC PyTorch 설치 가이드 (실행 가능한 버전)

## 빠른 시작

### 1. NGC API 키 준비
- https://ngc.nvidia.com/ 접속
- Setup > API Key에서 생성

### 2. 설치 스크립트 실행
```bash
./install_ngc_pytorch.sh
```

또는 수동으로:

### 수동 설치 절차

#### 1단계: Docker 권한 확인
```bash
# Docker 권한이 없다면
sudo usermod -aG docker $USER
# 로그아웃 후 재로그인
```

#### 2단계: NGC 로그인
```bash
docker login nvcr.io
# 사용자명: $oauthtoken
# 비밀번호: NGC API 키
```

#### 3단계: 컨테이너 다운로드
```bash
docker pull nvcr.io/nvidia/pytorch:24.08-py3
```

#### 4단계: 컨테이너 실행
```bash
# 기본 실행
docker run -it --gpus=all nvcr.io/nvidia/pytorch:24.08-py3

# 작업 디렉토리 마운트
docker run -it --gpus=all \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/pytorch:24.08-py3
```

## 컨테이너 내부에서 작업

```bash
# PyTorch 버전 확인
python -c "import torch; print(torch.__version__)"

# CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"
print(torch.cuda.get_device_name(0))"

# GPU 확인
nvidia-smi
```

## 현재 프로젝트와 연동

```bash
# 현재 디렉토리를 컨테이너의 /workspace에 마운트
docker run -it --gpus=all \
  -v /home/juke/git/CENTaUR:/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:24.08-py3

# 컨테이너 내부에서
cd /workspace
source venv/bin/activate  # 또는 컨테이너의 Python 사용
python scripts/extract_centaur_features.py --model exaone35-base ...
```

## 참고

- 컨테이너 태그: `24.08-py3` (2024년 8월 버전)
- 최신 버전 확인: https://catalog.ngc.nvidia.com/containers
- 공식 문서: https://docs.nvidia.com/dgx/dgx-spark/ngc.html

---

**작성일**: 2025-11-09

