# NGC PyTorch 설치 가이드 (NVIDIA 공식 문서 기반)

## 시스템 정보
- 아키텍처: aarch64 (ARM64)
- OS: Ubuntu 24.04.3 LTS
- GPU: NVIDIA GB10 (CUDA capability 12.1)
- Docker: 설치됨 (28.3.3)

## NGC PyTorch 설치 방법

### 방법 1: Docker 컨테이너 사용 (권장)

#### 1단계: NGC 계정 및 API 키
1. NGC 웹사이트 방문: https://ngc.nvidia.com/
2. 계정 생성 (무료)
3. Setup > API Key에서 API 키 생성

#### 2단계: NGC 레지스트리 로그인
```bash
docker login nvcr.io
# 사용자명: $oauthtoken
# 비밀번호: API 키
```

#### 3단계: PyTorch 컨테이너 다운로드
```bash
# 최신 버전 (2024년 8월)
docker pull nvcr.io/nvidia/pytorch:24.08-py3

# 또는 특정 버전
docker pull nvcr.io/nvidia/pytorch:24.01-py3
```

#### 4단계: 컨테이너 실행
```bash
# 기본 실행
docker run -it --gpus=all nvcr.io/nvidia/pytorch:24.08-py3

# 작업 디렉토리 마운트
docker run -it --gpus=all \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/pytorch:24.08-py3

# 포트 포워딩 및 환경변수 설정
docker run -it --gpus=all \
  -v $(pwd):/workspace \
  -p 8888:8888 \
  -e NVIDIA_VISIBLE_DEVICES=all \
  nvcr.io/nvidia/pytorch:24.08-py3
```

### 방법 2: NGC CLI 사용 (DGX Spark 권장)

#### 1단계: NGC CLI 설치 (ARM64)
```bash
# NGC CLI 다운로드 페이지에서 ARM64 Linux 버전 다운로드
# https://org.ngc.nvidia.com/setup/installers/cli

# 설치
wget https://ngc.nvidia.com/downloads/ngccli_arm64.zip
unzip ngccli_arm64.zip
sudo ./ngc-cli/ngc --version
```

#### 2단계: NGC CLI 설정
```bash
ngc config set
# API 키 입력
```

#### 3단계: 컨테이너 다운로드 및 실행
```bash
ngc registry image pull nvcr.io/nvidia/pytorch:24.08-py3
```

## DGX Spark 특별 고려사항

1. **ARM64 아키텍처**: ARM64용 컨테이너 사용 필요
2. **Blackwell GPU**: 최신 컨테이너 버전 사용 권장
3. **NGC CLI**: ARM64 버전 설치 필요

## 컨테이너 내부에서 작업

컨테이너 실행 후:
```bash
# PyTorch 버전 확인
python -c "import torch; print(torch.__version__)"

# CUDA 확인
python -c "import torch; print(torch.cuda.is_available())"

# GPU 확인
nvidia-smi
```

## 참고 문서

- [NGC — DGX Spark 사용자 가이드](https://docs.nvidia.com/dgx/dgx-spark/ngc.html)
- [NGC 컨테이너 레지스트리](https://catalog.ngc.nvidia.com/containers)
- [PyTorch 컨테이너 문서](https://docs.nvidia.com/deeplearning/frameworks/pytorch-release-notes/index.html)

---

**작성일**: 2025-11-09
**기반**: NVIDIA 공식 문서

