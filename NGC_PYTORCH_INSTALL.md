# NGC PyTorch 설치 가이드

## 현재 환경
- GPU: NVIDIA GB10 (CUDA capability 12.1)
- 문제: 현재 PyTorch가 CUDA 12.1을 완전히 지원하지 않음

## NGC PyTorch 설치 방법

### 방법 1: NGC PyTorch Wheel 직접 설치 (권장)

NGC에서 제공하는 최신 PyTorch wheel 파일을 설치합니다.

```bash
# NGC PyTorch 최신 버전 확인 및 설치
pip install --upgrade pip

# NGC PyTorch 설치 (CUDA 12.1 지원)
pip install nvidia-pytorch --index-url https://pypi.ngc.nvidia.com

# 또는 특정 버전 설치
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### 방법 2: NGC Container 사용

NGC 컨테이너를 사용하여 완전한 환경을 구성합니다.

```bash
# NGC CLI 설치 (필요시)
pip install nvidia-pyindex

# NGC 컨테이너 실행
docker run --gpus all -it --rm \
  -v $(pwd):/workspace \
  nvcr.io/nvidia/pytorch:24.01-py3
```

### 방법 3: Conda를 통한 설치

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

## 설치 후 확인

```bash
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'CUDA capability: {torch.cuda.get_device_capability(0) if torch.cuda.is_available() else \"N/A\"}')
"
```

## 주의사항

1. **기존 PyTorch 제거**: 새로 설치하기 전에 기존 PyTorch를 제거하는 것이 좋습니다.
2. **가상환경**: 현재 venv를 사용 중이므로 venv 내에서 설치하세요.
3. **의존성**: 다른 패키지와의 호환성을 확인하세요.

## 설치 후 작업 재시작

PyTorch 설치 후:
1. Qwen2.5-32B Base LOO CV 재시작
2. DeepSeek-R1 Base LOO CV 재시작
3. EXAONE-3.5-32B Feature Extraction은 계속 실행 중

---

**마지막 업데이트**: 2025-11-08

