# PyTorch 설치 출처 확인 결과

## 현재 설치된 PyTorch

- **버전**: 2.5.1
- **CUDA 버전**: 12.4
- **출처**: PyTorch 공식 사이트 (`download.pytorch.org/whl/cu124`)
- **NGC 여부**: ❌ **아니요**

## 설치 이력

현재 설치된 PyTorch는 다음 명령어로 설치되었습니다:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

이것은 **PyTorch 공식 사이트**에서 설치한 것이며, **NGC가 아닙니다**.

## NGC PyTorch와의 차이

### PyTorch 공식 사이트 (현재 설치됨)
- 출처: `download.pytorch.org`
- 일반적인 PyTorch 배포판
- CUDA 12.1 완전 지원 안 됨

### NGC PyTorch
- 출처: `pypi.ngc.nvidia.com` 또는 NGC 컨테이너
- NVIDIA가 최적화한 버전
- 최신 GPU 지원 가능

## NGC PyTorch 설치 방법

### 방법 1: NGC 컨테이너 사용 (권장)
```bash
docker pull nvcr.io/nvidia/pytorch:24.01-py3
docker run --gpus all -it nvcr.io/nvidia/pytorch:24.01-py3
```

### 방법 2: NGC PyPI 인덱스 사용
```bash
pip install nvidia-pytorch --index-url https://pypi.ngc.nvidia.com
```

### 방법 3: NGC CLI 사용
```bash
# NGC CLI 설치
pip install nvidia-pyindex

# NGC에서 패키지 설치
pip install nvidia-pytorch
```

## 권장 사항

CUDA 12.1 (GB10) 완전 지원을 위해서는:
1. **NGC 컨테이너 사용** (가장 권장)
2. **PyTorch Nightly 빌드** 시도
3. 현재 버전으로 CPU fallback 사용

---

**확인 날짜**: 2025-11-09

