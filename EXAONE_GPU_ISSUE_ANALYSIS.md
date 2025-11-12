# EXAONE Feature Extraction GPU 사용 불가 원인 분석

## 🔍 문제 상황

EXAONE Feature Extraction이 CPU로 실행되고 있습니다:
- **CPU 사용률**: 751% (멀티코어 사용)
- **GPU 사용률**: 2% (거의 사용 안 함)
- **실행 시간**: 매우 느림 (약 4분/샘플)

## 📋 코드 분석

### 스크립트 설정 (`extract_centaur_features.py`)

```python
# Line 123-125: GPU 사용 설정
model_kwargs = {
    "device_map": "auto",  # GPU 자동 배치
    "torch_dtype": torch.bfloat16
}

# Line 205: 입력을 모델 디바이스로 이동
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
```

**결론**: 스크립트는 GPU를 사용하도록 설정되어 있습니다.

## ⚠️ 실제 원인

### 1. PyTorch와 GPU 호환성 문제

**NVIDIA GB10 GPU 정보:**
- CUDA Capability: **sm_121** (Blackwell 아키텍처)
- 현재 PyTorch 설치가 지원하는 capability: **sm_50, sm_80, sm_86, sm_89, sm_90, sm_90a**

**오류 메시지:**
```
NVIDIA GB10 with CUDA capability sm_121 is not compatible with the current PyTorch installation.
The current PyTorch install supports CUDA capabilities sm_50 sm_80 sm_86 sm_89 sm_90 sm_90a.
```

**결과**: PyTorch가 GPU를 인식하지만 실제 커널 실행이 불가능하여 CPU로 fallback됩니다.

### 2. bitsandbytes 라이브러리 오류

**오류:**
```
OSError: libnvJitLink.so.12: cannot open shared object file
```

**원인:**
- bitsandbytes는 CUDA 12.x용 라이브러리(`libnvJitLink.so.12`)를 찾음
- 시스템에는 CUDA 13.0만 설치되어 있음 (`libnvJitLink.so.13`만 존재)
- 라이브러리 경로가 설정되지 않음

**결과**: Quantization이 실패하고, 모델이 CPU로 로드됩니다.

### 3. 모델 로딩 시 CPU Fallback

`device_map="auto"`가 설정되어 있어도:
1. PyTorch가 GPU 커널을 실행할 수 없음
2. bitsandbytes가 작동하지 않음
3. Transformers 라이브러리가 자동으로 CPU로 fallback
4. 모델이 CPU에 로드됨

## 🔧 해결 방법

### 방법 1: CUDA 라이브러리 경로 설정 (임시 해결)

```bash
export CUDA_HOME=/usr/local/cuda-13.0
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/targets/sbsa-linux/lib:/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

# 심볼릭 링크 생성 (sudo 필요)
sudo ln -s /usr/local/cuda-13.0/targets/sbsa-linux/lib/libnvJitLink.so.13 \
           /usr/local/cuda-13.0/targets/sbsa-linux/lib/libnvJitLink.so.12
```

**효과**: bitsandbytes 오류는 해결되지만, PyTorch 호환성 문제는 남아있습니다.

### 방법 2: PyTorch 재설치 (근본 해결)

**문제**: 현재 PyTorch 버전이 sm_121을 지원하지 않음

**해결책**:
1. sm_121을 지원하는 PyTorch 버전 설치 (최신 nightly 또는 특정 버전)
2. 또는 CPU 모드로 명시적 실행 (현재 상태 유지)

**주의**: NVIDIA GB10 (Blackwell)는 매우 최신 GPU이므로, PyTorch 공식 릴리즈가 아직 지원하지 않을 수 있습니다.

### 방법 3: CPU 모드로 명시적 실행 (현재 상태)

현재는 CPU로 실행되고 있으므로, 명시적으로 CPU 모드로 설정:

```python
# extract_centaur_features.py 수정
model_kwargs = {
    "device_map": "cpu",  # 명시적 CPU 사용
    "torch_dtype": torch.bfloat16
}
```

**장점**: 예측 가능한 동작
**단점**: 매우 느림 (현재 상태)

## 📊 현재 상태 요약

| 항목 | 상태 | 설명 |
|------|------|------|
| 스크립트 설정 | ✅ GPU 사용 설정 | `device_map="auto"` |
| PyTorch CUDA 지원 | ❌ sm_121 미지원 | 현재 버전 한계 |
| bitsandbytes | ❌ 라이브러리 오류 | libnvJitLink.so.12 누락 |
| 실제 실행 | ⚠️ CPU로 실행 | 자동 fallback |
| 성능 | ❌ 매우 느림 | CPU 제한 |

## 💡 권장 사항

1. **단기**: CPU 모드로 명시적 실행 (현재 상태 유지)
2. **중기**: CUDA 라이브러리 경로 설정으로 bitsandbytes 오류 해결
3. **장기**: sm_121을 지원하는 PyTorch 버전 대기 또는 설치

## 🔗 참고 자료

- PyTorch CUDA Capability: https://pytorch.org/get-started/locally/
- bitsandbytes Issues: https://github.com/bitsandbytes-foundation/bitsandbytes/issues
- NVIDIA GB10 (Blackwell): 최신 아키텍처로 PyTorch 지원이 제한적

