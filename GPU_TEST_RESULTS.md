# EXAONE-3.5-32B GPU 테스트 결과

## 테스트 결과 요약

### ✅ 성공한 부분
- CUDA 사용 가능: True
- 기본 GPU 연산: 성공 (torch.matmul 등)
- Tokenizer 로딩: 성공
- Quantization 설정 생성: 성공

### ❌ 문제점

1. **CUDA Capability 호환성 경고**
   ```
   NVIDIA GB10 with CUDA capability sm_121 is not compatible with the current PyTorch installation.
   The current PyTorch install supports CUDA capabilities sm_50 sm_80 sm_86 sm_89 sm_90 sm_90a.
   ```

2. **GPU 메모리 할당 실패**
   ```
   CUDA error: no kernel image is available for execution on the device
   ```
   - PyTorch가 CUDA 12.1을 완전히 지원하지 않아 발생

3. **bitsandbytes 라이브러리 오류**
   ```
   libnvJitLink.so.12: cannot open shared object file: No such file or directory
   ```
   - CUDA 라이브러리 의존성 문제

4. **모델 로딩 실패**
   - Quantization 사용 시 GPU 메모리 부족 또는 호환성 문제

## 해결 방안

### 옵션 1: CPU로 실행 (권장)
- 느리지만 안정적으로 동작
- Quantization 없이 실행 가능
- CPU offload 옵션 사용

### 옵션 2: PyTorch Nightly 빌드
```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124
```

### 옵션 3: bitsandbytes 재설치
```bash
pip uninstall bitsandbytes
pip install bitsandbytes
```

### 옵션 4: CPU Offload 사용
```python
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto",
    llm_int8_enable_fp32_cpu_offload=True,  # CPU offload 활성화
    trust_remote_code=True
)
```

## 현재 상태

- **기본 GPU 연산**: ✅ 작동
- **모델 로딩 (quantization)**: ❌ 실패
- **CPU fallback**: ✅ 가능

## 권장 사항

EXAONE-3.5-32B Feature Extraction은 현재 **CPU로 실행**하는 것이 가장 안정적입니다.
GPU 호환성 문제가 해결되면 GPU로 전환할 수 있습니다.

---

**테스트 날짜**: 2025-11-09

