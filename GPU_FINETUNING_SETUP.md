# CENTaUR Fine-tuning GPU 강제 사용 설정

## ✅ 구현 완료

### GPU 사용 강제 확인 로직 추가

1. **초기 GPU 확인** (학습 시작 전)
   - CUDA 사용 가능 여부 확인
   - GPU 개수 및 메모리 확인
   - CUDA_VISIBLE_DEVICES 설정 확인

2. **모델 로딩 후 GPU 확인**
   - 모델 파라미터가 GPU에 로드되었는지 확인
   - GPU 파라미터 vs CPU 파라미터 개수 확인
   - GPU 메모리 사용량 확인
   - GPU에 로드되지 않으면 RuntimeError 발생

3. **학습 시작 전 최종 확인**
   - CUDA 사용 가능 여부 재확인
   - 모델 디바이스 확인
   - GPU가 아닌 경우 RuntimeError 발생

4. **학습 중 GPU 모니터링**
   - 학습 전/후 GPU 사용률 확인
   - nvidia-smi를 통한 GPU 상태 모니터링

## 📝 주요 코드 변경

### `ko_centaur/training/train_qwen25_32b_qlora.py`

```python
# [0/6] GPU 사용 확인 및 강제 설정
if not torch.cuda.is_available():
    raise RuntimeError("❌ CUDA를 사용할 수 없습니다! GPU가 필요합니다.")

# 모델 로딩 후 GPU 확인
gpu_params = 0
cpu_params = 0
for name, param in model.named_parameters():
    if param.device.type == 'cuda':
        gpu_params += 1
    else:
        cpu_params += 1

if gpu_params == 0:
    raise RuntimeError("❌ 오류: 모델이 GPU에 로드되지 않았습니다!")

# 학습 시작 전 최종 확인
model_device = next(model.parameters()).device
if model_device.type != 'cuda':
    raise RuntimeError(f"❌ 오류: 모델이 GPU가 아닌 {model_device}에 있습니다!")
```

## ⚠️ 현재 문제

### 1. bitsandbytes 라이브러리 오류
```
OSError: libnvJitLink.so.12: cannot open shared object file
```
- **원인**: CUDA 라이브러리 경로 문제
- **해결**: CUDA 라이브러리 경로 설정 필요

### 2. GPU 메모리 부족
```
ValueError: Some modules are dispatched on the CPU or the disk
```
- **원인**: 32B 모델이 GPU 메모리에 다 들어가지 않음
- **해결**: CPU offloading 허용 또는 모델 크기 축소

## 🔧 해결 방안

### 옵션 1: CPU Offloading 허용 (일부 모듈만)
```python
device_map = "auto"  # GPU 우선, 부족하면 CPU
max_memory = {0: "100GB", "cpu": "200GB"}  # GPU 100GB, CPU 200GB
```

### 옵션 2: 더 작은 모델 사용
- Qwen2.5-7B 또는 14B 사용
- 또는 EXAONE-7.8B 사용

### 옵션 3: bitsandbytes 라이브러리 수정
- CUDA 라이브러리 경로 설정
- 또는 다른 quantization 방법 사용

## 🚀 실행 방법

```bash
ssh dgx-spark
cd ~/git/CENTaUR
bash scripts/start_centaur_finetuning.sh qwen25
```

## 📊 모니터링

- **로그**: `tail -f logs/qwen25-32b-qlora_finetuning.log`
- **GPU 사용률**: `watch -n 1 nvidia-smi`
- **프로파일**: `profiles/qwen25-32b-qlora_finetuning.nsys-rep`

## ✅ GPU 사용 확인 포인트

1. ✅ 초기 GPU 확인 통과
2. ✅ 모델 로딩 후 GPU 파라미터 확인
3. ✅ 학습 시작 전 모델 디바이스 확인
4. ✅ 학습 중 GPU 사용률 모니터링

**모든 단계에서 GPU 사용이 확인되지 않으면 RuntimeError 발생하여 학습이 중단됩니다.**

