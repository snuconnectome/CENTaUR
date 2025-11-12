# GPU 성능 분석 리포트

**생성일**: 2025-11-09  
**서버**: dgx-spark  
**GPU**: NVIDIA GB10 (128.5 GB)

---

## 📊 실행 중인 작업 현황

### 현재 실행 중
1. **EXAONE-3.5-32B Feature Extraction**
   - PID: 941241
   - CPU 사용률: 731%
   - 메모리 사용: 6.7% (약 21GB)
   - 상태: 실행 중 (프로파일링 포함)

2. **Qwen2.5-32B Base LOO CV**
   - PID: 941522
   - CPU 사용률: 772%
   - 메모리 사용: 0.4%
   - 상태: 실행 중 (CPU 모드)

---

## 🔍 GPU 프로파일링 결과

### Qwen2.5-32B Base LOO CV 프로파일

**프로파일 파일**: `profiles/qwen25_base_loo_cv.nsys-rep` (157KB)

#### CUDA API 호출 분석
| API | 시간 비율 | 총 시간 (ns) | 호출 횟수 | 평균 시간 (ns) |
|-----|----------|-------------|----------|---------------|
| `cudaGetDeviceProperties_v12000` | 49.8% | 4,661,024 | 1 | 4,661,024 |
| `cudaLaunchKernel` | 31.9% | 2,982,800 | 1 | 2,982,800 |
| `cudaMalloc` | 11.5% | 1,080,320 | 2 | 540,160 |
| `cudaStreamSynchronize` | 3.4% | 314,800 | 3 | 104,933 |
| `cudaMemcpyAsync` | 3.3% | 310,128 | 3 | 103,376 |

**총 CUDA API 호출**: 13회

#### GPU 메모리 사용
- **Host-to-Device 메모리 복사**: 2.049 MB (3회)
- 평균 복사 크기: 0.683 MB
- 최대 복사 크기: 2.048 MB

#### 주요 발견사항
1. **GPU 커널 실행 불가**: CUDA 호환성 문제로 실제 GPU 커널이 실행되지 않음
   - GPU: NVIDIA GB10 (CUDA capability sm_121)
   - PyTorch 지원: sm_50, sm_80, sm_86, sm_89, sm_90, sm_90a
   - 결과: CPU fallback으로 실행됨

2. **초기화 오버헤드**: `cudaGetDeviceProperties`가 전체 시간의 49.8% 차지
   - GPU 초기화 과정에서 발생
   - 실제 연산보다 초기화에 더 많은 시간 소요

3. **메모리 할당**: 작은 메모리 할당 (약 2MB)만 발생
   - 대부분의 연산이 CPU에서 수행됨

---

## ⚠️ GPU 호환성 문제

### 문제
- **GPU 모델**: NVIDIA GB10
- **CUDA Capability**: sm_121
- **PyTorch 지원**: sm_50 ~ sm_90a
- **결과**: GPU 커널 실행 불가, CPU fallback

### 영향
1. **Feature Extraction**: CPU로 실행 (느림)
2. **LOO CV**: CPU로 실행 (느림)
3. **GPU 프로파일링**: CUDA API 호출만 추적 가능, 실제 커널 실행 없음

### 해결 방안
1. **PyTorch 재컴파일**: sm_121 지원하는 PyTorch 빌드 필요
2. **NGC 컨테이너 사용**: NVIDIA에서 제공하는 호환 컨테이너 사용
3. **CPU 실행**: 현재 상태로 계속 실행 (느리지만 동작함)

---

## 📈 성능 지표

### 현재 GPU 상태
- **GPU 사용률**: 0%
- **메모리 사용률**: 0%
- **전력 소비**: 12.85 W (idle)
- **온도**: 50°C

### 작업별 리소스 사용
| 작업 | CPU 사용률 | 메모리 사용 | GPU 사용 | 상태 |
|------|----------|-----------|---------|------|
| EXAONE Feature Extraction | 731% | 21GB | 0% | 실행 중 |
| Qwen2.5 LOO CV | 772% | ~1GB | 0% | 실행 중 |

---

## 🎯 다음 단계

### 즉시 실행 가능
1. ✅ **Qwen2.5-32B Base LOO CV** - 실행 중 (CPU)
2. ✅ **EXAONE-3.5-32B Feature Extraction** - 실행 중 (CPU)
3. ⏳ **DeepSeek-R1 Base LOO CV** - 대기 중

### 프로파일링 계획
1. EXAONE Feature Extraction 완료 후 프로파일 분석
2. 모든 작업 완료 후 종합 성능 리포트 생성
3. GPU 호환성 문제 해결 방안 검토

---

## 📝 프로파일 파일 위치

- `profiles/qwen25_base_loo_cv.nsys-rep` - Qwen2.5 LOO CV 프로파일
- `profiles/exaone35_feature_extraction.nsys-rep` - EXAONE Feature Extraction 프로파일 (생성 중)
- `gpu_performance_report.json` - 종합 분석 리포트

---

## 🔧 사용된 도구

- **Nsight Systems**: NVIDIA Nsight Systems 2025.3.2.474
- **프로파일링 스크립트**: `scripts/run_with_profiling.sh`
- **분석 스크립트**: `scripts/analyze_gpu_performance.py`
- **요약 스크립트**: `scripts/generate_performance_summary.sh`

---

**업데이트**: 작업 진행 상황에 따라 실시간 업데이트 예정

