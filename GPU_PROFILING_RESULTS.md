# GPU Profiling 결과 분석 리포트

**생성일**: 2025-11-09  
**분석 도구**: NVIDIA Nsight Systems (nsys)  
**대상 서버**: dgx-spark (NVIDIA GB10 GPU)

---

## 📊 프로파일 파일 목록

| 파일명 | 크기 | 생성 시간 | 작업 유형 |
|--------|------|----------|----------|
| `qwen25_base_loo_cv.nsys-rep` | 157KB | 2025-11-09 14:24 | Qwen2.5 Base LOO CV |
| `qwen25-32b-qlora_finetuning.nsys-rep` | 188KB | 2025-11-09 16:00 | Qwen2.5-32B QLoRA Fine-tuning |
| `exaone35_feature_extraction.nsys-rep` | 8.4MB | 2025-11-09 17:58 | EXAONE-3.5 Feature Extraction |

---

## 1. Qwen2.5 Base LOO CV 프로파일

### 📈 CUDA API 요약

| 시간 비율 | 총 시간 (ns) | 호출 횟수 | 평균 (ns) | API 이름 |
|----------|------------|----------|----------|----------|
| 49.8% | 4,661,024 | 1 | 4,661,024 | `cudaGetDeviceProperties_v12000` |
| 31.9% | 2,982,800 | 1 | 2,982,800 | `cudaLaunchKernel` |
| 11.5% | 1,080,320 | 2 | 540,160 | `cudaMalloc` |
| 3.4% | 314,800 | 3 | 104,933 | `cudaStreamSynchronize` |
| 3.3% | 310,128 | 3 | 103,376 | `cudaMemcpyAsync` |

### 💾 GPU 메모리 작업 (시간 기준)

| 시간 비율 | 총 시간 (ns) | 횟수 | 평균 (ns) | 작업 |
|----------|------------|------|----------|------|
| 100.0% | 43,232 | 3 | 14,411 | Host-to-Device 전송 |

### 💾 GPU 메모리 작업 (크기 기준)

| 총 크기 (MB) | 횟수 | 평균 (MB) | 최대 (MB) | 작업 |
|-------------|------|----------|----------|------|
| 2.049 | 3 | 0.683 | 2.048 | Host-to-Device 전송 |

### 🔍 주요 발견 사항

1. **짧은 실행 시간**: 프로파일 파일 크기가 작음 (157KB) → 매우 짧은 실행 시간
2. **GPU 커널 실행**: `cudaLaunchKernel` 호출 확인 → GPU 사용 확인
3. **적은 메모리 전송**: 총 2MB 미만의 데이터 전송
4. **CPU 실행 가능성**: 대부분의 작업이 CPU에서 실행되었을 가능성

---

## 2. Qwen2.5-32B QLoRA Fine-tuning 프로파일

### 📈 CUDA API 요약

| 시간 비율 | 총 시간 (ns) | 호출 횟수 | 평균 (ns) | 최대 (ns) | API 이름 |
|----------|------------|----------|----------|----------|----------|
| 95.3% | 20,968,131,552 | 10 | 2,096,831,155 | 9,125,352,848 | `cudaMemcpyAsync` |
| 3.1% | 691,580,896 | 5 | 138,316,179 | 300,923,872 | `cudaMalloc` |
| 1.5% | 330,137,664 | 2 | 165,068,832 | 330,068,960 | `cudaMemGetInfo` |
| 0.0% | 4,323,088 | 1 | 4,323,088 | 4,323,088 | `cudaGetDeviceProperties_v12000` |

### 💾 GPU 메모리 작업 (시간 기준)

| 시간 비율 | 총 시간 (ns) | 횟수 | 평균 (ns) | 최대 (ns) | 작업 |
|----------|------------|------|----------|----------|------|
| 99.9% | 20,800,558,016 | 8 | 2,600,069,752 | 9,124,728,672 | Host-to-Device 전송 |
| 0.1% | 15,942,880 | 2 | 7,971,440 | 8,114,656 | Device-to-Host 전송 |

### 💾 GPU 메모리 작업 (크기 기준)

| 총 크기 (MB) | 횟수 | 평균 (MB) | 중간값 (MB) | 최대 (MB) | 작업 |
|-------------|------|----------|------------|----------|------|
| **4,246.753** | 8 | 530.844 | 283.116 | 1,557.135 | Host-to-Device 전송 |
| **566.231** | 2 | 283.116 | 283.116 | 283.116 | Device-to-Host 전송 |

### 🔍 주요 발견 사항

1. **대량 메모리 전송**: 
   - Host-to-Device: **4.25GB** (모델 로딩, 데이터 전송)
   - Device-to-Host: **566MB** (결과 반환)
   - 총 **4.8GB** 메모리 전송

2. **메모리 전송이 주요 병목**:
   - 전체 시간의 95.3%가 `cudaMemcpyAsync`에 소요
   - 평균 전송 시간: 2.6초 (최대 9.1초)

3. **큰 메모리 할당**:
   - `cudaMalloc` 호출 5회, 평균 138MB
   - 최대 할당: 300MB

4. **프로파일 시간**:
   - 파일 크기 188KB → 모델 로딩 단계만 프로파일링됨
   - 실제 학습은 프로파일링 범위 밖에서 실행 중

---

## 3. EXAONE-3.5 Feature Extraction 프로파일

### 📈 CUDA API 요약

| 시간 비율 | 총 시간 (ns) | 호출 횟수 | 평균 (ns) | API 이름 |
|----------|------------|----------|----------|----------|
| 100.0% | 4,580,592 | 1 | 4,580,592 | `cudaGetDeviceProperties_v12000` |
| 0.0% | 624 | 1 | 624 | `cuModuleGetLoadingMode` |

### 💾 GPU 메모리 작업

**GPU 메모리 데이터 없음** → CPU에서 실행됨

### 🔍 주요 발견 사항

1. **CPU 실행 확인**:
   - GPU 메모리 작업이 전혀 없음
   - `cudaGetDeviceProperties`만 호출 (GPU 정보 확인용)
   - 실제 GPU 연산 없음

2. **큰 프로파일 파일**:
   - 8.4MB → 긴 실행 시간
   - CPU에서 실행되어 많은 이벤트 기록

3. **GPU 사용 불가 원인**:
   - PyTorch/GPU 호환성 문제 (sm_121 미지원)
   - `bitsandbytes` 라이브러리 오류
   - CPU fallback 발생

---

## 📊 종합 분석

### 메모리 전송 패턴 비교

| 작업 | Host-to-Device (MB) | Device-to-Host (MB) | 총 전송 (MB) |
|------|-------------------|-------------------|-------------|
| Qwen2.5 Base LOO CV | 2.049 | 0 | 2.049 |
| Qwen2.5-32B Fine-tuning | **4,246.753** | **566.231** | **4,812.984** |
| EXAONE Feature Extraction | 0 (CPU 실행) | 0 (CPU 실행) | 0 |

### GPU 사용 현황

| 작업 | GPU 사용 | 메모리 전송 | 커널 실행 |
|------|---------|-----------|----------|
| Qwen2.5 Base LOO CV | ✅ 부분적 | ✅ 소량 (2MB) | ✅ 확인됨 |
| Qwen2.5-32B Fine-tuning | ✅ 활발 | ✅ 대량 (4.8GB) | ✅ 확인됨 |
| EXAONE Feature Extraction | ❌ CPU 실행 | ❌ 없음 | ❌ 없음 |

### 성능 병목 지점

1. **Qwen2.5-32B Fine-tuning**:
   - **주요 병목**: 메모리 전송 (95.3% 시간 소요)
   - **평균 전송 시간**: 2.6초 (최대 9.1초)
   - **최적화 방안**: 
     - 데이터 로더 최적화 (배치 크기 조정)
     - 비동기 전송 최적화
     - 메모리 프리페칭

2. **EXAONE Feature Extraction**:
   - **주요 문제**: GPU 사용 불가 (CPU fallback)
   - **해결 방안**:
     - NGC PyTorch 컨테이너 사용
     - CUDA 호환성 확인
     - `bitsandbytes` 라이브러리 업데이트

---

## ⚠️ 주의사항 및 권장사항

### 1. Fine-tuning 프로파일링 제한
- 현재 프로파일은 **모델 로딩 단계만** 포함
- 실제 학습 루프는 프로파일링 범위 밖
- **권장**: 학습 루프 전체를 포함하도록 프로파일링 재실행

### 2. EXAONE GPU 사용 불가
- 현재 CPU에서 실행 중
- GPU 사용 시 성능 대폭 향상 가능
- **권장**: GPU 호환성 문제 해결 후 재실행

### 3. 메모리 전송 최적화
- Fine-tuning에서 메모리 전송이 주요 병목
- **권장**: 
  - 데이터 로더 최적화
  - 배치 크기 조정
  - 메모리 프리페칭 활성화

---

## 🔧 상세 분석 방법

### 방법 1: nsys-ui 사용 (권장)
```bash
# 로컬에서 프로파일 파일 다운로드
scp dgx-spark:~/git/CENTaUR/profiles/*.nsys-rep ./

# nsys-ui로 열기
nsys-ui qwen25-32b-qlora_finetuning.nsys-rep
```

### 방법 2: SQLite 직접 쿼리
```bash
ssh dgx-spark
cd ~/git/CENTaUR
sqlite3 profiles/qwen25-32b-qlora_finetuning.sqlite

# 테이블 목록 확인
.tables

# CUDA 커널 정보 조회
SELECT * FROM CUPTI_ACTIVITY_KIND_KERNEL LIMIT 10;

# 메모리 전송 정보 조회
SELECT * FROM CUPTI_ACTIVITY_KIND_MEMCPY LIMIT 10;
```

### 방법 3: 추가 리포트 생성
```bash
# 사용 가능한 리포트 목록
nsys stats --help

# 특정 리포트 생성
nsys stats --report cuda_api_sum profiles/qwen25-32b-qlora_finetuning.nsys-rep
nsys stats --report cuda_gpu_mem_time_sum profiles/qwen25-32b-qlora_finetuning.nsys-rep
nsys stats --report cuda_gpu_mem_size_sum profiles/qwen25-32b-qlora_finetuning.nsys-rep
```

---

## 📈 다음 단계

1. **Fine-tuning 전체 프로파일링**:
   - 학습 루프 전체를 포함하도록 프로파일링 재실행
   - GPU 커널 실행 시간 분석

2. **EXAONE GPU 호환성 해결**:
   - NGC PyTorch 컨테이너 사용
   - GPU 사용 확인 후 재프로파일링

3. **메모리 전송 최적화**:
   - 데이터 로더 최적화
   - 배치 크기 및 프리페칭 설정 조정

4. **지속적 모니터링**:
   - Weights & Biases (W&B) 통합
   - 실시간 GPU 사용률 모니터링

---

## 🔗 참고 자료

- **nsys 문서**: https://docs.nvidia.com/nsight-systems/
- **프로파일 파일 위치**: `~/git/CENTaUR/profiles/`
- **SQLite 파일 위치**: `~/git/CENTaUR/profiles/*.sqlite`

---

**최종 업데이트**: 2025-11-09  
**분석자**: GPU Profiling System
