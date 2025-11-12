# Ko-CENTaUR 마스터 실행 계획 (업데이트)
**작성일**: 2025-11-09 (최종 업데이트)
**목적**: 5개 Base 모델 + 2개 Fine-tuned 모델 전체 실험 자동화

---

## 🎯 프로젝트 목표

**한국어 가능 LLM으로 CENTaUR 방법론 재현 및 성능 비교**

- **5개 Base 모델** vs **Fine-tuned 모델** 비교
- 다양한 크기 (20B, 32B, 1T) 및 아키텍처 (Dense, MoE) 비교
- 한국어 특화 모델 vs 다국어 모델 성능 평가
- Generation bias 문제 해결
- **최종 목표**: NLL < Random Baseline (0.6931)

---

## 📊 현재 상태 (2025-11-09 Updated)

### ✅ 완료

1. **환경 설정**
   - ✅ NGC PyTorch 24.08 설치 및 설정
   - ✅ Shell 환경 자동화 (`ngc-python` wrapper)
   - ✅ GPU 사용 가능 확인 (NVIDIA GB10, 119GB)

2. **모델 다운로드 (5개)**
   - ✅ Qwen2.5-32B-Instruct (62GB)
   - ✅ DeepSeek-R1-Distill-Qwen-32B (62GB)
   - ✅ EXAONE-3.5-32B-Instruct (120GB)
   - ✅ **GPT-OSS-20B** (39GB) - **새로 추가**
   - ✅ **Kimi-K2-Instruct** (408GB) - **새로 추가**
   - **총 다운로드**: ~751GB

3. **Fine-tuning**
   - ✅ Qwen2.5-32B QLoRA (완료)
     - Loss: 1.0426 → 0.3413
     - 저장: `models/qwen25-32b-qlora/`

4. **Base 모델 Feature Extraction**
   - ✅ Qwen2.5-32B Base (완료)
   - ✅ DeepSeek-R1 Base (완료)
   - ⏳ EXAONE-3.5-32B Base (진행 중?)

### ❌ 미완료

- **Base Feature Extraction**
  - GPT-OSS-20B Base (새로 추가)
  - Kimi-K2-Instruct Base (새로 추가)

- **Fine-tuned Feature Extraction**
  - Qwen2.5 Fine-tuned
  - DeepSeek Fine-tuned (훈련 후)

- **Fine-tuning**
  - DeepSeek-R1-32B QLoRA

- **LOO CV** (7개 모델)
  - Qwen2.5 Base, Qwen2.5 Fine-tuned
  - DeepSeek Base, DeepSeek Fine-tuned
  - EXAONE-3.5 Base
  - GPT-OSS-20B Base
  - Kimi-K2-Instruct Base

- **결과 분석 및 비교**

---

## 🗺️ 전체 파이프라인 (업데이트)

```
┌─────────────────────────────────────────────────────────────┐
│              Ko-CENTaUR Pipeline (5 Base + 2 FT)             │
└─────────────────────────────────────────────────────────────┘

Phase 1: Fine-tuning (GPU 필요, 선택적)
├── Qwen2.5-32B QLoRA          ✅ 완료 (1h 27m)
├── DeepSeek-R1 QLoRA          ❌ 대기 (~1-2h 예상)
└── (선택) EXAONE-3.5 QLoRA    ❌ 선택사항

Phase 2: Feature Extraction (GPU 필요)
├── Base 모델 (5개)
│   ├── Qwen2.5 Base               ✅ 완료
│   ├── DeepSeek Base              ✅ 완료
│   ├── EXAONE-3.5 Base            ⏳ 진행 중
│   ├── GPT-OSS-20B Base           ❌ 대기 (새로 추가)
│   └── Kimi-K2-Instruct Base      ❌ 대기 (새로 추가)
│
└── Fine-tuned 모델 (2개)
    ├── Qwen2.5 Fine-tuned         ❌ 대기
    └── DeepSeek Fine-tuned        ❌ 대기 (훈련 후)

Phase 3: LOO Cross-Validation (CPU 가능)
├── Base 모델 (5개)
│   ├── Qwen2.5 Base               ❌ 대기
│   ├── DeepSeek Base              ❌ 대기
│   ├── EXAONE-3.5 Base            ❌ 대기
│   ├── GPT-OSS-20B Base           ❌ 대기
│   └── Kimi-K2-Instruct Base      ❌ 대기
│
└── Fine-tuned 모델 (2개)
    ├── Qwen2.5 Fine-tuned         ❌ 대기
    └── DeepSeek Fine-tuned        ❌ 대기

Phase 4: Analysis & Comparison
├── Base vs Fine-tuned 비교 (Qwen2.5, DeepSeek)
├── 모델 크기 비교 (20B vs 32B vs 1T)
├── 아키텍처 비교 (Dense vs MoE)
├── 한국어 특화 분석 (EXAONE vs 다국어)
└── Generation bias 분석 (7개 모델)
```

---

## 📋 모델 상세 정보

### Base 모델 (5개)

| 모델 | 크기 | 아키텍처 | 특징 | 다운로드 |
|------|------|----------|------|----------|
| **Qwen2.5-32B-Instruct** | 62GB | Dense, 32B | 다국어, SOTA | ✅ 완료 |
| **DeepSeek-R1-32B** | 62GB | Dense, 32B | Reasoning 특화 | ✅ 완료 |
| **EXAONE-3.5-32B** | 120GB | Dense, 32B | 한국어 특화 (LG AI) | ✅ 완료 |
| **GPT-OSS-20B** | 39GB | Dense, 20B | OpenAI 오픈소스 | ✅ 완료 |
| **Kimi-K2-Instruct** | 408GB | MoE, 1T | Multi-Agent, 128K context | ✅ 완료 |

**총 Base 모델 크기**: ~751GB

### Fine-tuned 모델 (2개, 확장 가능)

| 모델 | Status | Adapter 크기 | 저장 위치 |
|------|--------|--------------|-----------|
| **Qwen2.5-32B QLoRA** | ✅ 완료 | ~513MB | models/qwen25-32b-qlora/ |
| **DeepSeek-R1 QLoRA** | ❌ 대기 | ~500MB 예상 | models/deepseek-r1-qlora/ |
| (선택) EXAONE-3.5 QLoRA | 선택사항 | ~500MB 예상 | models/exaone35-qlora/ |

---

## 🤖 자동화 실행 스크립트

### 전체 파이프라인 자동 실행

```bash
# 전체 실행 (한 번에, 권장)
./scripts/run_full_pipeline.sh

# 또는 단계별 실행
./scripts/run_phase1_finetuning.sh     # Fine-tuning (GPU, ~1-2h each)
./scripts/run_phase2_extraction.sh     # Feature Extraction (GPU, ~30min each)
./scripts/run_phase3_loo_cv.sh         # LOO CV (CPU/GPU, ~2-4h each)
./scripts/run_phase4_analysis.sh       # Analysis (CPU, ~10min)
```

### Phase 1: Fine-tuning (GPU 필요)

```bash
#!/bin/bash
# scripts/run_phase1_finetuning.sh

# Qwen2.5-32B (✅ 완료)
if [ ! -d "models/qwen25-32b-qlora" ]; then
    ./scripts/train_qwen25_ngc.sh
fi

# DeepSeek-R1-32B
if [ ! -d "models/deepseek-r1-qlora" ]; then
    ./scripts/train_deepseek_r1_ngc.sh
fi

# (선택) EXAONE-3.5-32B
# if [ ! -d "models/exaone35-qlora" ]; then
#     ./scripts/train_exaone35_ngc.sh
# fi
```

**예상 시간**: 각 1-2시간

### Phase 2: Feature Extraction (GPU 필요)

```bash
#!/bin/bash
# scripts/run_phase2_extraction.sh

# Base 모델 (5개)
./ngc-python scripts/extract_centaur_features.py --model qwen25-base
./ngc-python scripts/extract_centaur_features.py --model deepseek-base
./ngc-python scripts/extract_centaur_features.py --model exaone35-base
./ngc-python scripts/extract_centaur_features.py --model gpt-oss-base
./ngc-python scripts/extract_centaur_features.py --model kimi-k2

# Fine-tuned 모델 (2개)
./ngc-python scripts/extract_centaur_features.py --model qwen25
./ngc-python scripts/extract_centaur_features.py --model deepseek
```

**예상 시간**: 각 30분-1시간

### Phase 3: LOO Cross-Validation (CPU 가능)

```bash
#!/bin/bash
# scripts/run_phase3_loo_cv.sh

models=(
    "qwen25_base"
    "qwen25_finetuned"
    "deepseek_base"
    "deepseek_finetuned"
    "exaone35_base"
    "gpt_oss_20b_base"
    "kimi_k2_base"
)

for model in "${models[@]}"; do
    python scripts/fit_centaur_loo_cv.py \
        --features data/features/${model}_features.pth \
        --output data/results/${model}_loo_results.json \
        --model_name "$model"
done
```

**예상 시간**: 각 2-4시간 (CPU), 30분-1시간 (GPU)

### Phase 4: Analysis (CPU)

```bash
#!/bin/bash
# scripts/run_phase4_analysis.sh

# 종합 분석
python scripts/analyze_all_results.py \
    --results_dir data/results/ \
    --output reports/final_analysis.md

# Generation bias 분석
python scripts/analyze_generation_bias.py \
    --results_dir data/results/ \
    --features_dir data/features/ \
    --output reports/generation_bias_analysis.md

# 시각화
python scripts/visualize_results.py \
    --results_dir data/results/ \
    --output_dir figures/
```

**예상 시간**: ~10분

---

## 📊 모니터링 설정 (중요!)

### GPU 프로파일링 & W&B 통합

**모든 Fine-tuning 작업에 자동 모니터링 권장**

#### 1. GPU 프로파일링 (nsys)

```bash
# 자동 프로파일링 활성화
bash scripts/start_centaur_finetuning_with_monitoring.sh qwen25
```

**특징:**
- ✅ 자동 프로파일링 (모든 학습)
- ✅ 프로파일 파일 자동 관리 (최신 10개만 유지)
- ✅ 타임스탬프 포함 파일명
- ✅ 낮은 오버헤드 (< 1% CPU)
- 📁 저장 위치: `profiles/{MODEL}_{TIMESTAMP}.nsys-rep`

#### 2. Weights & Biases (W&B)

```bash
# W&B 설정 (한 번만)
export WANDB_API_KEY='your_api_key'
export WANDB_PROJECT='ko-centaur-training'

# W&B 로그인 (또는)
wandb login
```

**장점:**
- ✅ 실시간 모니터링 (Loss, LR, GPU 사용률)
- ✅ 웹 대시보드 (https://wandb.ai)
- ✅ 실험 비교 (여러 모델 동시)
- ✅ 자동 로깅 (Transformers 통합)
- ✅ 협업 & 히스토리 관리

**설정 파일 업데이트:**

`ko_centaur/configs/training_qwen25_32b_qlora.yaml`:
```yaml
misc:
  use_wandb: true
  wandb_project: "ko-centaur-training"
  report_to: ["wandb"]
```

#### 3. 이메일 알림 (선택사항)

```bash
# 장기 실행 시 10분 간격 이메일 알림
# ~/git/Emailer 이용
```

### 모니터링 비교

| 기능 | GPU 프로파일링 (nsys) | W&B | 이메일 알림 |
|------|---------------------|-----|------------|
| 실시간 메트릭 | ❌ | ✅ | ⚠️ (10분 간격) |
| GPU 성능 분석 | ✅ | ⚠️ (기본) | ❌ |
| 실험 비교 | ❌ | ✅ | ❌ |
| 웹 대시보드 | ❌ | ✅ | ❌ |
| 오버헤드 | 낮음 | 매우 낮음 | 낮음 |
| 디스크 사용 | 중간 | 낮음 | 낮음 |

### 권장 구성

**최적 모니터링:**
1. ✅ GPU 프로파일링: 모든 학습에 활성화
2. ✅ W&B: 모든 학습에 활성화 (필수)
3. ⚠️ 이메일: 장기 실행 시에만

**사용 예시:**

```bash
# W&B API 키 설정
export WANDB_API_KEY='your_api_key'

# 모니터링 포함 학습 시작
bash scripts/start_centaur_finetuning_with_monitoring.sh qwen25

# 모니터링 확인
# - 로그: tail -f logs/qwen25-32b-qlora_*.log
# - W&B: https://wandb.ai
# - GPU: watch -n 1 nvidia-smi
```

---

## 📅 실행 스케줄 (업데이트)

### Week 1: Fine-tuning & Feature Extraction

#### Day 1 (Today - Completed)
- [x] Qwen2.5-32B Fine-tuning 완료 (1h 27m)
- [x] 모든 모델 다운로드 확인 (5개)

#### Day 2-3
- [ ] DeepSeek-R1 Fine-tuning (~1-2h)
- [ ] Base 모델 Feature Extraction 완료 (5개, ~2-5h)
- [ ] Fine-tuned 모델 Feature Extraction (2개, ~1-2h)

#### Day 4-5
- [ ] 모든 Feature Extraction 검증
- [ ] Generation bias 초기 분석

### Week 2: LOO Cross-Validation

#### Day 6-8
- [ ] Base 모델 LOO CV (5개)
  - Qwen2.5 Base
  - DeepSeek Base
  - EXAONE-3.5 Base
  - GPT-OSS-20B Base
  - Kimi-K2 Base

#### Day 9-11
- [ ] Fine-tuned 모델 LOO CV (2개)
  - Qwen2.5 Fine-tuned
  - DeepSeek Fine-tuned

### Week 3: Analysis

#### Day 12-14
- [ ] Base vs Fine-tuned 비교 (Qwen2.5, DeepSeek)
- [ ] 모델 크기 비교 (20B vs 32B vs 1T)
- [ ] 아키텍처 비교 (Dense vs MoE)
- [ ] 한국어 특화 분석 (EXAONE)
- [ ] Generation bias 종합 분석

#### Day 15
- [ ] 최종 리포트 생성
- [ ] 결과 시각화

---

## 📊 예상 성능 목표

### Baseline

```
Random: NLL = 0.6931 (ln 2)
```

### 가설

**모델 크기 효과:**
```
GPT-OSS-20B < Qwen2.5-32B ≈ DeepSeek-32B < Kimi-K2-1T
```

**아키텍처 효과:**
```
Dense models vs MoE (Kimi-K2)
→ MoE는 더 큰 capacity, 하지만 실제 성능은?
```

**한국어 특화 효과:**
```
EXAONE-3.5 (한국어 특화) vs Qwen2.5 (다국어)
→ 한국어 데이터셋에서 EXAONE이 우수할 것으로 예상
```

**Fine-tuning 효과:**
```
Base 모델 < Fine-tuned 모델
→ Qwen2.5: Base vs Fine-tuned
→ DeepSeek: Base vs Fine-tuned
```

### 비교 분석 매트릭스

| 모델 | Type | 예상 NLL | Generation Bias | 비고 |
|------|------|----------|-----------------|------|
| GPT-OSS-20B | Base | ? | Low | 가장 작은 모델 |
| Qwen2.5-32B | Base | ? | Low | 기준선 (32B) |
| Qwen2.5-32B | Fine-tuned | ? | Medium? | Fine-tuning 효과 |
| DeepSeek-R1-32B | Base | ? | Low | Reasoning 특화 |
| DeepSeek-R1-32B | Fine-tuned | ? | Medium? | Fine-tuning 효과 |
| EXAONE-3.5-32B | Base | ? | Low | 한국어 특화 |
| Kimi-K2-1T | Base | ? | Low | 가장 큰 모델 (MoE) |

---

## ⚠️ 주의사항

### 리소스 관리

```
GPU: NVIDIA GB10 (119GB VRAM)
- Fine-tuning: ~26GB per model
- Feature Extraction: ~18-20GB per model
- Kimi-K2 (1T MoE): 4-bit quantization 필수
- LOO CV: CPU로 실행 가능 (병렬화 가능)

디스크 공간:
- 모델 다운로드: ~751GB (완료)
- LoRA adapters: ~500MB each
- Features: ~1-2MB each
- 결과: ~1MB each
- 총 여유 공간 필요: ~50GB (adapters + features + results)
```

### 실행 순서

```
1. GPU 작업 우선
   - Fine-tuning (순차)
   - Feature Extraction (순차 또는 병렬)

2. CPU 작업 병렬
   - LOO CV (7개 모델 동시 가능, 메모리 허용 시)

3. 분석
   - 모든 작업 완료 후
```

### 모델별 특수 요구사항

```
EXAONE-3.5-32B:
- trust_remote_code=True 필요
- is_local=True 설정

Kimi-K2-Instruct:
- trust_remote_code=True 필요
- is_local=True 설정
- 4-bit quantization 권장 (408GB → ~25GB)

GPT-OSS-20B:
- 모델 ID: "openai/gpt-oss-20b"
- HuggingFace cache 사용
```

---

## 🚀 다음 단계 (우선순위)

### Priority 1: 즉시 실행 가능

```bash
# 전체 파이프라인 자동 실행
./scripts/run_full_pipeline.sh
```

**이 명령어 하나로:**
- DeepSeek-R1 Fine-tuning 자동 시작
- 모든 Base 모델 Feature Extraction (5개)
- Fine-tuned 모델 Feature Extraction (2개)
- 모든 LOO CV (7개)
- 결과 분석 및 시각화

### Priority 2: 단계별 실행

```bash
# Step 1: DeepSeek Fine-tuning만 먼저
./scripts/train_deepseek_r1_ngc.sh

# Step 2: Feature Extraction만
./scripts/run_phase2_extraction.sh

# Step 3: LOO CV만
./scripts/run_phase3_loo_cv.sh

# Step 4: Analysis만
./scripts/run_phase4_analysis.sh
```

---

## 📝 실행 로그

### 2025-11-09 19:30 (Initial)

```
✅ Qwen2.5-32B Fine-tuning 완료
   - Duration: 1h 27m
   - Final Loss: 0.3413
   - Output: models/qwen25-32b-qlora/

⏳ EXAONE-3.5 Feature Extraction 진행 중?
   - 상태 확인 필요
```

### 2025-11-09 20:00 (Updated)

```
✅ 모델 다운로드 확인 (5개)
   - Qwen2.5-32B (62GB)
   - DeepSeek-R1-32B (62GB)
   - EXAONE-3.5-32B (120GB)
   - GPT-OSS-20B (39GB)
   - Kimi-K2-Instruct (408GB)

✅ 전체 파이프라인 스크립트 생성 완료
   - run_full_pipeline.sh
   - run_phase1_finetuning.sh
   - run_phase2_extraction.sh
   - run_phase3_loo_cv.sh
   - run_phase4_analysis.sh

✅ 분석 스크립트 생성 완료
   - analyze_all_results.py
   - analyze_generation_bias.py
   - visualize_results.py

❌ 대기 중:
   - DeepSeek-R1 Fine-tuning
   - GPT-OSS-20B Feature Extraction
   - Kimi-K2 Feature Extraction
   - 모든 LOO CV 작업
```

---

## 🎯 최종 목표

### Week 1 (현재)
- [x] Qwen2.5 Fine-tuning 완료
- [x] 모든 모델 다운로드 확인 (5개)
- [x] 전체 자동화 스크립트 완료
- [ ] DeepSeek-R1 Fine-tuning 완료
- [ ] 모든 Feature Extraction 완료 (7개)

### Week 2
- [ ] 모든 LOO CV 완료 (7개)
- [ ] 초기 결과 분석

### Week 3
- [ ] Base vs Fine-tuned 상세 비교
- [ ] 모델 크기/아키텍처 비교
- [ ] 한국어 특화 분석
- [ ] Generation bias 분석
- [ ] 최종 리포트 작성

---

## 🔬 예상 연구 질문

1. **Fine-tuning 효과**: Choices13k 데이터로 fine-tuning한 모델이 인지 모델링에서 더 나은 성능을 보이는가?

2. **모델 크기 효과**: 20B < 32B < 1T 순서대로 성능이 향상되는가?

3. **아키텍처 효과**: Dense models vs MoE (Kimi-K2) - MoE가 더 나은가?

4. **한국어 특화 효과**: EXAONE-3.5가 한국어 데이터셋에서 더 나은 성능을 보이는가?

5. **Generation bias**: Fine-tuning이 generation bias를 증가시키는가?

---

**다음 실행**:
```bash
./scripts/run_full_pipeline.sh
```

**업데이트**: 매일 진행 상황 기록

**문의**: 문제 발생 시 각 Phase 스크립트 개별 실행 가능
