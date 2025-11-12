# CENTaUR 실험 계획 (2025년 최신 버전)
**작성일**: 2025-11-12  
**목적**: 현재까지 완성된 작업과 남은 작업을 모두 고려한 체계적인 실험 계획

---

## 📊 현재 완료 상태

### ✅ 완료된 작업

#### 1. 모델 설정 및 추가
- ✅ GPT-OSS-20B 모델 추가
- ✅ EXAONE-3.5-32B 모델 추가 (코드 수정 완료)
- ✅ Kimi K2 모델 추가
- ✅ 모든 모델 BaselineModelManager 통합 완료

#### 2. Feature Extraction 완료
- ✅ **Qwen2.5-32B Base**: `outputs/qwen25_base_features.npz` (완료)
- ✅ **DeepSeek-R1 Base**: `outputs/deepseek_base_features.npz` (완료)

#### 3. 실험 및 분석 완료
- ✅ **Qwen2.5-32B Fine-tuned LOO CV**: 50/100 folds 완료 (중단됨)
  - NLL = 0.7623 (랜덤 baseline 0.6931보다 나쁨)
  - Generation bias 발견: 72% → B 편향
  - Feature similarity: 95.25% (너무 높음)
- ✅ **Generation Bias 분석 완료**: Fine-tuned 모델의 편향 문제 확인
- ✅ **Feature Similarity 분석 완료**: 편향이 원인임 확인
- ✅ **원인 분석 완료**: Fine-tuning이 편향을 학습함

#### 4. 인프라 및 도구
- ✅ 모니터링 시스템 설정 (10분마다 체크, 이메일 알림)
- ✅ NGC PyTorch 컨테이너 설치 및 GPU 성능 측정
- ✅ 실험 스크립트 생성 (`run_*.sh`)
- ✅ LOO CV 스크립트 (`fit_centaur_loo_cv_flexible.py`)

---

## 🎯 실험 목표

1. **Base 모델 성능 확인**: Fine-tuned 모델의 편향 문제를 피하기 위해 Base 모델로 실험
2. **다양한 모델 비교**: 5개 모델의 CENTaUR 성능 비교
3. **Generation Bias 분석**: 각 모델의 편향 패턴 분석
4. **최적 모델 선정**: 한국어 선택 과제 예측에 가장 적합한 모델 찾기

---

## 📋 실험 대상 모델

| 모델 | 파라미터 | 상태 | Feature | LOO CV | 우선순위 |
|------|----------|------|---------|--------|---------|
| **Qwen2.5-32B Base** | 32B | ✅ | ✅ 완료 | ❌ 미완료 | 🔥 **최우선** |
| **DeepSeek-R1 Base** | 32B | ✅ | ✅ 완료 | ❌ 미완료 | 🔥 **최우선** |
| **EXAONE-3.5-32B** | 32B | ✅ | ⚠️ 재실행 필요 | ❌ 미완료 | ⭐ 높음 |
| **GPT-OSS-20B** | 20B | ✅ | ❌ 미시작 | ❌ 미완료 | ⭐ 중간 |
| **Kimi K2** | 1T (32B 활성) | ✅ | ❌ 미시작 | ❌ 미완료 | ⚠️ 조건부 |

---

## 🚀 실험 계획 (우선순위별)

### Phase 1: 즉시 실행 가능 (이번 주)

#### 1-1. Qwen2.5-32B Base LOO CV ⭐⭐⭐
**상태**: Feature 있음, 바로 실행 가능  
**예상 시간**: 3-4시간  
**목적**: Base 모델 성능 확인 (Fine-tuned 편향 문제 회피)

```bash
# 실행 명령어
python scripts/fit_centaur_loo_cv_flexible.py \
    outputs/qwen25_base_features.npz \
    outputs/qwen25_base_nll_results.json \
    --model_name "Qwen2.5-32B-Base"

# 모니터링
tail -f logs/qwen25_base_loo_cv.log
```

**예상 결과**:
- NLL < 0.7623 (Fine-tuned보다 좋을 것으로 예상)
- Generation bias < 72% (더 균형잡힌 분포 기대)
- Feature similarity < 95.25% (더 다양한 feature 기대)

#### 1-2. DeepSeek-R1 Base LOO CV ⭐⭐⭐
**상태**: Feature 있음, 바로 실행 가능  
**예상 시간**: 3-4시간  
**목적**: 다른 Base 모델과 성능 비교

```bash
# 실행 명령어
python scripts/fit_centaur_loo_cv_flexible.py \
    outputs/deepseek_base_features.npz \
    outputs/deepseek_base_nll_results.json \
    --model_name "DeepSeek-R1-32B-Base"

# 모니터링
tail -f logs/deepseek_base_loo_cv.log
```

**예상 결과**:
- Qwen2.5 Base와 성능 비교
- 모델 아키텍처별 차이 확인

#### 1-3. EXAONE-3.5-32B Feature Extraction 재실행 ⭐⭐
**상태**: 코드 수정 완료, 재실행 필요  
**예상 시간**: 2-4시간  
**목적**: 한국어 최고 성능 모델 실험

```bash
# 실행 명령어
./run_exaone35_full.sh

# 또는 직접 실행
python scripts/extract_centaur_features.py \
    --model exaone35 \
    --output outputs/exaone35_features.npz

# 모니터링
tail -f logs/exaone35_extraction.log
```

**주의사항**:
- `trust_remote_code` 문제 해결됨 (`is_local = True` 설정)
- NGC PyTorch 컨테이너 사용 권장

---

### Phase 2: 다음 주 (Base 모델 결과 확인 후)

#### 2-1. EXAONE-3.5-32B LOO CV ⭐⭐
**조건**: Feature Extraction 완료 후  
**예상 시간**: 3-4시간

```bash
# 실행 명령어
./run_exaone35_loo_cv.sh

# 또는 직접 실행
python scripts/fit_centaur_loo_cv_flexible.py \
    outputs/exaone35_features.npz \
    outputs/exaone35_nll_results.json \
    --model_name "EXAONE-3.5-32B"
```

**예상 결과**:
- 한국어 최고 성능 모델의 CENTaUR 성능 확인
- Base 모델들과 비교

#### 2-2. GPT-OSS-20B Feature Extraction ⭐
**예상 시간**: 2-4시간  
**목적**: OpenAI 오픈소스 모델 성능 확인

```bash
# 실행 명령어
./run_gpt_oss_full.sh

# 모니터링
tail -f logs/gpt_oss_extraction.log
```

#### 2-3. GPT-OSS-20B LOO CV ⭐
**조건**: Feature Extraction 완료 후  
**예상 시간**: 3-4시간

```bash
# 실행 명령어
./run_gpt_oss_loo_cv.sh
```

---

### Phase 3: 조건부 (저장공간 확인 후)

#### 3-1. Kimi K2 Feature Extraction ⚠️
**조건**: 저장공간 확인 필요 (~500GB INT4 기준)  
**예상 시간**: 4-6시간 (다운로드 포함)

```bash
# 저장공간 확인
df -h

# 실행 명령어 (조건부)
./run_kimi_k2_full.sh
```

#### 3-2. Kimi K2 LOO CV ⚠️
**조건**: Feature Extraction 완료 후  
**예상 시간**: 3-4시간

```bash
# 실행 명령어
./run_kimi_k2_loo_cv.sh
```

---

## 📅 일정 계획

### Week 1 (현재 주)
**목표**: Base 모델 실험 완료

- [ ] **Day 1-2**: Qwen2.5-32B Base LOO CV 실행
  - Feature 확인: `ls -lh outputs/qwen25_base_features.npz`
  - LOO CV 실행 및 모니터링
  - 결과 확인 및 분석

- [ ] **Day 3-4**: DeepSeek-R1 Base LOO CV 실행
  - Feature 확인: `ls -lh outputs/deepseek_base_features.npz`
  - LOO CV 실행 및 모니터링
  - 결과 확인 및 분석

- [ ] **Day 5**: EXAONE-3.5-32B Feature Extraction 재실행
  - 코드 확인 및 실행
  - 모니터링

**Week 1 성공 기준**:
- ✅ Qwen2.5-32B Base LOO CV 완료
- ✅ DeepSeek-R1 Base LOO CV 완료
- ✅ EXAONE-3.5-32B Feature Extraction 완료

---

### Week 2
**목표**: EXAONE-3.5-32B 완료, GPT-OSS-20B 시작

- [ ] **Day 8-9**: EXAONE-3.5-32B LOO CV 실행
  - Feature 확인
  - LOO CV 실행
  - 결과 분석

- [ ] **Day 10-11**: Base 모델 결과 비교 분석
  - Qwen2.5 Base vs DeepSeek Base 비교
  - Generation bias 분석
  - Feature similarity 분석

- [ ] **Day 12-14**: GPT-OSS-20B 실험
  - Feature Extraction
  - LOO CV 실행

**Week 2 성공 기준**:
- ✅ EXAONE-3.5-32B LOO CV 완료
- ✅ Base 모델 비교 분석 완료
- ✅ GPT-OSS-20B 실험 완료

---

### Week 3
**목표**: 모든 실험 완료, 종합 분석

- [ ] **Day 15-17**: Kimi K2 실험 (조건부)
  - 저장공간 확인
  - Feature Extraction
  - LOO CV 실행

- [ ] **Day 18-21**: 종합 분석 및 비교
  - 모든 모델 성능 비교표 작성
  - Generation bias 패턴 분석
  - Feature similarity 분석
  - 모델별 특징 분석

**Week 3 성공 기준**:
- ✅ 모든 실험 완료
- ✅ 종합 성능 비교표 작성
- ✅ 결과 분석 리포트 작성

---

### Week 4
**목표**: 결과 정리 및 논문 준비

- [ ] **Day 22-24**: 결과 리포트 작성
  - 실험 결과 통합
  - 성능 비교 차트 생성
  - 통계 분석

- [ ] **Day 25-28**: 논문 초안 작성
  - 실험 방법론 정리
  - 결과 섹션 작성
  - 토론 및 결론 작성

**Week 4 성공 기준**:
- ✅ 결과 리포트 완료
- ✅ 논문 초안 작성 완료

---

## 📊 예상 결과 및 성능 목표

### 성능 기준
- **Random Baseline**: NLL ≈ 0.6931
- **LLaMA-65B (원본 CENTaUR)**: NLL ≈ 30K (로그 스케일)
- **목표**: NLL < 0.6931 (랜덤보다 좋은 성능)

### 예상 결과

| 모델 | 예상 NLL | 예상 Generation Bias | 예상 Feature Similarity | 비고 |
|------|----------|----------------------|------------------------|------|
| Qwen2.5-32B Base | < 0.7623 | < 72% | < 95.25% | Fine-tuned보다 나을 것으로 예상 |
| DeepSeek-R1 Base | ? | ? | ? | Qwen과 비교 |
| EXAONE-3.5-32B | ? | ? | ? | 한국어 최고 성능 모델 |
| GPT-OSS-20B | ? | ? | ? | OpenAI 오픈소스 |
| Kimi K2 | ? | ? | ? | Multi-Agent 특화 |

---

## 🔍 분석 항목

### 각 모델별 분석
1. **NLL 성능**: 랜덤 baseline과 비교
2. **Generation Bias**: A vs B 생성 분포
3. **Feature Similarity**: Hidden state 유사도
4. **Accuracy**: 예측 정확도
5. **AUC**: ROC 곡선 아래 면적

### 비교 분석
1. **Base vs Fine-tuned**: 편향 문제 비교
2. **모델 아키텍처별**: Qwen vs DeepSeek vs EXAONE 비교
3. **파라미터 크기별**: 20B vs 32B 비교
4. **한국어 성능**: EXAONE의 한국어 특화 효과

---

## ⚠️ 주의사항 및 리스크

### 기술적 이슈
1. **GPU 호환성**: CUDA capability 12.1 지원 문제
   - 해결: NGC PyTorch 컨테이너 사용 또는 CPU fallback
2. **메모리 부족**: 대형 모델 로딩 시 OOM 가능
   - 해결: 양자화 사용 (NF4, 8-bit)
3. **실행 시간**: 각 실험은 수 시간 소요
   - 해결: tmux 세션 사용, 백그라운드 실행

### 데이터 이슈
1. **Generation Bias**: Fine-tuned 모델에서 발견된 편향
   - 해결: Base 모델 사용
2. **Feature Similarity**: 높은 유사도로 인한 성능 저하
   - 해결: 다양한 모델 테스트

### 리소스 관리
1. **저장공간**: 모델별 필요 공간 확인 필요
   - Kimi K2: ~500GB (INT4 기준)
2. **GPU 메모리**: 7x RTX GPU (24GB 각) = 168GB total
3. **실행 시간**: 각 실험 3-6시간 소요

---

## 🛠️ 실행 가이드

### 즉시 실행 가능한 작업

#### 1. Qwen2.5-32B Base LOO CV (가장 우선)
```bash
cd ~/git/CENTaUR

# tmux 세션 생성 (연결 끊겨도 계속 실행)
tmux new -s qwen25_base_cv

# Feature 파일 확인
ls -lh outputs/qwen25_base_features.npz

# LOO CV 실행
python scripts/fit_centaur_loo_cv_flexible.py \
    outputs/qwen25_base_features.npz \
    outputs/qwen25_base_nll_results.json \
    --model_name "Qwen2.5-32B-Base"

# Detach: Ctrl+B, D
# Reattach: tmux attach -t qwen25_base_cv
```

#### 2. DeepSeek-R1 Base LOO CV (병렬 실행 가능)
```bash
# 별도 tmux 세션
tmux new -s deepseek_base_cv

# Feature 파일 확인
ls -lh outputs/deepseek_base_features.npz

# LOO CV 실행
python scripts/fit_centaur_loo_cv_flexible.py \
    outputs/deepseek_base_features.npz \
    outputs/deepseek_base_nll_results.json \
    --model_name "DeepSeek-R1-32B-Base"
```

#### 3. EXAONE-3.5-32B Feature Extraction
```bash
# tmux 세션 생성
tmux new -s exaone35_extract

# Feature Extraction 실행
./run_exaone35_full.sh

# 또는 직접 실행
python scripts/extract_centaur_features.py \
    --model exaone35 \
    --output outputs/exaone35_features.npz
```

### 모니터링 명령어

```bash
# 실험 진행 상황 확인
tail -f logs/qwen25_base_loo_cv.log
tail -f logs/deepseek_base_loo_cv.log
tail -f logs/exaone35_extraction.log

# GPU 사용량 확인
watch -n 1 nvidia-smi

# 프로세스 확인
ps aux | grep fit_centaur_loo_cv
ps aux | grep extract_centaur_features

# 결과 파일 확인
ls -lh outputs/*.json
ls -lh outputs/*.npz
```

---

## 📈 진행 상황 추적

### 체크리스트

#### Phase 1: 즉시 실행 가능
- [ ] Qwen2.5-32B Base LOO CV 실행
- [ ] DeepSeek-R1 Base LOO CV 실행
- [ ] EXAONE-3.5-32B Feature Extraction 재실행

#### Phase 2: 다음 주
- [ ] EXAONE-3.5-32B LOO CV 실행
- [ ] GPT-OSS-20B Feature Extraction
- [ ] GPT-OSS-20B LOO CV 실행
- [ ] Base 모델 결과 비교 분석

#### Phase 3: 조건부
- [ ] Kimi K2 저장공간 확인
- [ ] Kimi K2 Feature Extraction (조건부)
- [ ] Kimi K2 LOO CV (조건부)

#### 최종 분석
- [ ] 모든 모델 성능 비교표 작성
- [ ] Generation bias 분석 리포트
- [ ] Feature similarity 분석 리포트
- [ ] 종합 결과 리포트 작성
- [ ] 논문 초안 작성

---

## 🎯 성공 기준

### Week 1 성공 기준
- ✅ Qwen2.5-32B Base LOO CV 완료
- ✅ DeepSeek-R1 Base LOO CV 완료
- ✅ EXAONE-3.5-32B Feature Extraction 완료
- ✅ Base 모델 성능 확인

### Week 2 성공 기준
- ✅ EXAONE-3.5-32B LOO CV 완료
- ✅ GPT-OSS-20B 실험 완료
- ✅ Base 모델 비교 분석 완료

### Week 3 성공 기준
- ✅ 모든 실험 완료 (Kimi K2 제외 가능)
- ✅ 종합 성능 비교표 작성
- ✅ 결과 분석 리포트 작성

### Week 4 성공 기준
- ✅ 논문 초안 작성 완료
- ✅ 실험 결과 정리 완료
- ✅ 향후 연구 방향 제시

---

## 📝 다음 단계 (즉시 실행)

### 오늘 바로 시작할 수 있는 작업

1. **Qwen2.5-32B Base LOO CV 시작** (가장 우선)
   ```bash
   cd ~/git/CENTaUR
   tmux new -s qwen25_base_cv
   python scripts/fit_centaur_loo_cv_flexible.py \
       outputs/qwen25_base_features.npz \
       outputs/qwen25_base_nll_results.json \
       --model_name "Qwen2.5-32B-Base"
   ```

2. **DeepSeek-R1 Base LOO CV 시작** (병렬 실행 가능)
   ```bash
   tmux new -s deepseek_base_cv
   python scripts/fit_centaur_loo_cv_flexible.py \
       outputs/deepseek_base_features.npz \
       outputs/deepseek_base_nll_results.json \
       --model_name "DeepSeek-R1-32B-Base"
   ```

3. **EXAONE-3.5-32B Feature Extraction 재실행**
   ```bash
   tmux new -s exaone35_extract
   ./run_exaone35_full.sh
   ```

---

## 📚 참고 문서

- **코드 온보딩**: `CODE_ONBOARDING.md`
- **작업 상태**: `WORK_STATUS.md`
- **이전 계획**: `WORK_PLAN.md`
- **실험 계획 리뷰**: `EXPERIMENT_PLAN_REVIEW.md`
- **Generation Bias 분석**: `claudedocs/GENERATION_BIAS_ANALYSIS_2025-11-06.md`
- **종합 분석**: `claudedocs/COMPREHENSIVE_FINDINGS_2025-11-06.md`

---

**작성일**: 2025-11-12  
**다음 리뷰**: 매주 금요일  
**업데이트**: 실험 진행에 따라 주기적으로 업데이트

