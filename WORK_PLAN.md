# CENTaUR 프로젝트 작업 계획
**작성일**: 2025-11-08  
**목적**: CENTaUR 실험 완료 및 결과 분석을 위한 체계적인 작업 계획

---

## 📊 현재 상태

### 완료된 작업
- ✅ Feature Extraction: Qwen2.5-32B, DeepSeek-R1 (Base)
- ✅ LOO CV: Qwen2.5-32B (50/100 folds, 중단됨)
- ✅ Generation Bias 분석: Qwen2.5-32B (72% B 편향 발견)
- ✅ 모델 추가: GPT-OSS-20B, EXAONE-3.5-32B, Kimi K2

### 발견된 문제
- ⚠️ Qwen2.5-32B: Generation bias (72% → B), NLL = 0.7623 (랜덤보다 나쁨)
- ⚠️ Feature similarity: 95.25% (너무 높음)
- ⚠️ Fine-tuned 모델의 편향 문제

### 추가된 모델
- ✅ EXAONE-3.5-32B (한국어 최고 성능, MMLU-Pro 81.8%)
- ✅ Kimi K2 (Multi-Agent 특화, 128K context)
- ✅ GPT-OSS-20B (이미 다운로드됨)

---

## 🎯 목표

1. **기존 모델 문제 해결**: Generation bias 원인 분석 및 해결
2. **새 모델 실험**: EXAONE-3.5-32B, Kimi K2로 feature extraction 및 LOO CV
3. **성능 비교**: 모든 모델의 NLL 성능 비교
4. **결과 분석**: Base vs Fine-tuned, 모델별 성능 분석
5. **논문 준비**: 실험 결과 정리 및 논문 작성 준비

---

## 📅 작업 계획 (4주)

### Week 1: 기존 모델 문제 해결 및 Base 모델 실험

#### Day 1-2: 기존 모델 재분석
- [ ] **Qwen2.5-32B Base 모델 테스트**
  - Fine-tuned 대신 Base 모델로 feature extraction
  - Generation bias 비교 분석
  - 목표: Fine-tuned 편향 문제 확인

- [ ] **DeepSeek-R1 Base 모델 테스트**
  - Base 모델로 feature extraction
  - Generation bias 확인
  - 목표: 모델별 편향 패턴 비교

#### Day 3-4: EXAONE-3.5-32B 실험 시작
- [ ] **EXAONE-3.5-32B Feature Extraction**
  ```bash
  ./run_exaone35_full.sh
  ```
  - 예상 시간: 2-4시간
  - 목표: 한국어 최고 성능 모델로 실험

- [ ] **EXAONE-3.5-32B Generation Bias 분석**
  - 생성된 토큰 분석
  - 편향 확인 (A vs B)
  - Feature similarity 확인

#### Day 5-7: Base 모델 LOO CV
- [ ] **Qwen2.5-32B Base LOO CV**
  ```bash
  python scripts/fit_centaur_loo_cv_flexible.py \
      outputs/qwen25_base_features.npz \
      outputs/qwen25_base_nll_results.json \
      --model_name "Qwen2.5-32B-Base"
  ```
  - 예상 시간: 3-4시간
  - 목표: Base 모델 성능 확인

- [ ] **DeepSeek-R1 Base LOO CV**
  ```bash
  python scripts/fit_centaur_loo_cv_flexible.py \
      outputs/deepseek_base_features.npz \
      outputs/deepseek_base_nll_results.json \
      --model_name "DeepSeek-R1-32B-Base"
  ```
  - 예상 시간: 3-4시간

**Week 1 목표**: Base 모델 성능 확인, EXAONE-3.5-32B 실험 시작

---

### Week 2: 새 모델 실험 및 비교 분석

#### Day 8-10: EXAONE-3.5-32B 완료
- [ ] **EXAONE-3.5-32B LOO CV**
  ```bash
  ./run_exaone35_loo_cv.sh
  ```
  - 예상 시간: 3-4시간
  - 목표: 한국어 최고 성능 모델 성능 확인

- [ ] **EXAONE-3.5-32B 결과 분석**
  - NLL 성능 확인
  - Generation bias 확인
  - Feature similarity 확인
  - Base 모델들과 비교

#### Day 11-12: GPT-OSS-20B 실험
- [ ] **GPT-OSS-20B Feature Extraction**
  ```bash
  ./run_gpt_oss_full.sh
  ```
  - 예상 시간: 2-4시간
  - 목표: OpenAI 오픈소스 모델 성능 확인

- [ ] **GPT-OSS-20B LOO CV**
  ```bash
  ./run_gpt_oss_loo_cv.sh
  ```
  - 예상 시간: 3-4시간

#### Day 13-14: Kimi K2 준비 및 시작
- [ ] **저장공간 확인**
  ```bash
  df -h
  ```
  - 필요: ~500GB (INT4 기준)
  - 목표: 다운로드 가능 여부 확인

- [ ] **Kimi K2 Feature Extraction 시작** (조건부)
  ```bash
  ./run_kimi_k2_full.sh
  ```
  - 예상 시간: 4-6시간 (다운로드 포함)
  - 목표: Multi-Agent 특화 모델 실험

**Week 2 목표**: EXAONE-3.5-32B 완료, GPT-OSS-20B 실험, Kimi K2 시작

---

### Week 3: Kimi K2 완료 및 종합 분석

#### Day 15-17: Kimi K2 완료
- [ ] **Kimi K2 Feature Extraction 완료**
  - 다운로드 및 extraction 모니터링
  - 결과 확인

- [ ] **Kimi K2 LOO CV**
  ```bash
  ./run_kimi_k2_loo_cv.sh
  ```
  - 예상 시간: 3-4시간
  - 목표: Multi-Agent 모델 성능 확인

- [ ] **Kimi K2 결과 분석**
  - NLL 성능 확인
  - 한국어 성능 확인 (테스트 필요)
  - 다른 모델들과 비교

#### Day 18-21: 종합 분석 및 비교
- [ ] **모든 모델 성능 비교표 작성**
  | 모델 | 파라미터 | NLL | Generation Bias | Feature Similarity | 한국어 성능 |
  |------|----------|-----|-----------------|-------------------|-------------|
  | Qwen2.5-32B Base | 32B | ? | ? | ? | ⭐⭐⭐⭐ |
  | Qwen2.5-32B Fine-tuned | 32B | 0.7623 | 72% → B | 95.25% | ⭐⭐⭐⭐ |
  | DeepSeek-R1 Base | 32B | ? | ? | ? | ⭐⭐⭐ |
  | EXAONE-3.5-32B | 32B | ? | ? | ? | ⭐⭐⭐⭐⭐ |
  | GPT-OSS-20B | 20B | ? | ? | ? | ⭐⭐⭐⭐ |
  | Kimi K2 | 1T (32B 활성) | ? | ? | ? | ⚠️ 확인 필요 |

- [ ] **Base vs Fine-tuned 비교 분석**
  - Fine-tuned 편향 문제 원인 분석
  - Base 모델이 더 나은 성능을 보이는지 확인
  - Fine-tuning 방법론 개선 방안 제시

- [ ] **모델별 특징 분석**
  - Generation bias 패턴
  - Feature similarity 패턴
  - 한국어 성능 비교

**Week 3 목표**: 모든 실험 완료, 종합 분석

---

### Week 4: 결과 정리 및 논문 준비

#### Day 22-24: 결과 리포트 작성
- [ ] **실험 결과 리포트 생성**
  ```bash
  python generate_report.py
  ```
  - 모든 모델 결과 통합
  - 성능 비교 차트 생성
  - 통계 분석

- [ ] **상세 분석 리포트 작성**
  - 모델별 성능 분석
  - Generation bias 분석
  - Feature similarity 분석
  - 한국어 성능 분석

#### Day 25-28: 논문 준비
- [ ] **실험 방법론 정리**
  - Feature extraction 방법
  - LOO CV 방법
  - 평가 지표 설명

- [ ] **결과 섹션 작성**
  - 모델별 성능 비교
  - Base vs Fine-tuned 비교
  - 한국어 모델 성능 분석

- [ ] **토론 및 결론 작성**
  - 발견된 문제점
  - 개선 방안
  - 향후 연구 방향

**Week 4 목표**: 결과 정리, 논문 초안 작성

---

## 🎯 우선순위별 작업

### Priority 1: 즉시 시작 (이번 주)
1. **EXAONE-3.5-32B Feature Extraction** ⭐
   - 가장 빠르게 시작 가능
   - 한국어 최고 성능 모델
   - 예상 시간: 2-4시간

2. **Qwen2.5-32B Base 모델 테스트**
   - Fine-tuned 편향 문제 해결을 위한 비교
   - 예상 시간: 2-4시간 (extraction) + 3-4시간 (LOO CV)

3. **DeepSeek-R1 Base 모델 테스트**
   - Base 모델 성능 확인
   - 예상 시간: 2-4시간 (extraction) + 3-4시간 (LOO CV)

### Priority 2: 다음 주
4. **EXAONE-3.5-32B LOO CV**
   - 한국어 최고 성능 모델 성능 확인
   - 예상 시간: 3-4시간

5. **GPT-OSS-20B 실험**
   - OpenAI 오픈소스 모델 성능 확인
   - 예상 시간: 2-4시간 (extraction) + 3-4시간 (LOO CV)

### Priority 3: 조건부 (저장공간 확인 후)
6. **Kimi K2 실험**
   - Multi-Agent 특화 모델
   - 매우 큰 모델 (저장공간 확인 필요)
   - 예상 시간: 4-6시간 (extraction) + 3-4시간 (LOO CV)

---

## 📋 일일 작업 체크리스트

### 오늘 할 일 (Day 1)
- [ ] EXAONE-3.5-32B Feature Extraction 시작
  ```bash
  cd ~/git/CENTaUR
  tmux new -s exaone35_exp
  ./run_exaone35_full.sh
  # Ctrl+b, d로 detach
  ```

- [ ] Qwen2.5-32B Base 모델 확인
  - Base 모델 feature extraction 실행 여부 확인
  - 없으면 실행 시작

- [ ] 실험 진행 상황 모니터링
  ```bash
  # EXAONE-3.5-32B 진행 상황 확인
  tail -f logs/exaone35_extraction.log
  
  # GPU 사용량 확인
  watch -n 1 nvidia-smi
  ```

### 내일 할 일 (Day 2)
- [ ] EXAONE-3.5-32B 결과 확인
  ```bash
  python -c "
  import numpy as np
  data = np.load('outputs/exaone35_features.npz')
  print(f'Features: {data[\"features\"].shape}')
  print(f'Labels: {data[\"labels\"].shape}')
  "
  ```

- [ ] EXAONE-3.5-32B Generation Bias 분석
  ```bash
  python scripts/analyze_full_generation.py --model exaone35-base
  ```

- [ ] Qwen2.5-32B Base Feature Extraction 시작 (아직 안 했다면)

---

## 🔧 실험 실행 가이드

### Feature Extraction 실행 순서
1. **EXAONE-3.5-32B** (가장 우선)
   ```bash
   ./run_exaone35_full.sh
   ```

2. **Qwen2.5-32B Base** (비교용)
   ```bash
   python scripts/extract_centaur_features.py \
       --model qwen25-base \
       --dataset ko_centaur/data/choices13k_100.jsonl \
       --output outputs/qwen25_base_features.npz
   ```

3. **DeepSeek-R1 Base** (비교용)
   ```bash
   python scripts/extract_centaur_features.py \
       --model deepseek-base \
       --dataset ko_centaur/data/choices13k_100.jsonl \
       --output outputs/deepseek_base_features.npz
   ```

4. **GPT-OSS-20B** (이미 다운로드됨)
   ```bash
   ./run_gpt_oss_full.sh
   ```

5. **Kimi K2** (저장공간 확인 후)
   ```bash
   ./run_kimi_k2_full.sh
   ```

### LOO CV 실행 순서
각 모델의 feature extraction 완료 후:
```bash
# EXAONE-3.5-32B
./run_exaone35_loo_cv.sh

# Qwen2.5-32B Base
python scripts/fit_centaur_loo_cv_flexible.py \
    outputs/qwen25_base_features.npz \
    outputs/qwen25_base_nll_results.json \
    --model_name "Qwen2.5-32B-Base"

# DeepSeek-R1 Base
python scripts/fit_centaur_loo_cv_flexible.py \
    outputs/deepseek_base_features.npz \
    outputs/deepseek_base_nll_results.json \
    --model_name "DeepSeek-R1-32B-Base"

# GPT-OSS-20B
./run_gpt_oss_loo_cv.sh

# Kimi K2
./run_kimi_k2_loo_cv.sh
```

---

## 📊 예상 결과 및 목표

### 성능 목표
- **Random Baseline**: NLL ≈ 0.6931
- **LLaMA-65B (원본)**: NLL ≈ 30K (로그 스케일)
- **목표**: NLL < 0.6931 (랜덤보다 좋은 성능)

### 예상 결과
1. **EXAONE-3.5-32B**: 가장 좋은 성능 기대 (한국어 최고 성능)
2. **Base 모델들**: Fine-tuned보다 편향이 적을 것으로 예상
3. **Kimi K2**: Multi-Agent 특화로 복잡한 추론에 유리할 수 있음

---

## ⚠️ 주의사항

### 리소스 관리
- **GPU 메모리**: 7x RTX GPU (24GB 각) = 168GB total
- **저장공간**: 모델별 필요 공간 확인
- **실행 시간**: 각 실험은 수 시간 소요

### 모니터링
- tmux 세션 사용 (연결 끊겨도 계속 실행)
- 로그 파일 모니터링
- GPU 사용량 확인

### 백업
- 중요한 결과 파일 백업
- 중간 체크포인트 저장

---

## 📝 다음 단계 (즉시 실행)

### 오늘 바로 시작할 수 있는 작업

1. **EXAONE-3.5-32B Feature Extraction 시작**
   ```bash
   cd ~/git/CENTaUR
   tmux new -s exaone35_exp
   ./run_exaone35_full.sh
   ```

2. **기존 Base 모델 확인**
   ```bash
   ls -lh outputs/*base*.npz
   ```

3. **실험 진행 상황 모니터링 설정**
   ```bash
   # 별도 터미널에서
   watch -n 60 'tail -20 logs/exaone35_extraction.log'
   ```

---

## 🎯 성공 기준

### Week 1 성공 기준
- ✅ EXAONE-3.5-32B Feature Extraction 완료
- ✅ Qwen2.5-32B Base 모델 테스트 완료
- ✅ 최소 2개 모델의 LOO CV 완료

### Week 2 성공 기준
- ✅ EXAONE-3.5-32B LOO CV 완료
- ✅ GPT-OSS-20B 실험 완료
- ✅ Base vs Fine-tuned 비교 분석 완료

### Week 3 성공 기준
- ✅ 모든 모델 실험 완료
- ✅ 종합 성능 비교표 작성
- ✅ 결과 분석 리포트 작성

### Week 4 성공 기준
- ✅ 논문 초안 작성
- ✅ 실험 결과 정리
- ✅ 향후 연구 방향 제시

---

**작성일**: 2025-11-08  
**다음 리뷰**: 매주 금요일  
**담당자**: 연구팀

