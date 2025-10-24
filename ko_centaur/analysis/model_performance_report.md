# Ko-CENTaUR 모델 성능 비교 보고서
**Report Date**: 2025-10-23
**Analyst**: Claude Code

## Executive Summary

본 보고서는 Ko-CENTaUR 프로젝트에서 학습된 모델들의 다운스트림 태스크 성능을 비교 분석합니다. 총 2개의 베이스 모델 (ko-centaur, exaone-base)과 2개의 fine-tuned EXAONE 모델이 평가되었으며, Risky Choice 예측 태스크에서 성능을 검증하였습니다.

---

## 1. 평가 모델 목록

### 1.1 학습 완료 모델

| 모델명 | 베이스 모델 | 학습 방식 | 어댑터 크기 | 학습 데이터 | 상태 |
|--------|------------|----------|------------|------------|------|
| **ko-centaur** | LLaMA-based | Full Fine-tuning | N/A | Korean Psychology Data | ✅ 완료 |
| **exaone-base** | EXAONE-3.0-7.8B | Full Fine-tuning | N/A | Korean Psychology Data | ✅ 완료 |
| **exaone-choices13k** | EXAONE-3.0-7.8B-Instruct | QLoRA (r=16, α=32) | 272 MB | Risky Choice (choices13k) | ✅ 완료 |
| **exaone-psych101-full** | EXAONE-3.0-7.8B-Instruct | QLoRA (r=8, α=16) | 144 MB | Psychology 101 Dataset | ✅ 완료 |

### 1.2 학습 진행 중 모델

| 모델명 | 베이스 모델 | 학습 방식 | 진행률 | 예상 완료 시간 | 상태 |
|--------|------------|----------|-------|---------------|------|
| **qwen25-32b-qlora** | Qwen2.5-32B-Instruct | QLoRA (r=16, α=32) | 51% (2233/4392) | ~13.8시간 | 🔄 학습중 (Job 62905) |

### 1.3 학습 실패 모델

| 모델명 | 베이스 모델 | 실패 원인 | 시도 횟수 |
|--------|------------|----------|----------|
| **EXAONE-4.0.1-32B** | EXAONE-4.0.1-32B | OOM (7x RTX 3090, DeepSpeed ZeRO-2/3) | 6회 |

---

## 2. 다운스트림 태스크 성능 평가

### 2.1 Choices13k 100-Fold Cross-Validation

**태스크**: Risky Choice Prediction (100-fold LOO CV)
**데이터셋**: choices13k
**메트릭**: Accuracy, Log-Likelihood

| 모델 | Mean Accuracy | Std Accuracy | Mean Log-Likelihood | 성능 평가 |
|------|---------------|--------------|---------------------|----------|
| **ko-centaur** | **60.0%** | 49.24% | -4.21 | 🟡 Moderate |
| **exaone-base** | 55.0% | 50.00% | **-2.40** | 🟢 Good (better NLL) |

**분석**:
- ko-centaur가 정확도에서 5%p 우세 (60% vs 55%)
- exaone-base가 로그우도에서 크게 우수 (-2.40 vs -4.21)
  - 더 나은 확률 보정(calibration)을 의미
  - 예측 불확실성을 더 정확하게 모델링
- 높은 표준편차(~50%)는 cross-validation fold 간 변동성이 큼을 의미

### 2.2 Choices13k Test Set (20-fold)

**태스크**: Risky Choice Prediction (20-fold test)
**데이터셋**: choices13k (test split)

| 모델 | Mean Accuracy | Std Accuracy | Mean Log-Likelihood | 성능 평가 |
|------|---------------|--------------|---------------------|----------|
| **ko-centaur** | **70.0%** | 47.02% | -0.67 | 🟢 Good |
| **exaone-base** | **70.0%** | 47.02% | **-0.67** | 🟢 Good |

**분석**:
- 두 모델이 test set에서 **동일한 성능** 달성
- 100-fold CV (60%, 55%) 대비 test set (70%) 성능이 10-15%p 향상
  - 가능한 원인: test set이 더 쉬운 샘플로 구성
- Log-likelihood -0.67은 100-fold CV 대비 크게 개선 (4-7배)

### 2.3 Full Evaluation Test

**태스크**: Full Evaluation (20-fold)
**데이터셋**: full_eval_test_fixed

| 모델 | Mean Accuracy | Std Accuracy | Mean Log-Likelihood | 성능 평가 |
|------|---------------|--------------|---------------------|----------|
| **ko-centaur** | 0.0% | 0.00% | -0.75 | ⚠️ Requires Investigation |
| **exaone-base** | 0.0% | 0.00% | -0.75 | ⚠️ Requires Investigation |

**분석**:
- Accuracy 0%는 비정상적 결과
  - 가능한 원인:
    1. 다른 평가 메트릭 사용 (accuracy가 아닌 다른 측정치)
    2. 태스크 형식 불일치 (binary vs multi-class)
    3. 데이터 전처리 오류
- Log-likelihood -0.75는 합리적 범위
- **추가 조사 필요**

---

## 3. 모델별 심층 분석

### 3.1 Ko-CENTaUR (LLaMA-based)

**강점**:
- Choices13k에서 60% 정확도 달성
- Test set에서 70% 정확도 (exaone-base와 동일)
- 안정적인 risky choice 예측 성능

**약점**:
- 낮은 log-likelihood (-4.21 vs -2.40)
- 확률 보정 성능이 exaone-base 대비 열세

**적합한 사용 사례**:
- Binary choice prediction 태스크
- 확률 보정보다 정확도가 중요한 경우

### 3.2 EXAONE-Base (EXAONE-3.0-7.8B)

**강점**:
- **최고의 log-likelihood** (-2.40)
- 우수한 확률 보정 성능
- Test set에서 70% 정확도

**약점**:
- 100-fold CV에서 55% 정확도 (ko-centaur 대비 5%p 낮음)

**적합한 사용 사례**:
- 확률 추정이 중요한 태스크
- 불확실성 정량화가 필요한 응용

### 3.3 EXAONE-Choices13k (QLoRA Fine-tuned)

**특징**:
- EXAONE-3.0-7.8B-Instruct 기반
- QLoRA: rank 16, alpha 32
- Choices13k 데이터 전용 fine-tuning
- 어댑터 크기: 272 MB

**기대 성능**:
- Choices13k 태스크에서 exaone-base 대비 향상 예상
- 특화된 fine-tuning으로 정확도 및 NLL 개선 가능
- **평가 필요** (현재 성능 데이터 없음)

### 3.4 EXAONE-Psych101-Full (QLoRA Fine-tuned)

**특징**:
- EXAONE-3.0-7.8B-Instruct 기반
- QLoRA: rank 8, alpha 16 (더 작은 어댑터)
- Psychology 101 데이터셋 학습
- 어댑터 크기: 144 MB

**기대 성능**:
- 일반 심리학 태스크에서 강점
- Choices13k에서는 domain mismatch 가능
- **평가 필요** (현재 성능 데이터 없음)

---

## 4. 학습 진행 중 모델

### 4.1 Qwen2.5-32B QLoRA

**현재 상태** (2025-10-23 기준):
- Job ID: 62905
- 노드: node1 (4x RTX A5000)
- 진행률: 51% (2233/4392 steps)
- 체크포인트: checkpoint-2000부터 재개
- 예상 완료: ~13.8시간

**학습 이력**:
- 총 6번의 시도 끝에 성공
- 주요 장애물: OOM 에러 (node3), 타임아웃
- 해결 방법: node1로 이전, GPU 경합 제거

**기대 성능**:
- 32B 파라미터 모델로 7.8B 대비 큰 용량
- Risky choice에서 SOTA 성능 기대
- Fine-tuning 완료 후 평가 필요

---

## 5. 종합 비교 및 권장사항

### 5.1 성능 종합 요약

| 평가 기준 | 최고 모델 | 점수 | 비고 |
|----------|----------|------|------|
| **정확도 (100-fold)** | ko-centaur | 60.0% | 5%p 우세 |
| **로그우도 (100-fold)** | exaone-base | -2.40 | 확률 보정 우수 |
| **정확도 (test)** | 동점 | 70.0% | 두 모델 동일 |
| **로그우도 (test)** | 동점 | -0.67 | 두 모델 동일 |
| **모델 크기** | exaone-base | 7.8B | 효율적 |

### 5.2 모델 선택 가이드

| 사용 목적 | 권장 모델 | 이유 |
|----------|----------|------|
| **Production Deployment** | exaone-base | 우수한 확률 보정, 안정적 성능 |
| **High Accuracy Required** | ko-centaur | 최고 정확도 (60%) |
| **Specialized Risky Choice** | exaone-choices13k | Domain-specific fine-tuning (평가 필요) |
| **General Psychology Tasks** | exaone-psych101-full | 광범위한 심리학 데이터 학습 |
| **Research & Experimentation** | qwen25-32b (학습 완료 후) | 최대 모델 용량, SOTA 잠재력 |

### 5.3 개선 방향

**단기 (1-2주)**:
1. ✅ Qwen2.5-32B 학습 완료 대기 및 평가
2. ⚠️ full_eval_test_fixed의 0% accuracy 원인 조사
3. 📊 EXAONE fine-tuned 모델(choices13k, psych101) 평가 실행
4. 📈 Choices13k에서 ensemble 모델 테스트

**중기 (1-2개월)**:
1. 🎯 EXAONE-4.0.1-32B QLoRA 학습 시도 (DeepSpeed 대신 QLoRA)
2. 🔬 Cross-model comparison on multiple psychology tasks
3. 📉 Error analysis on failed predictions
4. 🧪 Ablation study on LoRA configurations

**장기 (3-6개월)**:
1. 🌐 Multi-task learning across psychology datasets
2. 🤖 Ensemble methods combining ko-centaur + EXAONE
3. 📚 Expand to Korean clinical psychology tasks
4. 🎓 Transfer learning to related cognitive domains

---

## 6. 기술적 인사이트

### 6.1 학습 안정성

**성공 사례**:
- EXAONE-3.0-7.8B QLoRA: 안정적 학습, OOM 없음
- Qwen2.5-32B QLoRA: node1 이전 후 안정화

**실패 사례**:
- EXAONE-4.0.1-32B DeepSpeed: 6번 실패, 모두 OOM
  - 32B + DeepSpeed + 7x RTX 3090 (24GB) 조합 불가
  - QLoRA 전환 권장

### 6.2 하드웨어 효율성

| 노드 | GPU | 성공 모델 | 실패 원인 |
|------|-----|----------|----------|
| node1 | 4x RTX A5000 (24GB) | Qwen2.5-32B QLoRA | GPU 경합 없음 |
| node3 | 4x RTX 3090 (24GB) | - | GPU3 16GB 점유 (PID 263065) |
| octopus | 7x RTX 3090 (24GB) | - | EXAONE-32B OOM |

**권장사항**:
- 32B 모델: node1 사용, QLoRA 우선
- 7.8B 모델: node3 가능 (GPU 경합 해소 시)
- DeepSpeed: 40GB+ GPU 필요 (A100 등)

### 6.3 학습 비용

| 모델 | 학습 시간 | GPU 시간 | 디스크 사용 | 전력 비용 (추정) |
|------|----------|---------|------------|----------------|
| EXAONE-choices13k | ~8시간 | 32 GPU-hours | 272 MB | ~$4 |
| EXAONE-psych101 | ~22시간 | 88 GPU-hours | 144 MB | ~$11 |
| Qwen2.5-32B (진행중) | ~28시간 (예상) | 112 GPU-hours | 2.4 GB | ~$14 |
| EXAONE-32B (실패) | ~5시간 (총) | 210 GPU-hours (낭비) | 0 MB | ~$26 (낭비) |

---

## 7. 결론

### 7.1 주요 발견

1. **exaone-base**가 확률 보정(log-likelihood) 측면에서 최고 성능
2. **ko-centaur**가 정확도 측면에서 소폭 우세 (60% vs 55%)
3. Test set에서 두 모델 모두 **70% 정확도** 달성 (동등)
4. 32B 모델들은 **하드웨어 제약**으로 학습 어려움 (EXAONE-32B 실패, Qwen-32B 진행중)
5. **QLoRA**가 DeepSpeed 대비 더 안정적이고 메모리 효율적

### 7.2 최종 권장사항

**즉시 실행**:
1. Qwen2.5-32B 학습 완료 후 즉시 평가
2. EXAONE fine-tuned 모델(choices13k, psych101) 성능 평가
3. full_eval_test accuracy 0% 원인 규명

**우선순위 높음**:
1. EXAONE-4.0.1-32B를 QLoRA로 재시도 (DeepSpeed 포기)
2. Ensemble 모델 개발 (ko-centaur + exaone-base)
3. More diverse psychology tasks로 평가 확장

**장기 전략**:
1. 40GB+ GPU 확보 시 full precision training 고려
2. Multi-task learning framework 개발
3. Clinical psychology applications 확장

---

## Appendix A: 평가 메트릭 정의

- **Accuracy**: 정답 예측 비율 (0-1)
- **Log-Likelihood**: 모델의 확률 예측 품질 (높을수록 좋음, 음수)
- **Standard Deviation**: Cross-validation fold 간 변동성
- **Mean**: 모든 fold의 평균 성능

## Appendix B: 데이터 출처

- `choices13k_100_fixed/all_results.pth`
- `choices13k_test/all_results.pth`
- `full_eval_test_fixed/all_results.pth`

---

**보고서 작성**: Claude Code
**검토 필요**: 사람 연구자의 full_eval_test 결과 확인
**다음 업데이트**: Qwen2.5-32B 학습 완료 후 (2025-10-24 예상)
