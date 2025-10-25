# CENTaUR Downstream Tasks 종합 분석
**작성일**: 2025-10-25
**출처**: Binz et al. (2025) "A foundation model to predict and capture human cognition", Nature

## Executive Summary

CENTaUR는 **Psych-101**이라는 대규모 데이터셋으로 학습되었으며, 이 데이터셋은 **160개의 심리학 실험**에서 수집된 **60,092명의 참가자**의 **10,681,650개 선택**을 포함합니다. 본 문서는 Ko-CENTaUR 프로젝트에서 다양한 downstream task 평가를 수행하기 위한 참고 자료로 작성되었습니다.

---

## 1. Psych-101 데이터셋 개요

### 1.1 규모 및 범위

| 메트릭 | 수치 |
|--------|------|
| **총 실험 수** | 160개 |
| **참가자 수** | 60,092명 |
| **총 선택 수** | 10,681,650개 |
| **텍스트 토큰 수** | 253,597,411개 |
| **도메인 수** | 6개 주요 도메인 + 기타 |

### 1.2 주요 도메인 분포

논문 Extended Data Fig. 1에 따르면:

1. **Multi-armed Bandits** (다중 슬롯머신 과제)
2. **Decision-making** (의사결정)
3. **Memory** (기억)
4. **Supervised Learning** (지도학습)
5. **Markov Decision Processes** (마르코프 의사결정 과정)
6. **Miscellaneous** (기타 인지 과제)

---

## 2. 도메인별 Downstream Tasks 상세 분석

### 2.1 Multi-Armed Bandits (강화학습 과제)

#### **대표 실험**:

| 실험명 | 설명 | 평가 메트릭 | Centaur NLL | Cognitive Model NLL |
|--------|------|-------------|-------------|---------------------|
| **Horizon Task** | 2-armed bandit with exploration-exploitation trade-off | Accuracy, Information bonus, Reward | 0.4032 | 0.3595 |
| **Two-armed Bandit** | Standard 2-armed bandit | Log-likelihood | 0.2963 | 0.4187 |
| **Drifting Four-armed Bandit** | 4-armed bandit with non-stationary rewards | Log-likelihood | 0.7029 | 0.9043 |
| **Changing Bandit** | Bandit with changing reward distributions | Log-likelihood | 0.3025 | 0.4378 |
| **Spatially Correlated Multi-armed Bandit** | Spatial correlation in reward structure | Log-likelihood | 1.8319 | 2.7635 |
| **Structured Bandit** | Hierarchical structure in options | Log-likelihood | 0.6410 | 1.0530 |
| **Zoopermarket** | Shopping-based bandit task | Log-likelihood | 0.4850 | 0.6047 |
| **Gardening Task** | Contextual bandit with gardening theme | Log-likelihood | 0.3783 | 0.9105 |

**평가 방법**:
- Open-loop simulation으로 exploration strategy 검증
- Information bonus parameter 측정
- Reward maximization 평가

**Out-of-Distribution 평가**:
- **Maggie's Farm** (3-armed bandit, Psych-101에 없음)
  - Centaur NLL: **0.42**
  - Llama NLL: 0.62
  - Cognitive Model NLL: 0.98

---

### 2.2 Decision-Making (의사결정)

#### **대표 실험**:

| 실험명 | 설명 | 평가 메트릭 | Centaur NLL | Cognitive Model NLL |
|--------|------|-------------|-------------|---------------------|
| **Choices13k** | 13,000개 risky choice 문제 | Accuracy, Log-likelihood | 0.4274 | 0.6563 |
| **Decisions from Description** | 확률적 선택 (description-based) | Log-likelihood | 0.5336 | 0.6120 |
| **Decisions from Experience** | 확률적 선택 (experience-based) | Log-likelihood | 0.3686 | 0.5404 |
| **Intertemporal Choice** | 시간할인 의사결정 | Log-likelihood | 0.4340 | 0.6591 |
| **Risky Choice** | 위험 선택 과제 | Log-likelihood | 0.4281 | nan |
| **Iowa Gambling Task** | 장기 보상 학습 | Log-likelihood | 0.8890 | 1.1555 |
| **Balloon Analogue Risk Task (BART)** | 위험 감수 과제 | Log-likelihood | 0.0593 | 0.0922 |
| **Columbia Card Task** | 위험 의사결정 | Log-likelihood | 0.1867 | 0.2629 |
| **Multi-attribute Decision-making** | 다속성 의사결정 | Log-likelihood | 0.0619 | 0.1922 |

**평가 방법**:
- Prospect theory 모델과 비교
- Risk preference 측정
- Probability weighting function 분석

**Out-of-Distribution 평가**:
- **Moral Decision-Making** (Moral Machine experiment, 59개국)
  - Centaur NLL: **0.60**
  - Llama (70B) NLL: 0.91

---

### 2.3 Memory (기억)

#### **대표 실험**:

| 실험명 | 설명 | 평가 메트릭 | Centaur NLL | Cognitive Model NLL |
|--------|------|-------------|-------------|---------------------|
| **N-back** | Working memory 과제 (2-back) | Accuracy | 0.3954 | 0.5787 |
| **Recent Probes** | Short-term memory interference | Log-likelihood | 0.2572 | 0.3868 |
| **Digit Span** | Digit 기억 범위 | Log-likelihood | 0.5520 | 0.9359 |
| **Episodic Long-term Memory** | 장기 기억 인출 | Log-likelihood | 0.8684 | nan |
| **Recall and Recognition** | 회상 및 재인 | Log-likelihood | 1.0591 | nan |

**평가 방법**:
- Accuracy on memory retrieval
- Interference effects 분석

---

### 2.4 Supervised Learning (범주화)

#### **대표 실험**:

| 실험명 | 설명 | 평가 메트릭 | Centaur NLL | Cognitive Model NLL |
|--------|------|-------------|-------------|---------------------|
| **Shepard Categorization** | 6가지 논리 구조 기반 범주화 | Log-likelihood | 0.5394 | 0.6108 |
| **Medin Categorization** | 가족 유사성 범주화 | Log-likelihood | 0.4967 | 0.5313 |
| **Weather Prediction Task** | 확률적 범주 학습 | Log-likelihood | 0.5514 | 0.6267 |
| **THINGS Odd-one-out** | 자연물 범주화 (이질성 판단) | Log-likelihood | 0.8068 | 0.8253 |
| **Conditional Associative Learning** | 조건부 연합 학습 | Log-likelihood | 0.5380 | 0.8575 |
| **CPC18** | Complex categorization | Log-likelihood | 0.3390 | 0.6607 |

**평가 방법**:
- Generalized Context Model과 비교
- Category learning curves 분석

**Out-of-Distribution 평가**:
- **Naturalistic Category Learning**
  - Centaur NLL: **0.47**
  - Llama (70B) NLL: 0.56

---

### 2.5 Markov Decision Processes (순차적 의사결정)

#### **대표 실험**:

| 실험명 | 설명 | 평가 메트릭 | Centaur NLL | Cognitive Model NLL |
|--------|------|-------------|-------------|---------------------|
| **Two-step Task** | Model-based vs model-free RL | Model-basedness, Reward | 0.4998 | 0.6043 |
| **Probabilistic Instrumental Learning** | 도구적 학습 | Log-likelihood | 0.4937 | 0.5047 |
| **Multi-task Reinforcement Learning** | 다중 과제 RL | Log-likelihood | 0.5672 | 1.0424 |

**평가 방법**:
- Model-based learning parameter 추출
- Reward prediction error 분석
- Stay probability 측정

**Open-loop Simulation 결과** (Two-step task):
- Centaur와 인간 모두 **bimodal distribution** 생성
- Model-free, model-based, mixture 전략 모두 재현

**Out-of-Distribution 평가**:
- **Magic Carpet Cover Story** (Psych-101에 spaceship 버전만 있음)
  - Centaur NLL: **0.51**
  - Llama NLL: 0.63
  - Cognitive Model NLL: 0.61

---

### 2.6 Miscellaneous (기타 인지 과제)

#### **대표 실험**:

| 실험명 | 설명 | 평가 메트릭 | Centaur NLL | Cognitive Model NLL |
|--------|------|-------------|-------------|---------------------|
| **Go/No-go** | 반응 억제 과제 | Accuracy | 0.0000 | 0.0757 |
| **Serial Reaction Time Task** | 순차 학습 | Log-likelihood | 0.1718 | 0.1962 |
| **Grammar Judgement** | 문법 판단 | Log-likelihood | 1.4355 | 1.4127 |
| **Tile-revealing Task** | 탐색 과제 | Log-likelihood | 1.8713 | nan |
| **Virtual Subway Network** | 공간 탐색 | Log-likelihood | 1.1271 | nan |
| **Aversive Learning** | 혐오 학습 | Log-likelihood | 4.0733 | nan |
| **Probabilistic Reasoning** | 확률적 추론 | Log-likelihood | 2.3731 | nan |
| **Multiple-cue Judgement** | 다중 단서 판단 | Log-likelihood | 1.1236 | 1.9157 |

**Out-of-Distribution 평가**:
- **Logical Reasoning** (LSAT 문제, Psych-101에 없음)
  - Centaur NLL: **1.65**
  - Llama NLL: 1.92

- **Economic Games** (협력, 신뢰 게임)
  - Centaur NLL: **0.54**
  - Minitaur NLL: 0.56

- **Behavioral Propensities** (일상 행동 예측)
  - Centaur NLL: **1.43**
  - Llama NLL: 1.51

- **Deep Sequential Decision Task** (복잡한 MDP)
  - Centaur NLL: **0.96**
  - Llama NLL: 1.07

---

## 3. 평가 방법론

### 3.1 주요 평가 메트릭

| 메트릭 | 설명 | 사용 실험 |
|--------|------|-----------|
| **Negative Log-Likelihood (NLL)** | 모델의 확률 예측 품질 (낮을수록 좋음) | 모든 실험 |
| **Accuracy** | 정답 예측 비율 | Binary choice tasks |
| **Response Time Prediction** | Hick's law 기반 반응시간 예측 (R²) | Subset of experiments |
| **Open-loop Simulation** | 모델이 자체 응답으로 시뮬레이션 | Horizon, Two-step, Social prediction |
| **Parameter Recovery** | 인지 파라미터 추출 (e.g., information bonus) | Bandit tasks |

### 3.2 Cross-Validation 전략

**Held-out Participants**:
- 90% 학습, 10% 테스트 (각 실험마다)
- Leave-one-out cross-validation (100 folds)

**Held-out Experiments**:
- Cover story modification (magic carpet)
- Structural modification (3-armed bandit)
- Entirely new domains (logical reasoning, moral decision-making)

### 3.3 Noise Ceiling Analysis

**Choices13k**:
- Noise ceiling: ~0.7 NLL
- Centaur: **0.43 NLL** (ceiling 초과!)
- Centaur (independent prompts): 0.66 NLL

**Intertemporal Choice**:
- Noise ceiling: ~0.7 NLL
- Centaur: **0.44 NLL** (ceiling 초과!)
- Centaur (independent prompts): 0.69 NLL

→ Centaur는 context-dependent patterns을 포착하여 noise ceiling을 초과 가능

---

## 4. 성능 비교 요약

### 4.1 Overall Performance (Psych-101)

| 모델 | 평균 NLL | vs Llama Δ | vs Cognitive Model Δ | 승률 |
|------|---------|-----------|---------------------|------|
| **Centaur** | **0.44** | -0.14 (p<0.0001) | -0.13 (p<0.0001) | **159/160** |
| Llama 3.1 70B | 0.58 | baseline | -0.02 | - |
| Cognitive Models | 0.56 | - | baseline | 1/160 |

**통계**:
- Centaur vs Llama: t(1,985,732) = -144.22, p ≤ 0.0001, Cohen's d = 0.20
- Centaur vs Cognitive Models: t(1,985,732) = -127.58, p ≤ 0.0001, Cohen's d = 0.18

### 4.2 Out-of-Distribution Performance

| 실험 유형 | 실험명 | Centaur NLL | Llama NLL | 개선폭 |
|----------|--------|-------------|-----------|--------|
| **Modified Cover Story** | Two-step (magic carpet) | **0.51** | 0.63 | 19% |
| **Modified Structure** | Maggie's farm (3-armed) | **0.42** | 0.62 | 32% |
| **New Domain** | Logical reasoning | **1.65** | 1.92 | 14% |
| **New Domain** | Moral decision-making | **0.60** | 0.91 | 34% |
| **New Domain** | Economic games | **0.54** | 0.80 | 33% |
| **New Domain** | Naturalistic category learning | **0.47** | 0.57 | 18% |

### 4.3 Response Time Prediction

| 모델 | Conditional R² | Log Bayes Factor vs Centaur |
|------|----------------|----------------------------|
| **Centaur** | **0.87** | baseline |
| Llama | 0.75 | 53,773.5 |
| Cognitive Models | 0.77 | 14,995.5 |

---

## 5. Ko-CENTaUR 적용 가능한 Downstream Tasks

### 5.1 우선순위 높음 (즉시 구현 가능)

| Task | 데이터 가용성 | 구현 난이도 | Ko-CENTaUR 적합성 |
|------|--------------|------------|------------------|
| **Choices13k** | ✅ 이미 있음 | 낮음 | ⭐⭐⭐⭐⭐ |
| **Two-step Task** | ✅ Open datasets | 중간 | ⭐⭐⭐⭐⭐ |
| **Horizon Task** | ✅ Open datasets | 중간 | ⭐⭐⭐⭐⭐ |
| **N-back** | ✅ 쉽게 생성 가능 | 낮음 | ⭐⭐⭐⭐ |
| **Intertemporal Choice** | ✅ Open datasets | 낮음 | ⭐⭐⭐⭐ |
| **Weather Prediction** | ✅ Open datasets | 중간 | ⭐⭐⭐⭐ |

### 5.2 중간 우선순위 (한국어 적응 필요)

| Task | 데이터 가용성 | 구현 난이도 | Ko-CENTaUR 적합성 |
|------|--------------|------------|------------------|
| **Shepard Categorization** | ✅ Open | 중간 | ⭐⭐⭐ |
| **Iowa Gambling Task** | ✅ Open | 중간 | ⭐⭐⭐ |
| **Balloon Analogue Risk Task** | ✅ Open | 낮음 | ⭐⭐⭐ |
| **Multi-attribute Decision-making** | ❌ 생성 필요 | 중간 | ⭐⭐⭐⭐ |
| **Economic Games** | ✅ Open | 높음 | ⭐⭐⭐ |

### 5.3 장기 목표 (새로운 한국 데이터 필요)

| Task | 이유 | 예상 효과 |
|------|------|----------|
| **Korean Moral Decision-Making** | 문화적 차이 검증 | 한국 특화 인지 모델 |
| **Korean Intertemporal Choice** | 시간할인율 문화차 | 경제 행동 예측 |
| **Korean Category Learning** | 한국어 자연물 범주화 | 언어-인지 관계 |

---

## 6. 구현 로드맵

### Phase 1: 기본 평가 (1-2주)

**완료된 작업**:
- ✅ Choices13k 100-fold CV
- ✅ Choices13k test set evaluation

**진행 중**:
- 🔄 Qwen2.5-32B evaluation (Job 63129)
- 🔄 DeepSeek-R1-32B evaluation (Job 63130)

**다음 단계**:
1. Two-step task evaluation
2. Horizon task evaluation
3. 성능 비교 분석 문서 작성

### Phase 2: 확장 평가 (2-4주)

1. **Memory Tasks**:
   - N-back task 구현 및 평가
   - Digit span 평가

2. **Categorization Tasks**:
   - Shepard categorization
   - Weather prediction task

3. **Response Time Analysis**:
   - Hick's law 검증
   - Response entropy 분석

### Phase 3: Out-of-Distribution 평가 (4-8주)

1. **Cover Story Modification**:
   - Two-step task 한국어 cover story 생성
   - 성능 일반화 검증

2. **Structural Modification**:
   - 3-armed bandit 구현
   - Multi-armed bandit 확장

3. **New Domains**:
   - Korean moral decision-making
   - Korean economic games

---

## 7. 데이터 포맷 및 Prompt 구조

### 7.1 Natural Language Prompt 예시

**Multi-armed Bandit (Horizon Task)**:
```
In this task, you have to repeatedly choose between two slot machines
labelled B and C. When you select one of the machines, you will win or
lose points. Your goal is to choose the slot machines that will give you
the most points.

You press <<C>> and get -8 points.
You press <<B>> and get 0 points.
You press <<B>> and get 1 points.
```

**Decision-making (Risky Choice)**:
```
You will choose from two monetary lotteries by pressing N or U. Your
choice will trigger a random draw from the chosen lottery that will be
added to your bonus.

Lottery N offers 4.0 points with 80.0% or 0.0 points with 20.0%.
Lottery U offers 3.0 points with 100.0%.

You press <<U>>.
```

**Memory (N-back)**:
```
You will view a stream of letters on the screen, one letter at a time.
You have to remember the last two letters you saw since the beginning
of the block. If the letter you see matches the letter two trials ago,
press E, otherwise press K.

You see the letter V and press <<K>>.
You see the letter X and press <<K>>.
You see the letter V and press <<E>>.
```

### 7.2 Prompt 설계 원칙

1. **자연어 중심**: 실험 지시사항을 자연어로 변환
2. **Trial-by-trial History**: 전체 세션의 trial 이력 포함
3. **간결성**: 불필요한 정보 제거, 단순화
4. **최대 길이**: ~32,768 tokens

---

## 8. 기술적 세부사항

### 8.1 학습 설정

| 파라미터 | 값 |
|---------|-----|
| **Base Model** | Llama 3.1 70B |
| **Fine-tuning Method** | QLoRA (4-bit NF4) |
| **LoRA Rank** | 8 |
| **LoRA Alpha** | 16 |
| **Target Modules** | All linear layers (attention + FFN) |
| **Trainable Parameters** | 0.15% of base |
| **Epochs** | 1 |
| **Batch Size** | 32 (effective) |
| **Learning Rate** | 0.00005 |
| **Optimizer** | 8-bit AdamW |
| **Training Time** | ~5 days (A100 80GB) |

### 8.2 Loss Masking

- 인간 응답 토큰에만 loss 계산
- 지시사항 토큰은 loss masking
- 이를 통해 모델이 인간 행동 포착에 집중

### 8.3 Ko-CENTaUR 학습 현황

| 모델 | 학습 상태 | 파라미터 | 학습 시간 |
|------|----------|---------|----------|
| Qwen2.5-32B QLoRA | ✅ 완료 | 32B | 18.3h |
| DeepSeek-R1-Qwen-32B QLoRA | ✅ 완료 | 32B | 35.7h |
| EXAONE-3.0-7.8B QLoRA | ✅ 완료 | 7.8B | ~8h |

---

## 9. 주요 발견 및 시사점

### 9.1 CENTaUR의 강점

1. **Domain-General Capability**:
   - 160개 실험 중 159개에서 domain-specific models 능가

2. **Out-of-Distribution Generalization**:
   - Cover story 변경에 강건
   - 구조적 수정 (2-armed → 3-armed) 처리 가능
   - 완전히 새로운 도메인에서도 우수한 성능

3. **Human-like Behavior**:
   - Open-loop simulation에서 인간과 유사한 분포 생성
   - Exploration strategies (directed, random) 재현
   - Model-based/model-free mixture 생성

4. **Context Sensitivity**:
   - Noise ceiling 초과 → context-dependent patterns 포착
   - Response time prediction (R² = 0.87)

### 9.2 Ko-CENTaUR에 적용 시 고려사항

1. **한국어 데이터 필요성**:
   - Qwen2.5, DeepSeek 모두 다국어 지원하나 한국어 fine-tuning 필수
   - 문화적 차이로 인한 의사결정 패턴 차이 가능

2. **평가 데이터셋 구축**:
   - 최소 Choices13k, Two-step, Horizon task는 필수
   - 한국 참가자 데이터로 noise ceiling 검증 필요

3. **모델 크기와 성능**:
   - Centaur (70B) vs Minitaur (8B) 성능 차이 큼
   - Ko-CENTaUR는 32B → 적절한 중간 지점

4. **학습 데이터 확보**:
   - Psych-101 규모 (10M choices)는 현실적으로 어려움
   - 주요 도메인 (decision-making, bandit) 집중 전략 필요

---

## 10. 참고 자료

### 10.1 논문 정보
- **Title**: A foundation model to predict and capture human cognition
- **Authors**: Marcel Binz, Elif Akata, et al. (40+ authors)
- **Journal**: Nature, Vol 644, August 28, 2025
- **DOI**: 10.1038/s41586-025-09215-4

### 10.2 데이터 및 코드
- **Psych-101 Dataset**: https://huggingface.co/datasets/marcelbinz/Psych-101
- **Centaur Model**: https://huggingface.co/marcelbinz/Llama-3.1-Centaur-70B-adapter
- **Code**: https://github.com/marcelbinz/Llama-3.1-Centaur-70B

### 10.3 관련 연구
- **Tutorial Paper**: https://osf.io/preprints/psyarxiv/f7stn
- **Simplified Implementation**: https://github.com/Zak-Hussain/LLM4BeSci
- **CogBench**: Coda-Forno et al. (2024) - LLM cognitive benchmark

---

## Appendix A: 전체 실험 목록 (Extended Data Table 1)

### Decision-Making Tasks

| Experiment | Centaur NLL | Llama NLL | Cognitive NLL |
|------------|-------------|-----------|---------------|
| Shepard categorization | 0.5394 | 0.5818 | 0.6108 |
| Drifting four-armed bandit | 0.7029 | 0.8810 | 0.9043 |
| N-back | 0.3954 | 0.5209 | 0.5787 |
| Digit span | 0.5520 | 0.6618 | 0.9359 |
| Go/no-go | 0.0000 | 0.0062 | 0.0757 |
| Recent probes | 0.2572 | 0.3433 | 0.3868 |
| Horizon task | 0.4032 | 0.5237 | 0.3595 |
| Gardening task | 0.3783 | 0.5040 | 0.9105 |
| Columbia card task | 0.1867 | 0.2261 | 0.2629 |
| Balloon analog risk task | 0.0593 | 0.0753 | 0.0922 |
| Two-armed bandit | 0.2963 | 0.3829 | 0.4187 |
| Two-step task | 0.4998 | 0.6075 | 0.6043 |
| Conditional associative learning | 0.5380 | 0.6373 | 0.8575 |
| THINGS odd-one-out | 0.8068 | 1.1386 | 0.8253 |
| Multi-attribute decision-making | 0.0619 | 0.1502 | 0.1922 |
| Probabilistic instrumental learning | 0.4937 | 0.5382 | 0.5047 |
| Medin categorization | 0.4967 | 0.5772 | 0.5313 |
| Zoopermarket | 0.4850 | 0.6026 | 0.6047 |
| choices13k | 0.4274 | 0.5342 | 0.6563 |
| CPC18 | 0.3390 | 0.4118 | 0.6607 |
| Intertemporal choice | 0.4340 | 0.7336 | 0.6591 |
| Structured bandit | 0.6410 | 0.8114 | 1.0530 |
| Weather prediction task | 0.5514 | 0.5749 | 0.6267 |
| Iowa gambling task | 0.8890 | 0.9880 | 1.1555 |
| Virtual subway network | 1.1271 | 1.5347 | nan |
| Multi-task reinforcement learning | 0.5672 | 0.6604 | 1.0424 |
| Serial reaction time task | 0.1718 | 0.1900 | 0.1962 |
| Decisions from description | 0.5336 | 0.7569 | 0.6120 |
| Decisions from experience | 0.3686 | 0.4339 | 0.5404 |
| Changing bandit | 0.3025 | 0.3824 | 0.4378 |
| Multiple-cue judgment | 1.1236 | 1.2818 | 1.9157 |
| Recall and recognition | 1.0591 | 1.3759 | nan |
| Experiential-symbolic task | 0.4536 | 0.6983 | nan |
| Grammar judgement | 1.4355 | 1.9949 | 1.4127 |
| Risky choice | 0.4281 | 0.6475 | nan |
| Tile-revealing task | 1.8713 | 2.7380 | nan |
| Episodic long-term memory | 0.8684 | 1.1344 | nan |
| Aversive learning | 4.0733 | 5.1066 | nan |
| Spatially correlated multi-armed bandit | 1.8319 | 2.4479 | 2.7635 |
| Probabilistic reasoning | 2.3731 | 2.6406 | nan |

---

**문서 작성**: Claude Code (Sonnet 4.5)
**검토 필요**: Ko-CENTaUR 프로젝트 팀
**다음 업데이트**: 평가 결과 수집 후 (Qwen2.5-32B, DeepSeek-R1-32B)
