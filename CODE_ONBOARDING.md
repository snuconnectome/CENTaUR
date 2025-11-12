# CENTaUR 코드 온보딩 가이드

이 문서는 CENTaUR 프로젝트의 **코드 구조와 구현 세부사항**을 이해하기 위한 가이드입니다.

## 📋 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [코드 아키텍처](#코드-아키텍처)
3. [핵심 모듈 설명](#핵심-모듈-설명)
4. [데이터 흐름](#데이터-흐름)
5. [실행 파이프라인](#실행-파이프라인)
6. [주요 파일 상세](#주요-파일-상세)

---

## 프로젝트 개요

### CENTaUR란?

**CENTaUR** (Cognitive Embeddings for Natural Understanding & Representation)는 대규모 언어 모델(LLM)의 hidden state를 인지 표현(cognitive representation)으로 사용하여 인간의 의사결정을 예측하는 연구 프로젝트입니다.

### 핵심 아이디어

```
LLM 입력 (선택 과제 텍스트)
    ↓
Hidden State 추출 (마지막 레이어)
    ↓
Binomial Regression 학습
    ↓
인간 선택 예측 (NLL 메트릭)
```

### 평가 기준

- **Random Baseline**: ~120K NLL
- **LLaMA-65B (원본)**: ~30K NLL
- **목표**: 30K 이하 달성

---

## 코드 아키텍처

### 전체 구조

```
CENTaUR/
├── ko_centaur/              # 메인 모듈 (현대적 구현)
│   ├── baselines/          # 모델 로딩 및 관리
│   ├── evaluation/         # 평가 파이프라인
│   │   ├── extract_features.py    # Feature 추출
│   │   ├── cross_validation.py    # LOO CV 구현
│   │   └── metrics.py             # NLL 계산
│   ├── training/           # Fine-tuning 코드
│   ├── data/               # 데이터 처리
│   └── scripts/            # 실행 스크립트
│
├── scripts/                # 루트 레벨 스크립트
│   ├── extract_centaur_features.py  # Feature extraction 실행
│   └── fit_centaur_loo_cv.py        # LOO CV 실행
│
├── legacy/                 # 원본 CENTaUR 코드 (참고용)
│   └── models.py           # BinomialRegression 클래스
│
└── models.py               # BinomialRegression (현재 사용)
```

### 모듈 간 의존성

```
run_full_eval.py (또는 extract_centaur_features.py)
    ↓
BaselineModelManager (baselines/load_baselines.py)
    ↓
extract_features.py (evaluation/)
    ↓
cross_validation.py (evaluation/)
    ↓
BinomialRegression (models.py)
    ↓
metrics.py (evaluation/)
```

---

## 핵심 모듈 설명

### 1. BaselineModelManager (`ko_centaur/baselines/load_baselines.py`)

**역할**: 모든 모델을 통합 인터페이스로 로드 및 관리

**주요 기능**:
- 모델 로딩 (HuggingFace, QLoRA 어댑터 지원)
- Hidden state 추출
- 프롬프트 포맷팅 (모델별 템플릿)

**지원 모델**:
- `qwen25`: Qwen2.5-32B-Instruct (QLoRA fine-tuned)
- `deepseek`: DeepSeek-R1-Distill-Qwen-32B (QLoRA fine-tuned)
- `exaone35`: EXAONE-3.5-32B-Instruct
- `gpt-oss`: GPT-OSS-20B
- `kimi-k2`: Kimi K2

**핵심 메서드**:
```python
model_manager = BaselineModelManager(model_type="qwen25")
features = model_manager.extract_features(prompt)  # Shape: (1, hidden_size)
```

### 2. Feature Extraction (`ko_centaur/evaluation/extract_features.py`)

**역할**: 데이터셋에서 hidden state 추출

**주요 함수**:
- `extract_features_for_sample()`: 단일 샘플 처리
- `extract_features_batch()`: 배치 처리
- `save_features()`: `.pth` 파일로 저장
- `load_features()`: 저장된 feature 로드

**데이터 형식 지원**:
- **choices13k**: `{'text': str, 'choice': int}`
- **Psych-101**: `{'task_description': str, 'label': int, 'system_prompt': str}`

### 3. Cross-Validation (`ko_centaur/evaluation/cross_validation.py`)

**역할**: 100-fold LOO CV 및 nested CV 구현

**주요 함수**:
- `generate_loo_splits()`: LOO 분할 생성
- `check_data_leakage()`: 데이터 누수 검사
- `grid_search()`: 하이퍼파라미터 그리드 서치
- `nested_cv()`: Nested cross-validation

**하이퍼파라미터**:
- `alpha`: L2 정규화 계수 (그리드: `[0.01, 0.1, 1.0, 10.0, 100.0]`)
- Inner CV: 5-fold로 최적 alpha 선택
- Outer CV: 선택된 alpha로 최종 평가

### 4. BinomialRegression (`models.py`)

**역할**: Hidden state → 선택 확률 예측

**구조**:
```python
class BinomialRegression(nn.Module):
    def __init__(self, num_inputs, alpha=0):
        self.W = nn.Linear(num_inputs, 1, bias=False)  # 단일 레이어
    
    def forward(self, X):
        return self.W(X).squeeze(-1)  # Logits 반환
```

**학습**:
- Optimizer: LBFGS
- Loss: Negative Binomial Log-Likelihood + L2 regularization
- Early stopping: NaN/Inf 감지 시 중단

### 5. Metrics (`ko_centaur/evaluation/metrics.py`)

**역할**: 평가 메트릭 계산

**주요 메트릭**:
- **NLL** (Negative Log-Likelihood): 주요 평가 지표
- **Accuracy**: 예측 정확도
- **AUC**: ROC 곡선 아래 면적

---

## 데이터 흐름

### 전체 파이프라인

```
1. 데이터 로드
   └─> JSONL 파일 읽기 (choices13k_1000.jsonl)
   
2. Feature Extraction (GPU 필요)
   └─> 각 샘플에 대해:
       ├─> 텍스트 → 프롬프트 포맷팅
       ├─> 모델 forward pass
       └─> 마지막 레이어 hidden state 추출
   └─> outputs/{model}_features.pth 저장
   
3. Cross-Validation (CPU 가능)
   └─> Feature 로드
   └─> 100-fold LOO:
       ├─> Train/Test 분할
       ├─> Inner CV (5-fold)로 alpha 선택
       ├─> BinomialRegression 학습
       └─> Test NLL 계산
   └─> 평균 NLL 반환
```

### 데이터 형식

#### 입력 데이터 (choices13k)

```json
{
  "text": "Option A: 50% chance of $100\nOption B: $45 for sure",
  "choice": 0  // 0=A, 1=B
}
```

#### 추출된 Feature

```python
{
    'features': torch.Tensor,  # Shape: (n_samples, hidden_size)
    'metadata': {
        'model': 'qwen25',
        'n_samples': 1000,
        'hidden_size': 3584,
        'timestamp': '2025-01-XX'
    }
}
```

#### 평가 결과

```python
{
    'mean_nll': 28500.0,
    'std_nll': 1200.0,
    'mean_accuracy': 0.72,
    'per_fold_results': [...]
}
```

---

## 실행 파이프라인

### 1. Feature Extraction

**스크립트**: `scripts/extract_centaur_features.py`

**실행 방법**:
```bash
# 로컬 테스트 (10 샘플)
python scripts/extract_centaur_features.py \
    --model qwen25 \
    --n_samples 10 \
    --use_quantization

# 전체 실행
python scripts/extract_centaur_features.py \
    --model qwen25 \
    --use_quantization
```

**내부 동작**:
1. 데이터셋 로드 (`ko_centaur/data/choices13k_1000.jsonl`)
2. BaselineModelManager 초기화
3. 각 샘플에 대해 `extract_features_for_sample()` 호출
4. 배치로 처리하여 GPU 효율성 향상
5. `outputs/qwen25_features.pth` 저장

### 2. LOO Cross-Validation

**스크립트**: `scripts/fit_centaur_loo_cv.py`

**실행 방법**:
```bash
# 로컬 테스트 (5-fold)
python scripts/fit_centaur_loo_cv.py \
    --model qwen25 \
    --n_folds 5

# 전체 실행 (100-fold)
python scripts/fit_centaur_loo_cv.py \
    --model qwen25
```

**내부 동작**:
1. Feature 파일 로드 (`outputs/qwen25_features.pth`)
2. `generate_loo_splits()`로 100개 분할 생성
3. 각 fold에 대해:
   - Train/Test 분할
   - `nested_cv()` 실행:
     - Inner CV로 최적 alpha 선택
     - 선택된 alpha로 BinomialRegression 학습
   - Test NLL 계산
4. 평균 NLL 및 통계 반환

### 3. 통합 실행

**스크립트**: `ko_centaur/scripts/run_full_eval.py`

**실행 방법**:
```bash
python ko_centaur/scripts/run_full_eval.py \
    --dataset choices13k \
    --model qwen25 \
    --n_folds 100
```

**특징**:
- Feature extraction과 CV를 한 번에 실행
- SLURM 호환 (작업 제출 가능)
- 진행 상황 로깅

---

## 주요 파일 상세

### 실행 스크립트

#### `scripts/extract_centaur_features.py`
- **목적**: Feature extraction 실행
- **입력**: 모델명, 샘플 수
- **출력**: `outputs/{model}_features.pth`
- **핵심 로직**: `ko_centaur.evaluation.extract_features` 모듈 사용

#### `scripts/fit_centaur_loo_cv.py`
- **목적**: LOO CV 실행
- **입력**: 모델명, fold 수
- **출력**: NLL 값 및 평가 결과
- **핵심 로직**: `ko_centaur.evaluation.cross_validation` 모듈 사용

#### `ko_centaur/scripts/run_full_eval.py`
- **목적**: 전체 평가 파이프라인 실행
- **특징**: Feature extraction + CV 통합
- **SLURM 지원**: `--slurm` 플래그로 작업 제출

### 핵심 모듈

#### `ko_centaur/baselines/load_baselines.py`
- **클래스**: `BaselineModelManager`
- **기능**: 
  - 모델 로딩 (HuggingFace, QLoRA)
  - Hidden state 추출
  - 프롬프트 포맷팅
- **양자화 지원**: NF4, 8-bit

#### `ko_centaur/evaluation/extract_features.py`
- **함수**:
  - `extract_features_for_sample()`: 단일 샘플
  - `extract_features_batch()`: 배치 처리
  - `save_features()` / `load_features()`: 저장/로드

#### `ko_centaur/evaluation/cross_validation.py`
- **함수**:
  - `generate_loo_splits()`: LOO 분할
  - `nested_cv()`: Nested CV
  - `grid_search()`: 하이퍼파라미터 탐색

#### `models.py`
- **클래스**: `BinomialRegression`
- **기능**: Hidden state → 선택 확률 예측
- **학습**: LBFGS optimizer, Binomial loss

### 설정 파일

#### `ko_centaur/configs/training_*.yaml`
- Fine-tuning 설정 (QLoRA 파라미터)
- 모델별 설정 파일:
  - `training_qwen25_32b_qlora.yaml`
  - `training_deepseek_r1_qwen32b_qlora.yaml`
  - `training_exaone40.yaml`

### 데이터 파일

#### `ko_centaur/data/choices13k_*.jsonl`
- `choices13k_100.jsonl`: 테스트용 (100 샘플)
- `choices13k_1000.jsonl`: 전체 데이터셋
- 형식: `{"text": str, "choice": int}`

---

## 코드 읽기 가이드

### 시작점 추천 순서

1. **`models.py`** (가장 간단)
   - BinomialRegression 클래스 이해
   - Loss 함수 및 학습 로직

2. **`ko_centaur/baselines/load_baselines.py`**
   - 모델 로딩 방법
   - Hidden state 추출 메커니즘

3. **`ko_centaur/evaluation/extract_features.py`**
   - Feature extraction 파이프라인
   - 데이터 형식 처리

4. **`ko_centaur/evaluation/cross_validation.py`**
   - CV 구현 세부사항
   - Nested CV 로직

5. **`scripts/extract_centaur_features.py`**
   - 전체 실행 흐름
   - 명령줄 인터페이스

### 디버깅 팁

#### Feature Extraction 디버깅
```python
# 단일 샘플 테스트
from ko_centaur.baselines import BaselineModelManager
from ko_centaur.evaluation.extract_features import extract_features_for_sample

model_manager = BaselineModelManager("qwen25")
sample = {"text": "Option A: $50\nOption B: $40", "choice": 0}
features = extract_features_for_sample(model_manager, sample)
print(features.shape)  # (1, 3584)
```

#### CV 디버깅
```python
# 작은 fold로 테스트
from ko_centaur.evaluation.cross_validation import generate_loo_splits

splits = generate_loo_splits(10)  # 10-fold
print(len(splits))  # 10
```

---

## 주요 개념 정리

### Hidden State
- LLM의 마지막 레이어 출력 벡터
- Shape: `(batch_size, sequence_length, hidden_size)`
- CENTaUR에서는 마지막 토큰의 hidden state 사용

### Binomial Regression
- 입력: Hidden state (벡터)
- 출력: Logits (스칼라)
- 확률: `sigmoid(logits)` → 선택 확률

### LOO CV (Leave-One-Out)
- N개 샘플 → N개 fold
- 각 fold: 1개 테스트, N-1개 학습
- 일반화 성능 평가에 적합

### Nested CV
- Outer CV: 최종 성능 평가
- Inner CV: 하이퍼파라미터 선택
- 데이터 누수 방지

### NLL (Negative Log-Likelihood)
- 낮을수록 좋음
- Random: ~120K
- LLaMA-65B: ~30K
- 목표: 30K 이하

---

## 다음 단계

1. ✅ 코드 구조 이해 완료
2. 🔄 로컬에서 테스트 실행
3. 🔄 주요 함수 단위 테스트 작성
4. 🔄 새로운 모델 추가 방법 학습
5. 🔄 성능 최적화 기법 탐색

---

## 참고 자료

- **[ONBOARDING.md](ONBOARDING.md)**: 프로젝트 전체 온보딩
- **[README.md](README.md)**: 프로젝트 개요
- **[CLAUDE.md](CLAUDE.md)**: 상세 문서
- **원본 논문**: Binz & Schulz (2023) ICLR

---

**마지막 업데이트**: 2025-01-XX

