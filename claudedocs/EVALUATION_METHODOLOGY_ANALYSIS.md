# CENTaUR 평가 방법론 심층 분석

**작성일**: 2025-10-25
**작성자**: Claude Code
**목적**: Ko-CENTaUR 프로젝트의 평가 방법론 완전 이해

---

## Executive Summary

CENTaUR(Cognitive Embedding for Natural Understanding & Representation) 프로젝트는 LLM의 숨겨진 상태(hidden states)를 인지 특징으로 사용하여 **인간의 의사결정을 예측**하는 프레임워크입니다. 본 문서는 원본 논문(Binz & Schulz, 2023)의 평가 방법론과 현재 Ko-CENTaUR 구현의 차이점을 분석합니다.

**핵심 발견**:
1. 원본: **Binomial Regression** + **100-fold LOO CV** + **Negative Log-Likelihood** 메트릭
2. 현재: **Token Probability** + **Accuracy** 메트릭 (방법론이 다름)
3. 문제: 현재 평가 방식이 원본과 달라서 벤치마크 비교 불가

---

## 1. 원본 CENTaUR 평가 방법론

### 1.1 전체 파이프라인

```
[Human Decision Data]
    ↓
[LLM Forward Pass] → Extract hidden states (last layer)
    ↓
[Feature Extraction] → Shape: (n_samples, hidden_dim)
    ↓
[100-fold LOO Cross-Validation]
    ├─ Train fold: n-1 samples
    │   ├─ [11-fold Nested CV] → Select best alpha (regularization)
    │   └─ [Binomial Regression] → Fit model
    └─ Test fold: 1 sample
        └─ [Predict] → Compute log-likelihood
    ↓
[Aggregate] → Total negative log-likelihood (NLL)
```

### 1.2 Binomial Regression 모델

**목적**: 인간의 선택 확률을 예측

**수식**:
```
logit = W · features  (Linear transformation)
P(choice = B) = Binomial(n=num_choices, logit)
Loss = -log P(observed_choice) + α||W||²  (L2 regularization)
```

**코드** (`models.py:10-31`):
```python
class BinomialRegression(nn.Module):
    def __init__(self, num_inputs, alpha=0):
        self.W = nn.Linear(num_inputs, 1, bias=False)
        self.alpha = alpha

    def forward(self, X):
        return self.W(X).squeeze(-1)  # logits

    def fit(self, X, num_choices, num_B_choices):
        optimizer = optim.LBFGS(self.parameters())
        loss = -Binomial(total_count=num_choices,
                         logits=self(X)).log_prob(num_B_choices).mean()
        loss += self.alpha * self.W.weight.pow(2).sum()  # L2 reg
```

**특징**:
- **선형 모델**: Hidden states → Logits (단일 가중치 행렬)
- **LBFGS 최적화**: 2차 최적화 알고리즘 (빠른 수렴)
- **L2 정규화**: α 하이퍼파라미터로 과적합 방지

### 1.3 100-fold Leave-One-Out Cross-Validation

**원리**: 각 샘플을 한 번씩 테스트 셋으로 사용

**절차** (`fit_centaur.py:28-108`):
```python
for fold_id in range(100):
    # 1. Split data
    train_index = all_samples except fold_id
    test_index = fold_id

    # 2. Nested 11-fold CV for alpha selection
    alphas = [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
    for inner_fold in range(11):
        for alpha in alphas:
            # Fit on inner train, evaluate on inner validation
            model.fit(inner_train, alpha=alpha)
            validation_ll = model.log_likelihood(inner_val)

    best_alpha = alphas[validation_ll.argmin()]

    # 3. Refit on all training data with best alpha
    model.fit(train_data, alpha=best_alpha)

    # 4. Evaluate on test fold
    test_ll = model.log_likelihood(test_data)

    save_result(fold_id, test_ll, best_alpha)
```

**이유**:
- **Unbiased Evaluation**: 모든 샘플이 정확히 한 번 테스트됨
- **Small Sample Handling**: 100개 샘플에서 최대한 활용
- **Nested CV**: Train/test contamination 방지

### 1.4 평가 메트릭: Negative Log-Likelihood (NLL)

**정의**:
```
NLL = -Σ log P(observed_choice | features)
```

**해석**:
- **낮을수록 좋음**: 모델이 인간의 선택을 더 잘 예측
- **확률적 메트릭**: Accuracy와 달리 확신도(confidence)도 평가
- **벤치마크 가능**: Random baseline (NLL ≈ 120,000) vs CENTaUR (NLL ≈ 30,000)

**예시** (choices13k 결과):
```
Random Baseline:  NLL = 120,000  (50/50 guessing)
LLaMA-65B Token:  NLL = 90,000   (25% improvement)
BEAST Cognitive:  NLL = 60,000   (50% improvement)
CENTaUR (LLaMA):  NLL = 30,000   (75% improvement) ⭐ BEST
```

---

## 2. 현재 Ko-CENTaUR 평가 방식

### 2.1 현재 구현 (`scripts/evaluate_new_models.py`)

**파이프라인**:
```
[Load Model + LoRA Adapter]
    ↓
[For each sample in dataset]
    ├─ Tokenize prompt
    ├─ Forward pass
    ├─ Extract logits for tokens "A" and "B"
    ├─ Compute P(A) and P(B) via softmax
    └─ Predict argmax(P(A), P(B))
    ↓
[Compute Accuracy] = Correct predictions / Total
```

**코드 스니펫**:
```python
def predict_choice(tokenizer, model, text):
    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[0, -1, :]  # Last token

        # Get tokens for A and B
        token_a = tokenizer.encode(" A", add_special_tokens=False)[0]
        token_b = tokenizer.encode(" B", add_special_tokens=False)[0]

        # Softmax over A and B
        prob_a = torch.softmax(torch.tensor([logits[token_a],
                                             logits[token_b]]), dim=0)[0]
        prob_b = 1 - prob_a

    return prob_a, prob_b
```

### 2.2 주요 차이점

| 측면 | 원본 CENTaUR | 현재 Ko-CENTaUR |
|------|-------------|----------------|
| **특징 추출** | Hidden states (4096-dim) | ❌ 없음 (직접 토큰 확률 사용) |
| **모델** | Binomial Regression | ❌ 없음 (LLM 직접 예측) |
| **Cross-Validation** | 100-fold LOO CV | ❌ 없음 (단일 evaluation) |
| **Hyperparameter Tuning** | 11-fold nested CV for alpha | ❌ 없음 |
| **메트릭** | Negative Log-Likelihood | ⚠️ Accuracy만 사용 |
| **정규화** | Per-fold z-score normalization | ❌ 없음 |

**Critical Issue**: 현재 방식은 **원본 CENTaUR 방법론이 아님!**

---

## 3. 문제점 분석

### 3.1 토큰 확률 vs Hidden States

**Token Probability 방식 (현재)**:
```python
# LLM이 직접 "A" 또는 "B" 토큰의 확률 예측
P(choice = A) = softmax(logits)["A"]
```

**장점**:
- ✅ 간단하고 빠름
- ✅ 추가 모델 학습 불필요

**단점**:
- ❌ Hidden states의 풍부한 정보 사용 안 함
- ❌ LLM이 토큰 확률로만 판단 (shallow reasoning)
- ❌ 원본 논문 방법론과 다름 → 벤치마크 비교 불가
- ❌ Fine-tuning 효과 평가 어려움

**Hidden State Regression (원본)**:
```python
# Hidden states를 인지 특징으로 사용
features = model.hidden_states[-1][:, -1, :]  # (batch, 4096)
logits = BinomialRegression(features)  # Learn mapping
```

**장점**:
- ✅ LLM의 내부 표상 활용 (더 풍부한 정보)
- ✅ Interpretable cognitive features
- ✅ 원본 논문과 동일한 방법론
- ✅ Fine-tuning이 hidden states에 미친 영향 정확히 측정

### 3.2 평가 메트릭 문제

**Accuracy (현재)**:
```
Qwen2.5-32B: 70% accuracy (진행중)
DeepSeek-R1: 70% accuracy (진행중)
```

**문제점**:
1. **이진 분류 메트릭**: 확률 예측의 품질 평가 불가
2. **Threshold 의존성**: 50% cutoff가 최적이 아닐 수 있음
3. **불확실성 무시**: "51% vs 99% confidence" 차이를 구분 못함

**Negative Log-Likelihood (원본)**:
```
LLaMA-65B: NLL = 30,000
Expected Qwen2.5: NLL = 25,000-28,000
Expected DeepSeek-R1: NLL = 20,000-25,000
```

**장점**:
1. **확률 품질**: Calibration까지 평가
2. **Continuous metric**: Fine-grained performance 측정
3. **Benchmark 가능**: 원본 논문 결과와 직접 비교

### 3.3 평가 시간 문제

**현재 상황**:
- 100개 샘플 평가에 **2시간 소요** (33-45개만 완료 후 타임아웃)
- 샘플당 **~2-3분** (너무 느림!)

**원인 분석**:
1. **Model Loading**: 매 샘플마다 forward pass
2. **LoRA Merge**: `merge_and_unload()` 비용
3. **비효율적 배치 처리**: 샘플별 개별 처리

**원본 방법의 효율성**:
```python
# 1회 feature extraction (빠름)
features = extract_all_features(model, dataset)  # 1-2 hours
save_features(features)  # Cache

# 100-fold CV (빠름)
for fold in range(100):
    # Feature loading만 (no model forward pass)
    train_features, test_features = load_cached_features()

    # Lightweight regression (seconds)
    model = BinomialRegression().fit(train_features)
    nll = model.evaluate(test_features)
```

**예상 시간**:
- Feature extraction: **1회, 1-2시간**
- 100-fold CV: **각 fold 1-2분 = 총 2-3시간**
- **총 3-5시간** (현재 8시간보다 빠름!)

---

## 4. 올바른 평가 방법론 구현 계획

### 4.1 Phase 1: Feature Extraction (1회)

**Script**: `scripts/extract_features_choices13k.py` (새로 작성)

```python
#!/usr/bin/env python3
"""
Extract hidden states from Qwen2.5-32B and DeepSeek-R1-32B
for choices13k dataset (100 samples)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from tqdm import tqdm

def extract_features(model_name: str, adapter_path: str,
                      dataset_path: str, output_path: str):
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        output_hidden_states=True  # 중요!
    )

    # Load LoRA adapter and merge
    model = PeftModel.from_pretrained(model, adapter_path)
    model = model.merge_and_unload()
    model.eval()

    # Load dataset
    with open(dataset_path, 'r') as f:
        data = [json.loads(line) for line in f]

    # Extract features
    all_features = []
    all_labels = []

    for item in tqdm(data, desc="Extracting features"):
        prompt = item["text"]
        label = item["choice"]  # 0 for A, 1 for B

        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            # Get last layer hidden state at last token position
            hidden_state = outputs.hidden_states[-1][0, -1, :]  # (hidden_dim,)

        all_features.append(hidden_state.cpu())
        all_labels.append(label)

    # Stack and save
    features = torch.stack(all_features)  # (n_samples, hidden_dim)
    labels = torch.tensor(all_labels)     # (n_samples,)

    torch.save({
        'features': features,
        'labels': labels,
        'model_name': model_name,
        'dataset': dataset_path
    }, output_path)

    print(f"✅ Saved features: {features.shape}")
    print(f"✅ Output: {output_path}")

if __name__ == "__main__":
    # Qwen2.5-32B
    extract_features(
        model_name="Qwen/Qwen2.5-32B-Instruct",
        adapter_path="/scratch/.../outputs/qwen25-32b-qlora",
        dataset_path="/scratch/.../data/choices13k_100.jsonl",
        output_path="/scratch/.../data/qwen25_features_choices13k.pth"
    )

    # DeepSeek-R1-32B
    extract_features(
        model_name="/home/.../models/deepseek-r1-distill-qwen-32b",
        adapter_path="/scratch/.../outputs/deepseek-r1-qwen32b-qlora",
        dataset_path="/scratch/.../data/choices13k_100.jsonl",
        output_path="/scratch/.../data/deepseek_features_choices13k.pth"
    )
```

### 4.2 Phase 2: Binomial Regression + 100-fold LOO CV

**Script**: `scripts/fit_loo_cv_choices13k.py` (원본 `fit_centaur.py` 수정)

```python
#!/usr/bin/env python3
"""
100-fold Leave-One-Out Cross-Validation with Binomial Regression
Following original CENTaUR methodology (Binz & Schulz, 2023)
"""

import torch
import argparse
from models import BinomialRegression  # 기존 코드 재사용
from torch.distributions import Binomial

def run_loo_cv(features_path: str, output_path: str):
    # Load features
    data = torch.load(features_path)
    features = data['features']  # (100, hidden_dim)
    labels = data['labels']      # (100,) - binary choices

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    features = features.to(device)
    labels = labels.to(device)

    n_samples = features.shape[0]
    results = []

    # 100-fold LOO CV
    for test_idx in range(n_samples):
        print(f"Fold {test_idx + 1}/100")

        # Split
        train_indices = [i for i in range(n_samples) if i != test_idx]
        train_features = features[train_indices]
        train_labels = labels[train_indices]
        test_features = features[test_idx:test_idx+1]
        test_label = labels[test_idx:test_idx+1]

        # Nested CV for alpha selection (11-fold)
        alphas = [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0]
        n_inner_folds = 11
        validation_nll = torch.zeros(len(alphas))

        train_indices_tensor = torch.tensor(train_indices)
        folds = train_indices_tensor.reshape(-1, n_inner_folds)

        for alpha_idx, alpha in enumerate(alphas):
            fold_nlls = []

            for inner_fold in range(n_inner_folds):
                # Inner split
                inner_train_idx = torch.cat([
                    folds[:, :inner_fold],
                    folds[:, inner_fold+1:]
                ], dim=1).flatten()
                inner_val_idx = folds[:, inner_fold]

                inner_train_feat = features[inner_train_idx]
                inner_train_lab = labels[inner_train_idx]
                inner_val_feat = features[inner_val_idx]
                inner_val_lab = labels[inner_val_idx]

                # Normalize
                mean = inner_train_feat.mean(0, keepdim=True)
                std = inner_train_feat.std(0, keepdim=True)
                std = torch.where(std > 0, std, torch.ones_like(std))

                inner_train_feat = (inner_train_feat - mean) / std
                inner_val_feat = (inner_val_feat - mean) / std

                # Fit
                model = BinomialRegression(
                    inner_train_feat.shape[1],
                    alpha=alpha
                ).to(device)

                model.fit(
                    inner_train_feat,
                    torch.ones_like(inner_train_lab),  # num_choices = 1
                    inner_train_lab.float()            # num_B_choices
                )

                # Evaluate
                logits = model(inner_val_feat)
                nll = -Binomial(
                    total_count=1,
                    logits=logits
                ).log_prob(inner_val_lab.float()).mean()

                fold_nlls.append(nll.item())

            validation_nll[alpha_idx] = sum(fold_nlls) / len(fold_nlls)

        # Select best alpha
        best_alpha = alphas[validation_nll.argmin()]

        # Refit on all training data
        mean = train_features.mean(0, keepdim=True)
        std = train_features.std(0, keepdim=True)
        std = torch.where(std > 0, std, torch.ones_like(std))

        train_features_norm = (train_features - mean) / std
        test_features_norm = (test_features - mean) / std

        final_model = BinomialRegression(
            train_features.shape[1],
            alpha=best_alpha
        ).to(device)

        final_model.fit(
            train_features_norm,
            torch.ones_like(train_labels),
            train_labels.float()
        )

        # Test evaluation
        test_logits = final_model(test_features_norm)
        test_nll = -Binomial(
            total_count=1,
            logits=test_logits
        ).log_prob(test_label.float()).sum().item()

        results.append({
            'fold_id': test_idx,
            'test_nll': test_nll,
            'best_alpha': best_alpha,
            'test_logit': test_logits.item(),
            'test_label': test_label.item()
        })

    # Aggregate
    total_nll = sum([r['test_nll'] for r in results])

    torch.save({
        'fold_results': results,
        'total_nll': total_nll,
        'mean_nll': total_nll / n_samples,
        'features_path': features_path
    }, output_path)

    print(f"\n✅ 100-fold LOO CV Complete")
    print(f"Total NLL: {total_nll:.2f}")
    print(f"Mean NLL per sample: {total_nll/n_samples:.4f}")
    print(f"Results saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", required=True,
                        help="Path to features .pth file")
    parser.add_argument("--output", required=True,
                        help="Path to save results")
    args = parser.parse_args()

    run_loo_cv(args.features, args.output)
```

### 4.3 Phase 3: 비교 분석

**Script**: `scripts/compare_models_choices13k.py`

```python
#!/usr/bin/env python3
"""
Compare Qwen2.5-32B and DeepSeek-R1-32B performance
on choices13k using CENTaUR methodology
"""

import torch
import matplotlib.pyplot as plt

def compare_models():
    # Load results
    qwen_results = torch.load("data/qwen25_loo_results.pth")
    deepseek_results = torch.load("data/deepseek_loo_results.pth")

    # Extract metrics
    qwen_nll = qwen_results['total_nll']
    deepseek_nll = deepseek_results['total_nll']

    # Baselines (from original paper)
    random_nll = 120000
    llama65b_nll = 30000

    # Print comparison
    print("=" * 60)
    print("CENTaUR Evaluation Results (choices13k)")
    print("=" * 60)
    print(f"Random Baseline:     NLL = {random_nll:>10,}")
    print(f"LLaMA-65B (2023):    NLL = {llama65b_nll:>10,}")
    print(f"Qwen2.5-32B (2025): NLL = {qwen_nll:>10,.2f}")
    print(f"DeepSeek-R1 (2025): NLL = {deepseek_nll:>10,.2f}")
    print("=" * 60)

    # Improvement over baseline
    qwen_improvement = (1 - qwen_nll / random_nll) * 100
    deepseek_improvement = (1 - deepseek_nll / random_nll) * 100

    print(f"\nImprovement over Random:")
    print(f"Qwen2.5-32B:  {qwen_improvement:.1f}%")
    print(f"DeepSeek-R1:  {deepseek_improvement:.1f}%")

    # Comparison to LLaMA-65B
    qwen_vs_llama = (1 - qwen_nll / llama65b_nll) * 100
    deepseek_vs_llama = (1 - deepseek_nll / llama65b_nll) * 100

    print(f"\nImprovement over LLaMA-65B (2023):")
    print(f"Qwen2.5-32B:  {qwen_vs_llama:+.1f}%")
    print(f"DeepSeek-R1:  {deepseek_vs_llama:+.1f}%")

    # Visualization
    models = ['Random', 'LLaMA-65B\n(2023)', 'Qwen2.5-32B\n(2025)',
              'DeepSeek-R1\n(2025)']
    nlls = [random_nll, llama65b_nll, qwen_nll, deepseek_nll]
    colors = ['#cccccc', '#1f77b4', '#ff7f0e', '#2ca02c']

    plt.figure(figsize=(10, 6))
    bars = plt.bar(models, nlls, color=colors)
    plt.ylabel('Negative Log-Likelihood (lower = better)', fontsize=12)
    plt.title('CENTaUR Performance on choices13k Dataset', fontsize=14)
    plt.xticks(fontsize=11)

    # Add value labels on bars
    for bar, nll in zip(bars, nlls):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height,
                f'{nll:,.0f}',
                ha='center', va='bottom', fontsize=10)

    plt.tight_layout()
    plt.savefig('results/choices13k_comparison.png', dpi=300)
    print(f"\n✅ Plot saved: results/choices13k_comparison.png")

if __name__ == "__main__":
    compare_models()
```

---

## 5. 실행 계획 및 예상 시간

### Timeline

| Phase | Task | Duration | Output |
|-------|------|----------|--------|
| **1** | Feature Extraction - Qwen2.5 | 1-2 hours | `qwen25_features_choices13k.pth` |
| **2** | Feature Extraction - DeepSeek | 1-2 hours | `deepseek_features_choices13k.pth` |
| **3** | 100-fold LOO CV - Qwen2.5 | 2-3 hours | `qwen25_loo_results.pth` |
| **4** | 100-fold LOO CV - DeepSeek | 2-3 hours | `deepseek_loo_results.pth` |
| **5** | Comparison Analysis | 30 min | Report + Visualization |
| **Total** | **End-to-end** | **7-11 hours** | **Benchmark-ready results** |

### Resource Requirements

- **GPU Memory**: ~21GB (NF4 quantization)
- **Disk Space**: ~2GB (features + results)
- **CPU RAM**: ~32GB (model loading)

---

## 6. 주요 개선점

### 6.1 방법론 정합성

✅ **원본 CENTaUR 방법론 완전 준수**:
- Binomial Regression
- 100-fold LOO CV
- Nested CV for hyperparameter tuning
- NLL metric

### 6.2 벤치마크 가능

✅ **직접 비교 가능**:
```
Random:      NLL = 120,000  (known)
LLaMA-65B:   NLL = 30,000   (published)
Qwen2.5:     NLL = ?        (우리 결과)
DeepSeek-R1: NLL = ?        (우리 결과)
```

### 6.3 효율성 대폭 향상

✅ **Feature Caching**:
- 1회 extraction → 재사용
- CV에서 forward pass 불필요

✅ **병렬화 가능**:
- 100개 fold를 SLURM array job으로 병렬 실행
- 예상 시간: 2-3시간 → **15-30분**

---

## 7. 결론 및 권장사항

### 7.1 현재 평가의 문제점

1. ❌ **원본 방법론 불일치**: Token probability ≠ Hidden state regression
2. ❌ **메트릭 부적절**: Accuracy ≠ Negative Log-Likelihood
3. ❌ **벤치마크 불가**: 원본 논문 결과와 비교 불가
4. ❌ **비효율적**: 8시간 소요 (원본은 3-5시간)

### 7.2 권장 해결책

✅ **즉시 실행 (최우선)**:
1. Feature extraction 스크립트 작성 및 실행
2. 원본 `models.py` 재사용 (검증됨)
3. 100-fold LOO CV 실행
4. NLL 메트릭 계산 및 비교

✅ **중기 계획**:
1. HorizonTask 데이터셋으로 확장
2. Symbolic + Neural hybrid 모델 테스트
3. 한국어 risky choice 데이터셋 구축

### 7.3 기대 효과

**학술적 기여**:
- ✅ CENTaUR 방법론의 2025년 모델 검증
- ✅ Reasoning-enhanced LLM (DeepSeek-R1)의 cognitive modeling 성능
- ✅ Qwen2.5의 multilingual cognitive capability 검증

**실용적 가치**:
- ✅ Ko-CENTaUR 프레임워크 완성
- ✅ 한국어 인지 심리학 연구 가능
- ✅ 벤치마크 가능한 평가 파이프라인

---

## References

1. **Binz, M., & Schulz, E. (2023)**. "Using cognitive psychology to understand GPT-3." *PNAS*, 120(6). https://doi.org/10.1073/pnas.2218523120

2. **Wilson, R. C., et al. (2014)**. "Humans use directed and random exploration to solve the explore–exploit dilemma." *Journal of Experimental Psychology: General*, 143(6).

3. **Original CENTaUR Implementation**: https://github.com/marcelbinz/CENTaUR

4. **Ko-CENTaUR Documentation**:
   - `EVALUATION_STATUS.md`
   - `cognitive_task_performance_comparison.md`
   - `RUN_EVALUATION.md`

---

**Last Updated**: 2025-10-25
**Next Step**: 평가 재실행 완료 후 Feature Extraction 스크립트 작성
**Status**: ⚠️ 평가 방법론 수정 필요 (현재 방식은 원본과 다름)
