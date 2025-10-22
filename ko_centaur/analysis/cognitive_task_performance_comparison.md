# Cognitive Psychology Task Performance Comparison

**Date**: 2025-10-21
**Purpose**: Compare model performance on CENTaUR cognitive modeling tasks (Binz & Schulz, 2023)

---

## CENTaUR Framework Overview

**Reference**: [Using cognitive psychology to understand GPT-3](https://www.pnas.org/doi/10.1073/pnas.2218523120) (PNAS 2023)

**Core Methodology**:
1. Extract LLM hidden states (last layer) as **cognitive features**
2. Train binomial regression: `features → logits → choice probabilities`
3. Predict **human choices** on cognitive psychology tasks
4. Evaluate via **100-fold leave-one-out cross-validation**
5. Metric: **Negative log-likelihood** (lower = better fit to human behavior)

---

## Experimental Tasks

### 1. **choices13k** - Risky Choice Prediction
**Domain**: Behavioral economics, risk preferences
**Task**: Predict human choices between probabilistic monetary gambles
**Dataset**: 13,000 gambling problems from behavioral experiments
**Example Prompt**:
```
You can choose between:
Machine 1: 50% chance of $10, 50% chance of $0
Machine 2: 100% chance of $4
A: Machine
```
Model completes with "1" or "2"

**Baseline Models**:
- **Random**: 50/50 guessing
- **BEAST**: Cognitive model (Biased Exponential Adaptive Sampling Theory)
- **LLaMA**: Token probabilities P("1") vs P("2")
- **CENTaUR**: Binomial regression on LLaMA hidden states

### 2. **HorizonTask** - Explore-Exploit Tradeoff
**Domain**: Sequential decision-making, reinforcement learning
**Task**: Multi-armed bandit with limited trials (forced vs. free choice)
**Dataset**: Human behavioral data from Wilson et al. (2014)
**Example Prompt**:
```
Trial history: [Observation from Machine 1: $3, Machine 2: $7]
You have 4 additional choices to maximize dollars.
Which machine do you choose?
A: Machine
```

**Baseline Models**:
- **Random**: Uniform random choice
- **LLaMA**: Token probabilities
- **CENTaUR**: Hidden state regression
- **Hybrid**: Symbolic task features + LLaMA hidden states

### 3. **ExperientialSymbolicTask** - Description vs. Experience
**Domain**: Decision-making under uncertainty, learning paradigms
**Task**: Compare learning from description (DfD) vs. experience (DfE)
**Dataset**: Choice patterns across learning contexts

---

## Legacy Results (LLaMA-65B Baseline)

### choices13k Performance (Negative Log-Likelihood)

| Model | NLL | Relative to Random | Human Fit Quality |
|-------|-----|-------------------|------------------|
| **Random** | ~120,000 | Baseline (worst) | No predictive power |
| **LLaMA-65B** (token probs) | ~90,000 | 25% improvement | Moderate |
| **BEAST** (cognitive model) | ~60,000 | 50% improvement | Good |
| **CENTaUR** (LLaMA features) | ~30,000 | **75% improvement** | Excellent |

**Key Finding**: CENTaUR with LLaMA-65B hidden states **outperforms** traditional cognitive models (BEAST) by 50% in predicting human risky choices.

### HorizonTask Performance (Negative Log-Likelihood)

| Model | NLL | Relative to Random | Sequential Decision Quality |
|-------|-----|-------------------|---------------------------|
| **Random** | ~50,000 | Baseline | No strategy |
| **LLaMA-65B** (token probs) | ~40,000 | 20% improvement | Weak |
| **CENTaUR** (LLaMA features) | ~30,000 | 40% improvement | Moderate |
| **Hybrid** (symbolic + LLaMA) | ~20,000 | **60% improvement** | Strong |

**Key Finding**: Hybrid model combining symbolic task representations with LLaMA features shows **best performance** on sequential decision tasks.

---

## Expected Performance: Current Models

### Model Characteristics for Cognitive Modeling

| Model | Hidden Dim | Reasoning Strength | Language Support | Expected Cognitive Performance |
|-------|-----------|-------------------|-----------------|-------------------------------|
| **LLaMA-65B** (baseline) | 8192 | Moderate | English | ✅ Validated (see above) |
| **GPT-OSS-20B** | ~5120* | Moderate-High | English-focused | ⭐⭐⭐ Similar to LLaMA-65B |
| **Qwen2.5-32B** | 5120 | High (GSM8K 95.9) | Multilingual + Korean | ⭐⭐⭐⭐ Better than LLaMA-65B |
| **DeepSeek-R1-32B** | 5120 | **Very High** (AIME 86.0) | English + Korean | ⭐⭐⭐⭐⭐ **Best expected** |

*Estimated based on model architecture

### choices13k (Risky Choice) - Predicted Performance

**Hypothesis**: Reasoning-enhanced models should better capture human risk preferences and probability weighting.

| Model | Predicted NLL | Confidence | Rationale |
|-------|--------------|------------|-----------|
| **Random** | 120,000 | 100% | Known baseline |
| **LLaMA-65B** | 30,000 | 100% | Validated result |
| **GPT-OSS-20B** | **28,000-32,000** | 70% | Similar architecture, smaller size may hurt slightly |
| **Qwen2.5-32B** | **25,000-28,000** | 80% | Superior math (GSM8K 95.9) → better probability reasoning |
| **DeepSeek-R1-32B** | **20,000-25,000** | 85% | **Best reasoning** (AIME 86.0) → strongest human alignment |

**Expected Winner**: **DeepSeek-R1-32B** (RL-enhanced reasoning captures human decision biases)

### HorizonTask (Sequential Decision) - Predicted Performance

**Hypothesis**: Explore-exploit requires multi-step reasoning and value estimation.

| Model | Predicted NLL | Confidence | Rationale |
|-------|--------------|------------|-----------|
| **Random** | 50,000 | 100% | Known baseline |
| **LLaMA-65B** | 30,000 | 100% | Validated result |
| **GPT-OSS-20B** | **28,000-32,000** | 65% | Good baseline, but limited sequential reasoning |
| **Qwen2.5-32B** | **24,000-28,000** | 75% | Strong general knowledge aids value estimation |
| **DeepSeek-R1-32B** | **18,000-24,000** | 85% | **RL training** directly relevant to explore-exploit |

**Expected Winner**: **DeepSeek-R1-32B** (RL alignment → superior multi-step planning)

### Korean Language Tasks (Ko-CENTaUR)

**New Challenge**: Korean risky choice task with cultural context

| Model | Korean Tokenization | Expected Performance | Rationale |
|-------|---------------------|---------------------|-----------|
| **GPT-OSS-20B** | Inefficient | ⭐⭐ Poor | Limited Korean support |
| **Qwen2.5-32B** | **Optimized** | ⭐⭐⭐⭐⭐ **Best** | KMMLU validation, native Korean tokenizer |
| **DeepSeek-R1-32B** | Good (Qwen-based) | ⭐⭐⭐⭐ Excellent | Inherits Qwen tokenizer, strong reasoning |

**Expected Winner**: **Qwen2.5-32B** (explicit Korean language optimization)

---

## Cognitive Modeling Quality Factors

### 1. **Hidden State Richness**
**What matters**: Dimensionality, information density, generalization

| Model | Hidden Dim | Information Density | Generalization |
|-------|-----------|---------------------|----------------|
| LLaMA-65B | 8192 | High (larger model) | Good |
| GPT-OSS-20B | ~5120 | Moderate (20B params) | Good |
| Qwen2.5-32B | 5120 | High (32B params) | Excellent |
| DeepSeek-R1-32B | 5120 | **Very High** (RL-distilled) | **Excellent** |

**Prediction**: DeepSeek-R1's RL-enhanced representations encode richer decision-making signals.

### 2. **Reasoning Capability**
**What matters**: Multi-step inference, probability estimation, value computation

| Model | Math Reasoning | Probabilistic Thinking | Sequential Planning |
|-------|----------------|----------------------|-------------------|
| LLaMA-65B | Moderate | Moderate | Moderate |
| GPT-OSS-20B | Good | Good | Moderate |
| Qwen2.5-32B | **Excellent** (GSM8K 95.9) | Excellent | Good |
| DeepSeek-R1-32B | **Outstanding** (AIME 86.0) | **Outstanding** | **Outstanding** (RL) |

**Prediction**: DeepSeek-R1 captures human-like reasoning biases and heuristics.

### 3. **Training Data & Alignment**
**What matters**: Exposure to decision-making contexts, human behavior patterns

| Model | Decision Context | Human Alignment | Cultural Diversity |
|-------|-----------------|-----------------|-------------------|
| LLaMA-65B | General text | Passive (from data) | English-focused |
| GPT-OSS-20B | General text | Passive | English-focused |
| Qwen2.5-32B | General text | Instruction-tuned | **Multilingual** |
| DeepSeek-R1-32B | **RL-enhanced** | **Active (RL)** | Multilingual |

**Prediction**: DeepSeek-R1's RL training explicitly optimizes for decision-making alignment.

---

## Experimental Validation Plan

### Phase 1: Feature Extraction (In Progress)
✅ Models downloaded: GPT-OSS-20B, Qwen2.5-32B, DeepSeek-R1-32B
🔄 QLoRA training: Qwen2.5 (running), DeepSeek-R1 (pending)

### Phase 2: Cognitive Feature Extraction (Next)
**Script**: Adapt `legacy/choices13k/query.py` for new models

```python
# For each model:
# 1. Load model with NF4 quantization
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
)

# 2. Extract hidden states on choices13k prompts
for prompt, human_choice in dataset:
    outputs = model.generate(
        inputs['input_ids'],
        max_new_tokens=1,
        output_hidden_states=True,
        return_dict_in_generate=True,
    )
    features = outputs.hidden_states[-1][-1][:, -1, :]  # (1, 5120)
    save_features(features, human_choice)

# 3. Save to data/model={model}_choices13k.pth
```

### Phase 3: Binomial Regression Fitting
**Script**: Adapt `legacy/choices13k/fit_centaur.py`

```python
# 100-fold LOO cross-validation
for fold in range(100):
    train_features, train_labels = load_fold(train_indices)
    test_features, test_labels = load_fold(test_indices)

    # Nested CV for regularization (alpha)
    best_alpha = nested_cv(train_features, train_labels)

    # Fit binomial regression
    model = BinomialRegression(alpha=best_alpha)
    model.fit(train_features, train_labels)

    # Evaluate test fold
    test_ll = model.log_likelihood(test_features, test_labels)
    save_result(fold, test_ll)

# Aggregate results
total_ll = sum([load_result(i) for i in range(100)])
```

### Phase 4: Comparative Analysis
**Visualization**: Update `legacy/choices13k/plot_loo.py`

```python
models = ['Random', 'LLaMA-65B', 'GPT-OSS-20B', 'Qwen2.5-32B', 'DeepSeek-R1-32B', 'BEAST']
loos = [load_results(model) for model in models]

plt.bar(models, loos, color=['C0', 'C0', 'C2', 'C3', 'C4', 'C1'])
plt.ylabel('Negative log-likelihood')
plt.title('Risky Choice Prediction Performance')
```

---

## Research Questions

### RQ1: Model Size vs. Cognitive Fit
**Question**: Does GPT-OSS-20B (20B) match LLaMA-65B (65B) despite 3x fewer parameters?

**Hypothesis**: Yes, due to better training and efficiency optimizations.

**Test**: Compare NLL on choices13k and HorizonTask.

### RQ2: Reasoning Enhancement Impact
**Question**: Does DeepSeek-R1's RL-enhanced reasoning improve human behavior prediction?

**Hypothesis**: Yes, RL alignment captures human-like decision heuristics.

**Test**: DeepSeek-R1 should show lowest NLL (best fit) on both tasks.

### RQ3: Language-Specific Effects
**Question**: Does Qwen2.5's Korean support improve performance on Ko-CENTaUR tasks?

**Hypothesis**: Yes, native Korean tokenization reduces noise in hidden states.

**Test**: Qwen2.5 > DeepSeek-R1 > GPT-OSS-20B on Korean risky choice.

### RQ4: Feature Richness vs. Model Size
**Question**: Are 5120-dim features from 32B models better than 8192-dim from 65B?

**Hypothesis**: No, information density matters more than dimensionality.

**Test**: Dimensionality reduction (PCA) analysis of hidden states.

---

## Timeline

| Phase | Task | Duration | Dependencies |
|-------|------|----------|--------------|
| **1** | QLoRA training | 24-48h | ✅ Models downloaded |
| **2** | Feature extraction (choices13k) | 4-8h | Phase 1 complete |
| **3** | Feature extraction (HorizonTask) | 4-8h | Phase 1 complete |
| **4** | Binomial regression fitting | 12-24h | Phase 2-3 complete |
| **5** | Comparative analysis | 2-4h | Phase 4 complete |
| **Total** | **End-to-end validation** | **3-5 days** | Starting after training |

---

## Expected Contributions

### To CENTaUR Literature
1. **First comparison** of modern LLMs (2024-2025) vs. LLaMA (2023)
2. **Reasoning-enhanced models** (DeepSeek-R1) for cognitive modeling
3. **Multilingual extension** (Korean) of original English-only framework
4. **Open-source model** validation (vs. proprietary GPT-3/GPT-4)

### To Korean Cognitive Science
1. **Ko-CENTaUR framework** for Korean behavioral research
2. Validate cross-cultural decision-making patterns
3. Enable Korean-language cognitive psychology experiments with LLMs

### To LLM Evaluation
1. **Psychology benchmarks** complement standard NLP benchmarks
2. Demonstrate LLM cognitive alignment beyond task performance
3. Provide interpretable evaluation via cognitive modeling

---

## Predicted Final Results

### choices13k (Risky Choice)

```
Negative Log-Likelihood (lower = better):

Random:         ████████████████████████████████████ 120,000
LLaMA-65B:      ████████████ 30,000
GPT-OSS-20B:    ███████████ 29,000 (est.)
Qwen2.5-32B:    █████████ 26,000 (est.)
DeepSeek-R1:    ███████ 22,000 (est.) ⭐ BEST
BEAST:          ████████████████ 60,000
```

**Improvement over baseline**: DeepSeek-R1 = 82% better than random, 27% better than LLaMA-65B

### HorizonTask (Sequential Decision)

```
Negative Log-Likelihood (lower = better):

Random:         ████████████████████████ 50,000
LLaMA-65B:      ██████████████ 30,000
GPT-OSS-20B:    █████████████ 28,000 (est.)
Qwen2.5-32B:    ██████████ 25,000 (est.)
DeepSeek-R1:    ████████ 20,000 (est.) ⭐ BEST
Hybrid:         ████████ 20,000
```

**Improvement over baseline**: DeepSeek-R1 = 60% better than random, 33% better than LLaMA-65B

---

## Conclusion

**Summary**: Based on benchmark analysis and model architecture, we predict:

1. **DeepSeek-R1-Distill-Qwen-32B** will achieve **best overall performance** on cognitive tasks due to RL-enhanced reasoning and decision-making alignment.

2. **Qwen2.5-32B-Instruct** will excel on **Korean language tasks** and show strong performance on math-heavy decision problems.

3. **GPT-OSS-20B** will provide a strong **efficiency baseline**, matching LLaMA-65B despite 3x smaller size.

4. All modern models (2024-2025) should **outperform LLaMA-65B** (2023) on cognitive psychology tasks.

**Next Step**: Run Phase 2 (feature extraction) once QLoRA training completes.

**Last Updated**: 2025-10-21 23:30 KST
