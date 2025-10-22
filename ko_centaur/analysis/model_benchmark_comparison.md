# Ko-CENTaUR Model Performance Comparison

**Date**: 2025-10-21
**Purpose**: Compare downstream task performance of models deployed for Ko-CENTaUR cognitive modeling

---

## Models Under Evaluation

| Model | Parameters | Size | Location | Status |
|-------|-----------|------|----------|--------|
| GPT-OSS-20B | 20B | 13GB | /home/connectome/connectome1/models/gpt-oss-20b | ✅ Downloaded |
| Qwen2.5-32B-Instruct | 32B | 62GB | /scratch/connectome/connectome1/ko-centaur/models/qwen2.5-32b-instruct | ✅ Downloaded + Training |
| DeepSeek-R1-Distill-Qwen-32B | 32B | 62GB | /home/connectome/connectome1/models/deepseek-r1-distill-qwen-32b | ✅ Validated + Training Pending |

---

## Benchmark Performance Summary

### 1. Mathematical Reasoning

| Benchmark | GPT-OSS-20B | Qwen2.5-32B-Instruct | DeepSeek-R1-Distill-Qwen-32B | Best |
|-----------|-------------|---------------------|----------------------------|------|
| **GSM8K** (Grade School Math) | ~75* | **95.9** | ~85* | Qwen2.5 |
| **MATH-500** | N/A | ~85* | **94.3** | DeepSeek-R1 |
| **AIME 2024** (Competition Math) | N/A | ~50* | **86.0** vs o1-mini (63.6) | DeepSeek-R1 |
| **AIME 2025** | N/A | N/A | **76.3** | DeepSeek-R1 |

### 2. General Knowledge & Reasoning

| Benchmark | GPT-OSS-20B | Qwen2.5-32B-Instruct | DeepSeek-R1-Distill-Qwen-32B | Best |
|-----------|-------------|---------------------|----------------------------|------|
| **MMLU** (Massive Multitask) | **Strong** vs 120B | **79.7-85+** (context-dependent) | ~80* | Qwen2.5 |
| **BBH** (Big-Bench Hard) | N/A | **78.2** | N/A | Qwen2.5 |
| **GPQA Diamond** (Science QA) | N/A | ~55* | **62.1** | DeepSeek-R1 |

### 3. Code Generation

| Benchmark | GPT-OSS-20B | Qwen2.5-32B-Instruct | DeepSeek-R1-Distill-Qwen-32B | Best |
|-----------|-------------|---------------------|----------------------------|------|
| **HumanEval** (Python) | **Strong** vs 120B | ~85* | N/A | GPT-OSS-20B† |
| **LiveCodeBench** | N/A | ~50* | **57.2** | DeepSeek-R1 |
| **McEval** (40 languages) | N/A | **Strong** | N/A | Qwen2.5 |

### 4. Korean Language Support

| Task | GPT-OSS-20B | Qwen2.5-32B-Instruct | DeepSeek-R1-Distill-Qwen-32B | Notes |
|------|-------------|---------------------|----------------------------|-------|
| **KMMLU** (Korean MMLU) | Limited | **Excellent** (multilingual focus) | **Good** (Qwen-based) | Qwen2.5 has explicit Korean support |
| **Korean Tokenization** | Inefficient | **Optimized** | **Optimized** (inherits Qwen tokenizer) | Critical for Ko-CENTaUR |

*Estimated based on model class and related benchmarks
†GPT-OSS-20B reportedly outperforms larger 120B variant on HumanEval

---

## Key Performance Insights

### 🏆 DeepSeek-R1-Distill-Qwen-32B
**Strengths**:
- **Superior reasoning**: AIME 2024 score (86.0) outperforms OpenAI o1-mini (63.6) by **35%**
- **Advanced math**: MATH-500 (94.3), AIME 2025 (76.3)
- **Science reasoning**: GPQA Diamond (62.1)
- **Code generation**: LiveCodeBench (57.2)

**Architecture**: Distilled from 671B DeepSeek-R1 using reinforcement learning
**Reasoning capability**: Enhanced through RL without supervised fine-tuning
**CENTaUR suitability**: ✅ **Excellent** - 5120-dim features, strong reasoning signal

### 🎯 Qwen2.5-32B-Instruct
**Strengths**:
- **General knowledge**: MMLU (79.7-85), BBH (78.2)
- **Math**: GSM8K (95.9) - **highest score**
- **Multilingual**: Explicit Korean (KMMLU), Japanese, Arabic support
- **Code**: McEval across 40 programming languages

**Architecture**: Qwen2.5 series with instruction tuning
**Multilingual focus**: Optimized tokenizer for Asian languages
**CENTaUR suitability**: ✅ **Excellent** - Korean language + broad knowledge

### ⚡ GPT-OSS-20B
**Strengths**:
- **Efficiency**: Outperforms 120B variant despite 6x smaller size
- **Code**: HumanEval strong performance
- **General**: MMLU competitive with larger models

**Architecture**: OpenAI open-source base model
**Efficiency**: Lower memory (13GB), faster inference
**CENTaUR suitability**: ✅ **Good** - Lightweight, strong baseline

---

## Comparative Analysis

### Performance vs Model Size

```
Reasoning Tasks (AIME 2024):
DeepSeek-R1-32B: 86.0 ████████████████████████
Qwen2.5-32B:     ~50* ███████████
GPT-OSS-20B:     N/A  -

Math Tasks (GSM8K):
Qwen2.5-32B:     95.9 ████████████████████████
DeepSeek-R1-32B: ~85* ████████████████████
GPT-OSS-20B:     ~75* ████████████████

General Knowledge (MMLU):
Qwen2.5-32B:     85   ████████████████████████
DeepSeek-R1-32B: ~80* ██████████████████████
GPT-OSS-20B:     Good ████████████████████

Code Generation (HumanEval):
GPT-OSS-20B:     High ████████████████████████
Qwen2.5-32B:     ~85* ███████████████████
DeepSeek-R1-32B: N/A  -
```

### Domain Specialization

| Domain | Best Model | Reason |
|--------|-----------|--------|
| **Competition Math** | DeepSeek-R1 | RL-enhanced reasoning (AIME 86.0) |
| **Grade School Math** | Qwen2.5 | GSM8K (95.9) highest score |
| **Science Reasoning** | DeepSeek-R1 | GPQA Diamond (62.1) |
| **Korean Language** | Qwen2.5 | Explicit multilingual support + KMMLU |
| **Code Generation** | GPT-OSS-20B | HumanEval strong vs 120B |
| **Efficiency** | GPT-OSS-20B | 13GB, competitive performance |

---

## CENTaUR Workflow Implications

### Cognitive Feature Extraction Quality

**Hidden State Dimensionality**:
- GPT-OSS-20B: ~5120 (estimated)
- Qwen2.5-32B: 5120 (confirmed in training)
- DeepSeek-R1-32B: **5120** (confirmed via validation)

**Reasoning Signal Strength**:
1. **DeepSeek-R1**: Strongest reasoning signal (RL-enhanced, o1-level)
2. **Qwen2.5**: Broad knowledge + Korean language
3. **GPT-OSS-20B**: Efficient baseline

### Recommended Usage Strategy

**For Korean Risky Choice Task (choices13k_korean)**:
1. **Primary**: Qwen2.5-32B (Korean tokenization + broad knowledge)
2. **Comparison**: DeepSeek-R1-32B (reasoning capability)
3. **Baseline**: GPT-OSS-20B (efficiency reference)

**For Complex Reasoning Tasks**:
1. **Primary**: DeepSeek-R1-32B (AIME-level reasoning)
2. **Comparison**: Qwen2.5-32B (general knowledge)

**For Multilingual Analysis**:
1. **Primary**: Qwen2.5-32B (KMMLU, multilingual MMLU)
2. **Comparison**: DeepSeek-R1-32B (Qwen-based tokenizer)

---

## Training Status

| Model | Training Job | Status | Expected Completion |
|-------|-------------|--------|---------------------|
| Qwen2.5-32B | Job 62813 | ⏳ Running (16h elapsed) | ~4-8h remaining |
| DeepSeek-R1-32B | Job 62866 | ⏸️ Pending (resource wait) | Starts after Qwen2.5 |
| GPT-OSS-20B | Not scheduled | 📋 Planned | After DeepSeek-R1 |

---

## Benchmark Sources

- **DeepSeek-R1**: [arXiv:2501.12948](https://arxiv.org/abs/2501.12948) (Official paper)
- **Qwen2.5**: [Qwen Technical Blog](https://qwenlm.github.io/blog/qwen2.5-llm/) (Official release)
- **GPT-OSS**: [arXiv:2508.12461](https://arxiv.org/abs/2508.12461) (Community evaluation)

**Last Updated**: 2025-10-21 23:00 KST
