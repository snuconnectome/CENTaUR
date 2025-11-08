# LLM 리더보드 기반 모델 추천 (2025년 1월)

## 🎯 CENTaUR 실험 우선순위 추천

리더보드 성적, reasoning/inference 능력, multi-agent 지원을 종합하여 추천합니다.

---

## 🥇 1순위: EXAONE-3.5-32B-Instruct ⭐⭐⭐⭐⭐

### 리더보드 성적
- **MMLU-Pro**: 81.8% (Phi-4: 76%, Mistral Small: 73.4% 능가)
- **AIME 2025**: 85.3%
- **한국 모델 중 1위**, 전세계 오픈 모델 중 **4위**
- **한국어 성능**: 최고 수준

### CENTaUR 적합성
- ✅ **현재 EXAONE-3.0-7.8B의 4배 큰 버전** (32B)
- ✅ **추론 능력 우수** (MMLU-Pro 81.8%)
- ✅ **한국어 + 영어 + 코드 혼합 학습**
- ✅ **HuggingFace 즉시 다운로드 가능**: `LGAI-EXAONE/EXAONE-3.5-32B-Instruct`
- ✅ **적절한 크기**: 32B (현재 Qwen2.5-32B와 동일 크기)
- ✅ **NF4 양자화 시 ~8GB** (GPU 메모리 충분)

### 구현 난이도: ⭐ (쉬움)
- 현재 EXAONE-3.0-7.8B와 동일 계열
- `extract_centaur_features.py`에 옵션만 추가

**추천 이유**: 한국어 최고 성능 + 글로벌 상위권 + 적절한 크기

---

## 🥈 2순위: GPT-OSS-120B ⭐⭐⭐⭐

### 리더보드 성적
- **AIME 2025 수학 평가**: **1위**
- **성능**: Gemini 2.5 Flash, Claude Opus 4 수준
- **Reasoning 능력**: 탁월

### CENTaUR 적합성
- ✅ **OpenAI 최초 완전 오픈소스** (Apache 2.0)
- ✅ **Mixture-of-Experts (128 experts)** - 효율적 추론
- ✅ **추론 능력 뛰어남** (AIME 2025 1위)
- ✅ **HuggingFace**: `openai/gpt-oss-120b`
- ⚠️ **대형 모델**: NF4 양자화 시 ~30GB 필요
- ⚠️ **저장공간 확인 필요**

### 구현 난이도: ⭐⭐ (중간)
- GPT-OSS-20B와 동일 계열 (이미 구현됨)
- 대형 모델이라 메모리 관리 필요

**추천 이유**: 최신 OpenAI 모델 + 추론 능력 1위 + 완전 오픈소스

---

## 🥉 3순위: Qwen2.5-72B-Instruct ⭐⭐⭐⭐

### 리더보드 성적
- **HuggingFace Open LLM Leaderboard**: 상위권
- **다국어 지원**: 한국어 포함
- **현재 사용 중인 32B의 대형 버전**

### CENTaUR 적합성
- ✅ **현재 모델과 동일 계열** (구현 매우 쉬움)
- ✅ **더 나은 성능 기대** (32B → 72B)
- ✅ **HuggingFace**: `Qwen/Qwen2.5-72B-Instruct`
- ⚠️ **GPU 메모리 요구사항 높음**: NF4 양자화 시 ~18GB

### 구현 난이도: ⭐ (매우 쉬움)
- 현재 Qwen2.5-32B 코드 재사용 가능

**추천 이유**: 현재 모델 확장 + 검증된 성능 + 구현 쉬움

---

## 🔍 4순위: DeepSeek-R1 (Reasoning 특화) ⭐⭐⭐⭐

### 리더보드 성적
- **Reasoning 벤치마크**: 상위권
- **Chain-of-Thought 추론**: 우수
- **현재 사용 중**: DeepSeek-R1-Distill-Qwen-32B (fine-tuned)

### CENTaUR 적합성
- ✅ **Reasoning 능력 특화** (multi-step inference)
- ✅ **Base 모델 테스트 필요**: 현재 fine-tuned만 사용 중
- ✅ **HuggingFace**: `deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`
- ⚠️ **Base 모델 추가 필요**: fine-tuned 편향 문제 회피

### 구현 난이도: ⭐ (쉬움)
- 이미 DeepSeek 사용 중, base 버전만 추가

**추천 이유**: Reasoning 특화 + multi-step inference 지원

---

## 🌐 5순위: Llama 3.2/3.3 11B-Instruct ⭐⭐⭐

### 리더보드 성적
- **HuggingFace Open LLM Leaderboard**: 상위권
- **다국어 지원**: 한국어 포함

### CENTaUR 적합성
- ✅ **검증된 성능** (Meta)
- ✅ **적절한 크기**: 11B
- ✅ **HuggingFace**: `meta-llama/Llama-3.2-11B-Instruct`
- ⚠️ **인증 필요**: HuggingFace 라이선스 동의

### 구현 난이도: ⭐⭐ (중간)
- 표준 구현, 인증만 필요

**추천 이유**: 검증된 기준 모델 + 다국어 지원

---

## 📊 리더보드 기반 성능 비교

| 모델 | 파라미터 | MMLU-Pro | AIME 2025 | 한국어 | Reasoning | 우선순위 |
|------|----------|----------|-----------|--------|-----------|----------|
| **EXAONE-3.5-32B** | 32B | **81.8%** | 85.3% | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **1위** |
| **GPT-OSS-120B** | 120B | 높음 | **1위** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **2위** |
| **Qwen2.5-72B** | 72B | 높음 | 높음 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | **3위** |
| **DeepSeek-R1** | 32B | 높음 | 높음 | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | **4위** |
| **Llama 3.2 11B** | 11B | 높음 | 중간 | ⭐⭐⭐⭐ | ⭐⭐⭐ | **5위** |

---

## 🎯 Multi-Agent / Inference 능력 고려

### Reasoning 특화 모델
1. **GPT-OSS-120B**: AIME 2025 1위, 추론 능력 탁월
2. **DeepSeek-R1**: Chain-of-Thought 추론 특화
3. **EXAONE-3.5-32B**: MMLU-Pro 81.8% (추론 능력 우수)

### Multi-Agent 지원 가능 모델
- 모든 모델이 multi-agent 프레임워크와 결합 가능
- **EXAONE-3.5-32B**: 한국어 + 추론 능력으로 multi-agent에 적합
- **GPT-OSS-120B**: 대규모 모델로 복잡한 multi-agent 작업에 적합

---

## 💡 최종 추천 순서

### 즉시 추가 (1주일 내)

1. **EXAONE-3.5-32B** ⭐⭐⭐⭐⭐
   - 이유: 한국어 최고 성능 + 글로벌 상위권 + 적절한 크기
   - 작업: `extract_centaur_features.py`에 옵션 추가
   - 예상 시간: 1일

2. **Qwen2.5-72B** ⭐⭐⭐⭐
   - 이유: 현재 모델 확장 + 구현 쉬움
   - 작업: 기존 코드 재사용
   - 예상 시간: 0.5일

### 저장공간 확인 후 추가 (2주일 내)

3. **GPT-OSS-120B** ⭐⭐⭐⭐
   - 이유: OpenAI 최신 + 추론 능력 1위
   - 작업: 저장공간 확인 후 다운로드
   - 예상 시간: 2-3일 (다운로드 포함)

### Base 모델 추가 (현재 fine-tuned 편향 문제 해결)

4. **DeepSeek-R1 Base** ⭐⭐⭐
   - 이유: Reasoning 특화, fine-tuned 편향 회피
   - 작업: Base 모델로 feature extraction
   - 예상 시간: 1일

---

## 🚀 구현 계획

### Week 1: 즉시 추가 모델
- [ ] EXAONE-3.5-32B 추가
- [ ] Qwen2.5-72B 추가
- [ ] Feature extraction 실행

### Week 2: 대형 모델 (저장공간 확인 후)
- [ ] GPT-OSS-120B 다운로드
- [ ] Feature extraction 실행

### Week 3: Base 모델 비교
- [ ] DeepSeek-R1 Base 추가
- [ ] 모든 Base 모델 LOO CV 실행
- [ ] Fine-tuned vs Base 성능 비교

---

## 📝 결론

**최우선 추천**: **EXAONE-3.5-32B**
- 한국어 최고 성능
- 글로벌 리더보드 상위권 (4위)
- 적절한 크기 (32B)
- 구현 쉬움
- CENTaUR 실험에 최적

**다음 단계**: EXAONE-3.5-32B부터 추가하여 실험 시작

---

**작성일**: 2025-11-08  
**기준**: HuggingFace Open LLM Leaderboard, MMLU-Pro, AIME 2025, Agent Leaderboard

