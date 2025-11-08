# 추천 모델 리스트 (2025년 1월 기준)

CENTaUR 실험을 위한 추가 모델 후보들입니다.

---

## 🏆 Top Tier: OpenAI GPT-OSS (최우선 추천)

### GPT-OSS-120B
- **제공**: OpenAI
- **파라미터**: 120B (Mixture-of-Experts, 128 experts)
- **라이선스**: Apache 2.0 (완전 오픈)
- **성능**: Gemini 2.5 Flash, Claude Opus 4 수준
- **특징**:
  - OpenAI 최초 오픈 소스 모델
  - Cerebras에서 3,000 tokens/s (GPU의 60배)
  - AIME 2025 수학 평가 1위
  - Reasoning 능력 뛰어남
- **HuggingFace**: `openai/gpt-oss-120b`
- **추천 이유**: ✅ 대규모 모델, ✅ 최신 아키텍처, ✅ 추론 능력

### GPT-OSS-20B ✅ (이미 설정됨)
- **파라미터**: 20B
- **성능**: o3-mini와 동등 이상
- **현재 경로**: `/home/connectome/connectome1/models/gpt-oss-20b`

---

## 🇰🇷 Korean-Specialized Models (한국어 특화)

### 1. EXAONE-3.5-32B ⭐ **최우선 추천**
- **제공**: LG AI Research
- **파라미터**: 32B
- **라이선스**: Open
- **성능**:
  - MMLU-Pro: 81.8% (Phi-4: 76%, Mistral Small: 73.4%)
  - AIME 2025: 85.3%
  - 한국 모델 중 1위, 전세계 오픈 모델 중 4위
- **HuggingFace**: `LGAI-EXAONE/EXAONE-3.5-32B-Instruct`
- **추천 이유**:
  - ✅ 현재 EXAONE-3.0-7.8B보다 4배 큰 모델
  - ✅ 최신 버전 (3.5)
  - ✅ 한국어 + 영어 + 코드 혼합 학습
  - ✅ 글로벌 벤치마크 상위권

### 2. Motif-102B
- **제공**: Moreh
- **파라미터**: 102B
- **성능**: KMMLU 64.74 (GPT-4 초과)
- **HuggingFace**: `moreh/Motif-102B`
- **추천 이유**:
  - ✅ 한국어 특화 대규모 모델
  - ✅ GPT-4 수준 한국어 성능

### 3. Polyglot-Ko-12.8B
- **제공**: EleutherAI
- **파라미터**: 12.8B (1.3B, 3.8B, 5.8B도 가능)
- **HuggingFace**: `EleutherAI/polyglot-ko-12.8b`
- **추천 이유**:
  - ✅ 한국어 전용 학습
  - ✅ 다양한 크기 선택 가능
  - ⚠️ 2023년 모델 (다소 오래됨)

### 4. GECKO-7B
- **제공**: 개인 연구자
- **파라미터**: 7B
- **특징**: 한영 이중언어, 200B 토큰 학습
- **HuggingFace**: `sackoh/GECKO-7B`
- **추천 이유**: ✅ 경량, ✅ 이중언어

---

## 🌐 Global Open-Source Models

### Cerebras-GPT-13B
- **제공**: Cerebras
- **파라미터**: 13B (111M ~ 13B 다양)
- **특징**: Chinchilla scaling 기반 최적 학습
- **HuggingFace**: `cerebras/Cerebras-GPT-13B`
- **추천 이유**: ✅ 학습 효율성 검증됨

### GPT-NeoX-20B
- **제공**: EleutherAI
- **파라미터**: 20B
- **HuggingFace**: `EleutherAI/gpt-neox-20b`
- **추천 이유**: ✅ 널리 사용되는 기준 모델

### GPT-J-6B
- **제공**: EleutherAI
- **파라미터**: 6B
- **HuggingFace**: `EleutherAI/gpt-j-6b`
- **추천 이유**: ✅ 경량, ✅ 빠른 테스트용

---

## 📊 실험 우선순위

### Tier 1: 즉시 추가 (High Priority)
1. **EXAONE-3.5-32B** - 한국어 최고 성능, 글로벌 상위권
2. **GPT-OSS-120B** - OpenAI 최신 오픈소스, 추론 능력 탁월

### Tier 2: 비교 실험용 (Medium Priority)
3. **Motif-102B** - 한국어 특화 대규모 모델
4. **GPT-NeoX-20B** - 표준 기준 모델
5. **Polyglot-Ko-12.8B** - 한국어 전용 중형 모델

### Tier 3: 경량 실험용 (Low Priority)
6. **GECKO-7B** - 한영 이중언어 경량
7. **GPT-J-6B** - 빠른 프로토타입
8. **EXAONE-3.0-7.8B** ✅ (이미 있음)

---

## 💾 예상 저장 공간

| 모델 | 파라미터 | FP16 | NF4 (4-bit) |
|------|---------|------|-------------|
| GPT-OSS-120B | 120B | ~240GB | ~30GB |
| Motif-102B | 102B | ~204GB | ~26GB |
| EXAONE-3.5-32B | 32B | ~64GB | ~8GB |
| GPT-NeoX-20B | 20B | ~40GB | ~5GB |
| Polyglot-Ko-12.8B | 12.8B | ~26GB | ~3.3GB |
| GECKO-7B | 7B | ~14GB | ~1.8GB |

**서버 용량 확인 필요**: 7x RTX GPU (24GB 각) = 168GB total GPU memory
- NF4 양자화 사용 시 120B 모델도 가능 (30GB)

---

## 🔄 현재 설정된 모델

✅ Qwen2.5-32B-Instruct (Base + QLoRA)
✅ DeepSeek-R1-Distill-Qwen-32B (Base + QLoRA)
✅ EXAONE-3.0-7.8B-Instruct (Base)
✅ GPT-OSS-20B (Base)

---

## 📝 다음 단계

1. **EXAONE-3.5-32B 추가** - 한국어 최고 성능
2. **GPT-OSS-120B 다운로드** - OpenAI 최신 모델 (저장공간 확인 후)
3. **LOO CV 실행** - 현재 Base 모델들 (qwen25-base, deepseek-base)
4. **성능 비교** - Base vs Fine-tuned vs 새 모델들
