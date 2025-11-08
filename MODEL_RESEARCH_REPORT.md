# CENTaUR 실험용 추가 모델 조사 보고서
**작성일**: 2025-11-08  
**목적**: CENTaUR 실험에 추가할 수 있는 우수한 오픈소스 LLM 모델 조사

---

## 📊 조사 개요

현재 CENTaUR 프로젝트에서 사용 중인 모델:
- ✅ Qwen2.5-32B-Instruct (QLoRA fine-tuned)
- ✅ DeepSeek-R1-Distill-Qwen-32B (QLoRA fine-tuned)
- ✅ EXAONE-3.0-7.8B-Instruct
- ✅ GPT-OSS-20B (최근 추가)

**조사 기준**:
1. 오픈소스 모델 (HuggingFace에서 다운로드 가능)
2. 한국어 지원 또는 다국어 지원
3. 10B-70B 파라미터 범위 (현재 실험 환경에 적합)
4. 리더보드 상위 성능
5. CENTaUR 인지 모델링에 적합한 아키텍처

---

## 🌟 추천 모델 (우선순위별)

### Tier 1: 최우선 추천 모델

#### 1. **SOLAR-10.7B-Instruct** ⭐⭐⭐⭐⭐
- **개발사**: 업스테이지 (Upstage)
- **파라미터**: 10.7B
- **리더보드 성적**: 
  - HuggingFace Open LLM Leaderboard: 300B 이하 모델 중 1위 (과거)
  - 한국어 성능 우수
- **특징**:
  - Depth Up-Scaling 기법으로 효율적인 확장
  - 한국어 지원 우수
  - 이미 프로젝트에 baseline으로 포함됨 (확인 필요)
- **HuggingFace**: `upstage/SOLAR-10.7B-Instruct-v1.0`
- **추천 이유**: 한국어 성능 우수, 적절한 크기, 검증된 성능

#### 2. **LUXIA 2.5** ⭐⭐⭐⭐⭐
- **개발사**: 솔트룩스 (Saltlux)
- **파라미터**: ~35B 이하
- **리더보드 성적**:
  - HuggingFace Open LLM Leaderboard: 35B 이하 모델 기준 1위
  - HellaSwag (상식): 91.88점
  - ARC (추론): 77.47점
  - 평균 점수: ~77.74점
- **특징**:
  - 한국 기업 개발
  - 상식 및 추론 능력 우수
  - 요약 및 정확도에서 GPT-3.5-TURBO 능가
- **HuggingFace**: 확인 필요 (Saltlux/Saltlux-LUXIA-2.5 등)
- **추천 이유**: 리더보드 1위, 추론 능력 우수 (CENTaUR에 중요)

#### 3. **Sapie-gemma2-9B-IT** ⭐⭐⭐⭐
- **개발사**: 솔트웨어 (Saltware)
- **파라미터**: 9B
- **리더보드 성적**:
  - 호랑이 리더보드 (W&B): 오픈소스 모델 분야 1위
  - ChatGPT-4, Claude 3 Opus에 이어 높은 성능
- **특징**:
  - Gemma2 기반
  - 한국어 특화
  - 효율적인 리소스 활용
- **HuggingFace**: 확인 필요
- **추천 이유**: 한국어 리더보드 1위, 적절한 크기

---

### Tier 2: 고려할 만한 모델

#### 4. **Llama 3.2 / 3.3** ⭐⭐⭐⭐
- **개발사**: Meta
- **파라미터**: 3B, 11B, 70B 등 다양한 크기
- **리더보드 성적**:
  - HuggingFace Open LLM Leaderboard 상위권
  - 다국어 지원 (한국어 포함)
- **특징**:
  - 검증된 아키텍처
  - 널리 사용되는 오픈소스 모델
  - 다양한 크기 옵션
- **HuggingFace**: `meta-llama/Llama-3.2-11B-Instruct` 등
- **추천 이유**: 검증된 성능, 다국어 지원, 다양한 크기

#### 5. **Qwen2.5-72B-Instruct** ⭐⭐⭐⭐
- **개발사**: Alibaba Cloud
- **파라미터**: 72B
- **리더보드 성적**:
  - HuggingFace Open LLM Leaderboard 상위권
  - 다국어 지원 (한국어 포함)
- **특징**:
  - 현재 사용 중인 Qwen2.5-32B의 대형 버전
  - 더 나은 성능 기대
  - GPU 메모리 요구사항 높음
- **HuggingFace**: `Qwen/Qwen2.5-72B-Instruct`
- **추천 이유**: 현재 모델과 동일 계열, 더 큰 용량으로 성능 향상 기대

#### 6. **Mistral 7B / 8x7B** ⭐⭐⭐
- **개발사**: Mistral AI
- **파라미터**: 7B, 8x7B (MoE)
- **리더보드 성적**:
  - HuggingFace Open LLM Leaderboard 상위권
  - 다국어 지원
- **특징**:
  - 효율적인 아키텍처
  - MoE (Mixture of Experts) 버전 제공
- **HuggingFace**: `mistralai/Mistral-7B-Instruct-v0.3` 등
- **추천 이유**: 검증된 성능, 효율적 구조

#### 7. **Gemma 2 27B** ⭐⭐⭐
- **개발사**: Google DeepMind
- **파라미터**: 27B
- **리더보드 성적**:
  - HuggingFace Open LLM Leaderboard 상위권
- **특징**:
  - Google의 오픈소스 모델
  - 다국어 지원
- **HuggingFace**: `google/gemma-2-27b-it`
- **추천 이유**: Google 검증, 적절한 크기

---

### Tier 3: 한국어 특화 모델

#### 8. **EEVE-Korean-10.8B-v1.0** ⭐⭐⭐
- **개발사**: 야놀자 (Yanolja)
- **파라미터**: 10.8B
- **특징**:
  - 영어 중심 LLM을 한국어에 효율적으로 확장
  - 20억 토큰만으로 비영어권 성능 향상
- **HuggingFace**: 확인 필요
- **추천 이유**: 한국어 특화, 효율적인 확장 방법

#### 9. **Kanana (카나나)** ⭐⭐⭐
- **개발사**: 카카오 (Kakao)
- **파라미터**: 8B 이하
- **리더보드 성적**:
  - 호랑이 리더보드: 80억 파라미터 이하 모델 중 1위
- **특징**:
  - 카카오 자체 개발
  - 한국어 특화
- **HuggingFace**: 확인 필요
- **추천 이유**: 한국어 리더보드 1위

#### 10. **MoMo-70B** ⭐⭐⭐
- **개발사**: 모레 (Moreh)
- **파라미터**: 70B
- **리더보드 성적**:
  - HuggingFace Open LLM Leaderboard: 77.29점 (과거 1위)
- **특징**:
  - Alibaba Qwen 모델 미세조정
  - 대형 모델
- **HuggingFace**: 확인 필요
- **추천 이유**: 리더보드 1위 경력, 대형 모델

---

## 📈 리더보드 비교

### HuggingFace Open LLM Leaderboard (영어)
| 모델 | 평균 점수 | 순위 | 파라미터 |
|------|----------|------|----------|
| LUXIA 2.5 | 77.74 | 1위 (35B 이하) | ~35B |
| MoMo-70B | 77.29 | 1위 (과거) | 70B |
| SOLAR-10.7B | 높음 | 상위권 | 10.7B |
| Llama 3.2/3.3 | 높음 | 상위권 | 3B-70B |

### 한국어 리더보드
| 리더보드 | 1위 모델 | 점수 |
|----------|----------|------|
| 호랑이 리더보드 (W&B) | Sapie-gemma2-9B-IT | 오픈소스 1위 |
| 오픈 Ko-LLM 리더보드 (NIA) | 딥 솔라 (deep-solar) | 61.45점 |
| 호랑이 리더보드 (8B 이하) | Kanana (카카오) | 1위 |

---

## 🎯 CENTaUR 실험에 적합한 모델 선정 기준

### 필수 조건:
1. ✅ 오픈소스 (HuggingFace 다운로드 가능)
2. ✅ 한국어 지원 또는 다국어 지원
3. ✅ 10B-70B 파라미터 범위
4. ✅ 리더보드 상위 성능

### CENTaUR 특화 고려사항:
- **추론 능력**: 선택 예측에 중요 (ARC 점수)
- **상식 능력**: 인간 의사결정 이해에 중요 (HellaSwag 점수)
- **Generation bias**: Qwen2.5에서 발견된 문제 회피
- **Feature diversity**: 높은 similarity 문제 회피

---

## 💡 최종 추천 (우선순위)

### 즉시 추가 추천 (Top 3)

1. **SOLAR-10.7B-Instruct**
   - 이유: 한국어 성능 우수, 적절한 크기, 검증된 성능
   - 구현 난이도: ⭐ (이미 baseline에 포함 가능성)

2. **LUXIA 2.5**
   - 이유: 리더보드 1위, 추론 능력 우수 (ARC 77.47)
   - 구현 난이도: ⭐⭐ (HuggingFace 확인 필요)

3. **Sapie-gemma2-9B-IT**
   - 이유: 한국어 리더보드 1위, 적절한 크기
   - 구현 난이도: ⭐⭐ (HuggingFace 확인 필요)

### 추가 고려 모델

4. **Qwen2.5-72B-Instruct**
   - 이유: 현재 모델의 대형 버전, 성능 향상 기대
   - 구현 난이도: ⭐ (동일 계열)
   - 주의: GPU 메모리 요구사항 높음

5. **Llama 3.2/3.3 11B**
   - 이유: 검증된 성능, 다국어 지원
   - 구현 난이도: ⭐ (널리 사용)

---

## 🔧 구현 계획

### Phase 1: 즉시 추가 가능한 모델
1. **SOLAR-10.7B** - 이미 baseline에 포함 가능성 확인
2. **Llama 3.2/3.3** - HuggingFace에서 직접 다운로드 가능

### Phase 2: HuggingFace 확인 후 추가
1. **LUXIA 2.5** - HuggingFace 모델 ID 확인 필요
2. **Sapie-gemma2-9B-IT** - HuggingFace 모델 ID 확인 필요

### Phase 3: 대형 모델 (GPU 리소스 확인 후)
1. **Qwen2.5-72B** - GPU 메모리 요구사항 높음
2. **MoMo-70B** - 대형 모델, 리소스 확인 필요

---

## 📝 다음 단계

1. **HuggingFace 모델 ID 확인**
   - LUXIA 2.5, Sapie, Kanana 등 한국 모델들의 정확한 HuggingFace 경로 확인

2. **모델 다운로드 테스트**
   - 우선순위 상위 모델부터 다운로드 가능 여부 확인

3. **Feature Extraction 통합**
   - `extract_centaur_features.py`에 모델 추가
   - 실험 스크립트 생성

4. **Generation Bias 사전 분석**
   - Qwen2.5에서 발견된 문제를 다른 모델에서도 확인
   - 편향 없는 모델 선별

5. **성능 비교 실험**
   - 모든 모델로 feature extraction
   - LOO CV 비교
   - NLL 성능 비교

---

## 🔗 참고 자료

### 리더보드
- [HuggingFace Open LLM Leaderboard](https://huggingface.co/spaces/HuggingFaceH4/open_llm_leaderboard)
- [호랑이 리더보드 (W&B)](https://wandb.ai/wandb/ko-llm-leaderboard)
- [오픈 Ko-LLM 리더보드 (NIA)](https://open-ko-llm-leaderboard.kr/)

### 모델 정보
- SOLAR: [arXiv:2312.15166](https://arxiv.org/abs/2312.15166)
- LUXIA: 솔트룩스 공식 사이트
- Qwen2.5: [HuggingFace](https://huggingface.co/Qwen)
- Llama 3: [Meta AI](https://ai.meta.com/llama/)

---

**작성자**: AI Assistant  
**검토 필요**: HuggingFace 모델 ID 확인, GPU 리소스 확인

