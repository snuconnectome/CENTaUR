# Kimi 모델 조사 및 다운로드 결과

## HuggingFace 모델 확인 결과

[HuggingFace moonshotai 페이지](https://huggingface.co/moonshotai)에서 확인한 Kimi 모델들:

### Kimi-K2 시리즈 (1T 파라미터)

1. **moonshotai/Kimi-K2-Instruct** ✅
   - 파라미터: 1T (1 Trillion)
   - 특징: Multi-Agent 특화, Agentic Intelligence
   - 업데이트: 2일 전 (최신)
   - 다운로드: 79.1k
   - **상태**: 다운로드 시작됨

2. **moonshotai/Kimi-K2-Base** ✅
   - 파라미터: 1T
   - 상태: 공개됨

3. **moonshotai/Kimi-K2-Thinking** ✅
   - 파라미터: 1T
   - 특징: Thinking/Reasoning 특화
   - 업데이트: 18시간 전 (최신)

4. **moonshotai/Kimi-K2-Instruct-0905** ✅
   - 파라미터: 1T
   - 특징: 특정 날짜 버전

### Kimi-Linear 시리즈 (49B 파라미터)

5. **moonshotai/Kimi-Linear-48B-A3B-Instruct** ✅
   - 파라미터: 49B
   - 아키텍처: MoE with Kimi Delta Attention
   - 특징: 효율적인 Attention 메커니즘
   - **추천**: Kimi-K2보다 작고 실용적

6. **moonshotai/Kimi-Linear-48B-A3B-Base** ✅
   - 파라미터: 49B

### Kimi-Dev 시리즈

7. **moonshotai/Kimi-Dev-72B** ✅
   - 파라미터: 73B
   - 특징: SWE-Agents 특화

## 중요 발견

### 모델 ID 수정
- **잘못된 ID**: `moonshot-ai/Kimi-K2-Instruct` ❌
- **올바른 ID**: `moonshotai/Kimi-K2-Instruct` ✅

### Custom Code 필요
- 모든 Kimi 모델은 `trust_remote_code=True` 필요
- `extract_centaur_features.py`에서 `is_local = True` 설정 필요

## 코드 수정 완료

### extract_centaur_features.py
- 모델 ID 수정: `moonshot-ai` → `moonshotai`
- `is_local = True` 추가 (trust_remote_code 활성화)

### 다운로드 스크립트
- `trust_remote_code=True` 추가
- 백그라운드 다운로드 시작

## 다운로드 상태

### 완료된 모델
- ✅ EXAONE-3.5-32B: 238.46 GB
- ✅ GPT-OSS-20B: 76.93 GB

### 진행 중
- ⏳ Kimi-K2-Instruct: 다운로드 시작됨
  - 모델 ID 수정 완료
  - trust_remote_code 설정 완료
  - 백그라운드 다운로드 실행 중

## CENTaUR 실험 추천

### 우선순위 1: Kimi-K2-Instruct
- Multi-Agent 특화
- 최신 모델 (업데이트 2일 전)
- 가장 많이 다운로드됨 (79.1k)

### 우선순위 2: Kimi-Linear-48B-A3B-Instruct
- 더 작은 크기 (49B vs 1T)
- 실용적인 선택
- 효율적인 Attention 메커니즘

### 우선순위 3: Kimi-Dev-72B
- 중간 크기 (73B)
- SWE-Agents 특화

## 다음 단계

1. ✅ 모델 ID 수정 완료
2. ✅ trust_remote_code 설정 완료
3. ⏳ Kimi-K2-Instruct 다운로드 진행 중
4. 다운로드 완료 후 Feature Extraction 시작

---

**참고**: [moonshotai HuggingFace 페이지](https://huggingface.co/moonshotai)

