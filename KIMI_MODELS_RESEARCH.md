# Kimi 모델 조사 결과

## HuggingFace 모델 확인

HuggingFace의 [moonshotai 페이지](https://huggingface.co/moonshotai)에서 확인한 Kimi 모델들:

### 1. Kimi-K2 시리즈 (1T 파라미터, MoE)

#### moonshotai/Kimi-K2-Instruct ✅
- **파라미터**: 1T (1 Trillion)
- **아키텍처**: MoE (Mixture of Experts)
- **특징**: Multi-Agent 특화, Agentic Intelligence
- **업데이트**: 2일 전 (최신)
- **다운로드**: 79.1k
- **상태**: ✅ 공개됨

#### moonshotai/Kimi-K2-Base ✅
- **파라미터**: 1T
- **아키텍처**: MoE
- **업데이트**: Jul 13
- **다운로드**: 699
- **상태**: ✅ 공개됨

#### moonshotai/Kimi-K2-Thinking ✅
- **파라미터**: 1T
- **특징**: Thinking/Reasoning 특화
- **업데이트**: 18시간 전 (최신)
- **다운로드**: 12.5k
- **상태**: ✅ 공개됨

#### moonshotai/Kimi-K2-Instruct-0905 ✅
- **파라미터**: 1T
- **특징**: 0905 버전 (특정 날짜 버전)
- **업데이트**: 2일 전
- **다운로드**: 40k
- **상태**: ✅ 공개됨

### 2. Kimi-Linear 시리즈 (49B 파라미터)

#### moonshotai/Kimi-Linear-48B-A3B-Instruct ✅
- **파라미터**: 49B (48B + 1B)
- **아키텍처**: MoE with Kimi Delta Attention
- **특징**: 실험적 모델, 효율적인 Attention
- **업데이트**: 8일 전
- **다운로드**: 190k
- **상태**: ✅ 공개됨

#### moonshotai/Kimi-Linear-48B-A3B-Base ✅
- **파라미터**: 49B
- **아키텍처**: MoE with Kimi Delta Attention
- **업데이트**: 8일 전
- **다운로드**: 574
- **상태**: ✅ 공개됨

### 3. Kimi-Dev 시리즈

#### moonshotai/Kimi-Dev-72B ✅
- **파라미터**: 73B
- **특징**: SWE-Agents (Software Engineering Agents) 특화
- **업데이트**: Jun 17
- **다운로드**: 6.76k
- **상태**: ✅ 공개됨

### 4. Kimi-VL 시리즈 (Vision-Language)

#### moonshotai/Kimi-VL-A3B-Thinking-2506 ✅
- **파라미터**: 16B
- **특징**: Vision-Language, Thinking 모델
- **업데이트**: Aug 18
- **다운로드**: 183k
- **상태**: ✅ 공개됨

#### moonshotai/Kimi-VL-A3B-Thinking ✅
- **파라미터**: 16B
- **특징**: Vision-Language, Thinking 모델
- **업데이트**: Aug 18
- **다운로드**: 12k
- **상태**: ✅ 공개됨

#### moonshotai/Kimi-VL-A3B-Instruct ✅
- **파라미터**: 16B
- **특징**: Vision-Language, Instruct 모델
- **업데이트**: Jul 30
- **다운로드**: 83k
- **상태**: ✅ 공개됨

## 중요 발견

### 모델 ID 오류 수정 필요
- **잘못된 ID**: `moonshot-ai/Kimi-K2-Instruct` ❌
- **올바른 ID**: `moonshotai/Kimi-K2-Instruct` ✅

### CENTaUR 실험에 적합한 모델

#### 우선순위 1: Kimi-K2-Instruct
- **이유**: Multi-Agent 특화, 최신 모델
- **모델 ID**: `moonshotai/Kimi-K2-Instruct`
- **크기**: 1T (매우 큼, 저장공간 확인 필요)

#### 우선순위 2: Kimi-Linear-48B-A3B-Instruct
- **이유**: 49B로 상대적으로 작음, 효율적인 Attention
- **모델 ID**: `moonshotai/Kimi-Linear-48B-A3B-Instruct`
- **크기**: 49B (더 실용적)

#### 우선순위 3: Kimi-Dev-72B
- **이유**: 73B, SWE-Agents 특화
- **모델 ID**: `moonshotai/Kimi-Dev-72B`
- **크기**: 73B

## 권장 사항

1. **Kimi-K2-Instruct**: 가장 강력하지만 1T로 매우 큼
2. **Kimi-Linear-48B-A3B-Instruct**: 실용적인 크기, 최신 아키텍처
3. **Kimi-Dev-72B**: 중간 크기, 특화 모델

## 다음 단계

1. 모델 ID 수정 (`moonshot-ai` → `moonshotai`)
2. 저장공간 확인 후 적절한 모델 선택
3. 선택한 모델 다운로드 시작

---

**참고**: [moonshotai HuggingFace 페이지](https://huggingface.co/moonshotai)

