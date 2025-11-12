# CENTaUR 실험 계획 리뷰 (간단 버전)

## 실험 목표
LLM의 hidden state를 사용하여 CENTaUR 방법론으로 선택 과제 예측 성능 평가

## 실험 방법론
1. **Feature Extraction**: 각 모델의 last-layer hidden state 추출
2. **100-Fold LOO CV**: Leave-One-Out Cross-Validation
3. **Binomial Regression**: 선택 확률 예측
4. **평가 지표**: NLL (Negative Log-Likelihood)

## 대상 모델 (5개)

### 1. Qwen2.5-32B Base ✅
- **Feature**: 완료 (1.0 MB)
- **LOO CV**: 미완료
- **우선순위**: 높음 (Feature 있음, 바로 실행 가능)

### 2. DeepSeek-R1 Base ✅
- **Feature**: 완료 (1.0 MB)
- **LOO CV**: 미완료
- **우선순위**: 높음 (Feature 있음, 바로 실행 가능)

### 3. EXAONE-3.5-32B ⏳
- **Feature**: 진행 중/대기 중
- **LOO CV**: 대기 중
- **우선순위**: 높음 (NGC PyTorch 컨테이너 준비 완료)

### 4. GPT-OSS-20B ❌
- **Feature**: 미시작
- **LOO CV**: 대기 중
- **우선순위**: 중간

### 5. Kimi K2 ❌
- **Feature**: 미시작
- **LOO CV**: 대기 중
- **우선순위**: 중간

## 현재 상태

### 완료된 작업
- ✅ 모델 추가 및 설정 (모든 모델)
- ✅ Qwen2.5-32B Base Feature Extraction
- ✅ DeepSeek-R1 Base Feature Extraction
- ✅ 모니터링 시스템 설정 (10분마다 체크, 이메일 알림)
- ✅ NGC PyTorch 컨테이너 설치 및 GPU 성능 측정

### 진행 중/대기 중
- ⏳ EXAONE-3.5-32B Feature Extraction (NGC 컨테이너 준비 완료)
- ❌ Qwen2.5-32B Base LOO CV (GPU 호환성 문제)
- ❌ DeepSeek-R1 Base LOO CV (GPU 호환성 문제)
- ❌ GPT-OSS-20B Feature Extraction
- ❌ Kimi K2 Feature Extraction

## 다음 단계 (우선순위)

### 즉시 실행 가능
1. **Qwen2.5-32B Base LOO CV** (Feature 있음)
   - GPU 호환성 문제 해결 필요
   - 또는 CPU로 실행

2. **DeepSeek-R1 Base LOO CV** (Feature 있음)
   - GPU 호환성 문제 해결 필요
   - 또는 CPU로 실행

### 컨테이너에서 실행
3. **EXAONE-3.5-32B Feature Extraction**
   - NGC PyTorch 컨테이너 사용
   - GPU 성능 측정 완료 (28,100 GFLOPS)

### 나중에 실행
4. GPT-OSS-20B Feature Extraction
5. Kimi K2 Feature Extraction

## 예상 일정

### Week 1 (현재)
- ✅ 모델 설정 완료
- ✅ Feature Extraction 2개 완료
- ⏳ EXAONE-3.5-32B 진행 중
- ❌ LOO CV 2개 대기 중

### Week 2
- EXAONE-3.5-32B 완료
- LOO CV 3개 완료
- GPT-OSS-20B 시작

### Week 3
- 모든 Feature Extraction 완료
- 모든 LOO CV 완료
- 결과 분석 시작

### Week 4
- 종합 분석 및 리포트 작성
- 논문 초안 작성

## 주요 이슈

1. **GPU 호환성**: CUDA capability 12.1 지원 문제
   - 해결: NGC PyTorch 컨테이너 사용 또는 CPU fallback

2. **모니터링**: 자동화 완료 ✅
   - 10분마다 체크
   - 완료 시 이메일 알림

3. **성능**: NGC 컨테이너에서 우수한 성능 확인
   - 28,100 GFLOPS
   - 128GB 메모리 활용 가능

---

**리뷰 날짜**: 2025-11-09

