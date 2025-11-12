# 작업 진행 상황 요약 (2025-11-08)

## ✅ 완료된 작업

### 1. 모델 추가 및 설정
- ✅ GPT-OSS-20B 모델 추가
- ✅ EXAONE-3.5-32B 모델 추가 (코드 수정 완료)
- ✅ Kimi K2 모델 추가
- ✅ 실험 스크립트 생성 (run_*.sh)
- ✅ 문서 작성 (ONBOARDING.md, WORK_PLAN.md 등)

### 2. Feature Extraction 완료
- ✅ **Qwen2.5-32B Base**: `outputs/qwen25_base_features.npz` (완료)
- ✅ **DeepSeek-R1 Base**: `outputs/deepseek_base_features.npz` (완료)

### 3. LOO CV 부분 완료
- ✅ **Qwen2.5-32B**: 50/100 folds 완료 (중단됨)
  - NLL = 0.7623 (랜덤보다 나쁨)
  - Generation bias 발견 (72% → B)

### 4. 문제 분석 완료
- ✅ Generation bias 분석 완료
- ✅ Feature similarity 분석 완료 (95.25%)
- ✅ 원인 분석 완료 (Fine-tuned 모델 편향 문제)

---

## 🔄 진행 중인 작업

### EXAONE-3.5-32B Feature Extraction
- ⚠️ **상태**: 실행 시도 중, trust_remote_code 문제 해결 중
- **문제**: Custom code 실행 필요
- **해결**: 코드 수정 완료 (`is_local = True` 설정)
- **다음**: 재실행 필요

---

## ❌ 아직 시작하지 않은 작업

### Feature Extraction
- ❌ EXAONE-3.5-32B (재실행 필요)
- ❌ GPT-OSS-20B
- ❌ Kimi K2

### LOO CV
- ❌ Qwen2.5-32B Base (feature는 있음, LOO CV만 실행하면 됨)
- ❌ DeepSeek-R1 Base (feature는 있음, LOO CV만 실행하면 됨)
- ❌ EXAONE-3.5-32B
- ❌ GPT-OSS-20B
- ❌ Kimi K2

---

## 📊 현재 상태 요약

| 작업 | 상태 | 비고 |
|------|------|------|
| 모델 추가 | ✅ 완료 | EXAONE-3.5, Kimi K2, GPT-OSS 추가 |
| Qwen2.5 Base Feature | ✅ 완료 | 1MB |
| DeepSeek Base Feature | ✅ 완료 | 1MB |
| EXAONE-3.5 Feature | ⚠️ 진행 중 | trust_remote_code 수정 완료, 재실행 필요 |
| Qwen2.5 Base LOO CV | ❌ 미시작 | Feature 있음, 바로 실행 가능 |
| DeepSeek Base LOO CV | ❌ 미시작 | Feature 있음, 바로 실행 가능 |

---

## 🎯 다음 단계 (우선순위)

### 즉시 실행 가능
1. **Qwen2.5-32B Base LOO CV** (가장 빠름)
   - Feature 이미 있음
   - 바로 실행 가능
   ```bash
   python scripts/fit_centaur_loo_cv_flexible.py \
       outputs/qwen25_base_features.npz \
       outputs/qwen25_base_nll_results.json \
       --model_name "Qwen2.5-32B-Base"
   ```

2. **EXAONE-3.5-32B Feature Extraction 재실행**
   - 코드 수정 완료
   - 재실행 필요
   ```bash
   ./run_exaone35_full.sh
   ```

3. **DeepSeek-R1 Base LOO CV**
   - Feature 이미 있음
   ```bash
   python scripts/fit_centaur_loo_cv_flexible.py \
       outputs/deepseek_base_features.npz \
       outputs/deepseek_base_nll_results.json \
       --model_name "DeepSeek-R1-32B-Base"
   ```

---

## 📝 Git 상태

- ✅ 최근 커밋: `b917eaa7` - 모델 추가 및 문서 업데이트
- ✅ 모든 변경사항 푸시 완료

---

**마지막 업데이트**: 2025-11-08 20:50

