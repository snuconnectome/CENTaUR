# EXAONE-3.5-32B & Kimi K2 추가 완료

## ✅ 추가된 모델

### 1. EXAONE-3.5-32B-Instruct ⭐⭐⭐⭐⭐
- **모델 ID**: `LGAI-EXAONE/EXAONE-3.5-32B-Instruct`
- **옵션**: `--model exaone35-base`
- **특징**:
  - 한국어 최고 성능 (MMLU-Pro 81.8%)
  - 전세계 오픈 모델 중 4위
  - 현재 EXAONE-3.0-7.8B의 4배 큰 모델 (32B)
  - HuggingFace에서 자동 다운로드

### 2. Kimi K2-Instruct ⭐⭐⭐⭐
- **모델 ID**: `moonshot-ai/Kimi-K2-Instruct`
- **옵션**: `--model kimi-k2`
- **특징**:
  - Multi-Agent 기능 특화
  - 128K 토큰 컨텍스트 지원
  - MoE 아키텍처 (1T 파라미터, 활성 32B)
  - 네이티브 INT4 양자화 지원
  - HuggingFace에서 자동 다운로드

---

## 🚀 사용 방법

### EXAONE-3.5-32B

```bash
# Feature extraction
./run_exaone35_full.sh

# 또는 직접 실행
python scripts/extract_centaur_features.py \
    --model exaone35-base \
    --dataset ko_centaur/data/choices13k_100.jsonl \
    --output outputs/exaone35_features.npz

# LOO CV
./run_exaone35_loo_cv.sh
```

### Kimi K2

```bash
# Feature extraction (매우 큰 모델, 시간 오래 걸림)
./run_kimi_k2_full.sh

# 또는 직접 실행
python scripts/extract_centaur_features.py \
    --model kimi-k2 \
    --dataset ko_centaur/data/choices13k_100.jsonl \
    --output outputs/kimi_k2_features.npz

# LOO CV
./run_kimi_k2_loo_cv.sh
```

---

## 📝 생성된 파일

### 스크립트
- `run_exaone35_full.sh` - EXAONE-3.5-32B feature extraction
- `run_exaone35_loo_cv.sh` - EXAONE-3.5-32B LOO CV
- `run_kimi_k2_full.sh` - Kimi K2 feature extraction
- `run_kimi_k2_loo_cv.sh` - Kimi K2 LOO CV

### 코드 수정
- `scripts/extract_centaur_features.py` - 모델 옵션 추가

---

## ⚠️ 주의사항

### EXAONE-3.5-32B
- ✅ HuggingFace에서 자동 다운로드
- ✅ 적절한 크기 (32B)
- ✅ 한국어 성능 검증됨

### Kimi K2
- ⚠️ 매우 큰 모델 (1T 파라미터)
- ⚠️ 다운로드 시간 오래 걸림 (수 시간)
- ⚠️ 저장공간 필요 (~500GB INT4 기준)
- ⚠️ GPU 메모리 확인 필요 (활성 파라미터만 ~16GB INT4)
- ⚠️ 한국어 성능 미확인 (테스트 필요)

---

## 🎯 다음 단계

1. **EXAONE-3.5-32B 실험 시작** (가장 빠름)
   ```bash
   ./run_exaone35_full.sh
   ```

2. **Kimi K2 다운로드 및 테스트** (저장공간 확인 후)
   ```bash
   # 저장공간 확인
   df -h
   
   # 다운로드 시작 (백그라운드)
   ./run_kimi_k2_full.sh
   ```

3. **성능 비교**
   - EXAONE-3.5-32B vs 기존 모델들
   - Kimi K2 vs 기존 모델들
   - Multi-Agent 실험 (Kimi K2)

---

**작성일**: 2025-11-08  
**상태**: 모델 추가 완료, 실험 준비됨

