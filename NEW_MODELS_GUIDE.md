# 새 모델 추가 및 실험 가이드

이 가이드는 조사한 최신 LLM 모델들을 CENTaUR 실험에 추가하는 방법을 설명합니다.

---

## 📋 조사 완료 모델 목록

### ✅ 이미 설정된 모델
- Qwen2.5-32B-Instruct (Base + QLoRA)
- DeepSeek-R1-Distill-Qwen-32B (Base + QLoRA)
- EXAONE-3.0-7.8B-Instruct
- GPT-OSS-20B

### 🆕 새로 추가된 모델 (설정 완료)
1. **EXAONE-3.5-32B** ⭐ 최우선 추천
2. **GPT-OSS-120B** (다운로드 필요)
3. **Motif-102B** (다운로드 필요)
4. **GPT-NeoX-20B** (다운로드 필요)
5. **Polyglot-Ko-12.8B** (다운로드 필요)
6. **GECKO-7B** (다운로드 필요)
7. **GPT-J-6B** (다운로드 필요)
8. **Cerebras-GPT-13B** (다운로드 필요)

---

## 🚀 Quick Start

### 1단계: 우선순위 모델 다운로드

#### EXAONE-3.5-32B (즉시 사용 가능 - HuggingFace에서 자동 다운로드)
```bash
# 다운로드 없이 바로 실행 가능
cd ~/git/CENTaUR
./run_exaone35_full.sh
```

#### GPT-OSS-120B (다운로드 필요)
```bash
# 1. 모델 다운로드 (30GB with NF4, ~2-4시간)
python scripts/download_new_models.py --model gpt-oss-120b

# 2. Feature extraction
./run_gpt_oss_120b_full.sh
```

### 2단계: 우선순위별 다운로드

```bash
# Priority 1 모델만 다운로드
python scripts/download_new_models.py --priority 1

# Priority 2 모델 다운로드
python scripts/download_new_models.py --priority 2

# 모든 모델 다운로드
python scripts/download_new_models.py --model all
```

### 3단계: 모델 목록 확인

```bash
# 사용 가능한 모델 리스트 보기
python scripts/download_new_models.py --list
```

---

## 📊 실험 실행

### 개별 모델 테스트 (10 samples)

```bash
# EXAONE-3.5
python scripts/extract_centaur_features.py --model exaone35-base --n_samples 10

# GPT-OSS-120B (다운로드 후)
python scripts/extract_centaur_features.py --model gpt-oss-120b --n_samples 10

# Motif
python scripts/extract_centaur_features.py --model motif --n_samples 10

# GPT-NeoX
python scripts/extract_centaur_features.py --model gpt-neox --n_samples 10

# Polyglot-Ko
python scripts/extract_centaur_features.py --model polyglot-ko --n_samples 10

# GECKO
python scripts/extract_centaur_features.py --model gecko --n_samples 10

# GPT-J
python scripts/extract_centaur_features.py --model gpt-j --n_samples 10

# Cerebras
python scripts/extract_centaur_features.py --model cerebras --n_samples 10
```

### 전체 데이터셋 실행 (100 samples)

```bash
# EXAONE-3.5
./run_exaone35_full.sh

# GPT-OSS-120B
./run_gpt_oss_120b_full.sh

# 기타 모델 (수동 실행)
python scripts/extract_centaur_features.py --model motif
python scripts/extract_centaur_features.py --model gpt-neox
```

### LOO Cross-Validation

```bash
# Feature extraction 완료 후
python scripts/fit_centaur_loo_cv_flexible.py \
    outputs/exaone35_base_features.npz \
    outputs/exaone35_nll_results.json \
    --model_name "EXAONE-3.5-32B"
```

---

## 🔧 서버 설정 (dgx-spark 또는 connectome)

### 모델 다운로드 (SSH)

```bash
# 서버 접속
ssh server  # 또는 ssh dgx-spark

# Tmux 세션 시작
tmux new -s model_download

# CENTaUR 디렉토리로 이동
cd ~/git/CENTaUR
source venv/bin/activate

# 우선순위 1 모델 다운로드 (EXAONE-3.5, GPT-OSS-120B)
python scripts/download_new_models.py --priority 1

# Ctrl+b, d로 detach (백그라운드 실행 유지)
```

### 다운로드 진행 상황 모니터링

```bash
# 다른 터미널에서
ssh server 'tmux attach -t model_download'

# 또는 디렉토리 크기 확인
ssh server 'du -sh /home/connectome/connectome1/models/*'
```

---

## 💡 추천 실험 시나리오

### Scenario A: 빠른 검증 (1-2일)
1. **EXAONE-3.5-32B** - 한국어 최고 성능
2. **현재 Base 모델 LOO CV 완료** (qwen25-base, deepseek-base)
3. 성능 비교

### Scenario B: 종합 비교 (3-5일)
1. **Priority 1 모델** (EXAONE-3.5, GPT-OSS-120B)
2. **Priority 2 모델** (Motif, GPT-NeoX, Polyglot-Ko)
3. 모든 모델 LOO CV
4. 벤치마크 리포트 생성

### Scenario C: OSS GPT 집중 실험 (2-3일)
1. **GPT-OSS-120B** (OpenAI 최신)
2. **GPT-NeoX-20B** (EleutherAI 표준)
3. **GPT-J-6B** (경량 비교)
4. OSS GPT 계열 성능 분석

---

## 📈 예상 결과

### 성공 기준
- **NLL < 0.69** (random baseline보다 좋음)
- **Feature similarity < 90%**
- **Generation bias < 60%**

### 모델별 예상
- **EXAONE-3.5-32B**: 한국어 특화 → 높은 성능 기대
- **GPT-OSS-120B**: 대규모 MoE → 추론 능력 우수 예상
- **Motif-102B**: 한국어 GPT-4 수준 → 최고 성능 가능
- **GPT-NeoX-20B**: 표준 기준 → 중간 성능
- **Polyglot-Ko**: 한국어 전용 → EXAONE보다 낮을 수 있음

---

## 🛠️ 트러블슈팅

### 다운로드 실패
```bash
# 재시도
python scripts/download_new_models.py --model exaone35 --cache ~/.cache/huggingface
```

### GPU 메모리 부족
```bash
# 양자화 사용 (기본값)
python scripts/extract_centaur_features.py --model gpt-oss-120b

# 양자화 없이 실행 (메모리 충분할 때만)
python scripts/extract_centaur_features.py --model gpt-oss-120b --no-quantization
```

### 모델 경로 오류
```bash
# 모델이 다운로드되었는지 확인
ls -lh /home/connectome/connectome1/models/

# HuggingFace 캐시 확인
ls -lh ~/.cache/huggingface/hub/
```

---

## 📁 생성된 파일

### 스크립트
- `scripts/download_new_models.py` - 모델 다운로드
- `scripts/extract_centaur_features.py` - Feature extraction (업데이트됨)
- `run_exaone35_full.sh` - EXAONE-3.5 실행
- `run_gpt_oss_120b_full.sh` - GPT-OSS-120B 실행

### 문서
- `RECOMMENDED_MODELS.md` - 모델 상세 정보
- `NEW_MODELS_GUIDE.md` - 이 가이드

### 설정 변경
- `scripts/extract_centaur_features.py` - 8개 새 모델 추가

---

## 🔗 참고 자료

- **모델 상세 정보**: `RECOMMENDED_MODELS.md`
- **원본 실험 결과**: `claudedocs/COMPREHENSIVE_FINDINGS_2025-11-06.md`
- **서버 실행 가이드**: `QUICKSTART.md`
- **DGX 실험 가이드**: `DGX_EXPERIMENT_GUIDE.md`

---

## ✅ 다음 단계 체크리스트

- [ ] EXAONE-3.5-32B 테스트 실행 (다운로드 불필요)
- [ ] 현재 Base 모델 LOO CV 완료 확인
- [ ] GPT-OSS-120B 다운로드 (용량 확인 후)
- [ ] Priority 1 모델 실험
- [ ] 결과 비교 및 리포트 생성
