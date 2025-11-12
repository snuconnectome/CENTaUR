# 작업 진행 상황 실시간 업데이트

## 🚀 현재 실행 중인 작업

### 1. EXAONE-3.5-32B Feature Extraction ✅ 실행 중
- **프로세스 ID**: 3716237
- **상태**: 실행 중 (모델 로딩 중)
- **예상 시간**: 2-4시간
- **로그**: `logs/exaone35_extraction.log`
- **출력**: `outputs/exaone35_features.npz`

### 2. Qwen2.5-32B Base LOO CV ⏳ 확인 중
- **상태**: 실행 시도됨
- **로그**: `logs/qwen25_base_loo_cv.log`
- **출력**: `outputs/qwen25_base_nll_results.json`

---

## 📊 실시간 모니터링 명령어

### EXAONE-3.5-32B 진행 상황 확인
```bash
# 로그 확인
tail -f logs/exaone35_extraction.log

# 프로세스 확인
ps aux | grep extract_centaur_features | grep exaone35

# GPU 사용량 확인
watch -n 1 nvidia-smi
```

### Qwen2.5-32B Base LOO CV 진행 상황 확인
```bash
# 로그 확인
tail -f logs/qwen25_base_loo_cv.log

# 결과 파일 확인
ls -lh outputs/qwen25_base_nll_results.json
```

---

## ✅ 다음 단계

### EXAONE-3.5-32B 완료 후
1. Feature extraction 완료 확인
2. Generation bias 분석
3. LOO CV 실행

### Qwen2.5-32B Base LOO CV 완료 후
1. NLL 결과 확인
2. Generation bias 분석 (필요시)
3. DeepSeek-R1 Base LOO CV 시작

---

**업데이트 시간**: 2025-11-08 21:01

