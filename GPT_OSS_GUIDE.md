# GPT-OSS-20B 실험 가이드

## 개요

GPT-OSS-20B는 OpenAI의 오픈소스 GPT 모델입니다. CENTaUR 실험에 추가하여 다른 모델들과 성능을 비교할 수 있습니다.

## 모델 정보

- **모델명**: GPT-OSS-20B
- **크기**: ~14GB (safetensors, 최적화된 버전)
- **HuggingFace**: `openai/gpt-oss-20b`
- **로컬 경로**: `/home/connectome/connectome1/models/gpt-oss-20b` (서버)
- **특징**: OpenAI의 오픈소스 GPT 아키텍처

## 설치 및 다운로드

### 1. 로컬 환경

```bash
cd ~/git/CENTaUR
./download_gpt_oss.sh
```

### 2. 서버 환경 (SLURM)

```bash
sbatch ko_centaur/scripts/download_gpt_oss_optimized.sh
```

다운로드 진행 상황 확인:
```bash
squeue -u $USER
tail -f /scratch/connectome/connectome1/ko-centaur/logs/gpt_oss_download_*.out
```

### 3. 수동 다운로드

```bash
huggingface-cli download openai/gpt-oss-20b \
    --exclude "original/*" "metal/*" \
    --local-dir ~/models/gpt-oss-20b \
    --resume-download
```

## 실험 실행

### 1. Feature Extraction

```bash
cd ~/git/CENTaUR
source venv/bin/activate

# 전체 데이터셋
./run_gpt_oss_full.sh

# 또는 직접 실행
python scripts/extract_centaur_features.py \
    --model gpt-oss \
    --dataset ko_centaur/data/choices13k_100.jsonl \
    --output outputs/gpt_oss_features.npz
```

**예상 소요 시간**: 2-4시간 (GPU 필요)

### 2. LOO Cross-Validation

```bash
./run_gpt_oss_loo_cv.sh

# 또는 직접 실행
python scripts/fit_centaur_loo_cv_flexible.py \
    outputs/gpt_oss_features.npz \
    outputs/gpt_oss_nll_results.json \
    --model_name "GPT-OSS-20B"
```

**예상 소요 시간**: 3-4시간 (CPU 가능)

### 3. 결과 확인

```bash
# Feature 확인
python -c "
import numpy as np
data = np.load('outputs/gpt_oss_features.npz')
print(f'Features: {data[\"features\"].shape}')
print(f'Labels: {data[\"labels\"].shape}')
"

# NLL 결과 확인
python -c "
import json
with open('outputs/gpt_oss_nll_results.json') as f:
    results = json.load(f)
print(f'Mean NLL: {results[\"mean_nll\"]:.2f}')
print(f'Std NLL: {results[\"std_nll\"]:.2f}')
"
```

## 예상 결과

### 성능 기준

- **Random Baseline**: NLL ≈ 0.6931
- **LLaMA-65B (원본)**: NLL ≈ 30K (로그 스케일)
- **목표**: NLL < 0.6931 (랜덤보다 좋은 성능)

### 비교 모델

| 모델 | NLL | 상태 |
|------|-----|------|
| Random | 0.6931 | Baseline |
| Qwen2.5-32B | 0.7623 | 편향 문제 |
| DeepSeek-R1 | TBD | 실험 중 |
| **GPT-OSS-20B** | **TBD** | **실험 예정** |

## 트러블슈팅

### 모델을 찾을 수 없음

```bash
# 모델 경로 확인
ls -lh /home/connectome/connectome1/models/gpt-oss-20b

# 다운로드 확인
du -sh /home/connectome/connectome1/models/gpt-oss-20b
# 예상 크기: ~14GB
```

### GPU 메모리 부족

```bash
# Quantization 자동 사용 (기본값)
# --use_quantization 플래그는 기본적으로 활성화됨
```

### 다운로드 실패

```bash
# 디스크 공간 확인
df -h /home/connectome/connectome1

# 재시도
huggingface-cli download openai/gpt-oss-20b \
    --local-dir /home/connectome/connectome1/models/gpt-oss-20b \
    --resume-download
```

## 다음 단계

1. ✅ 모델 다운로드
2. ✅ Feature extraction 실행
3. ✅ LOO CV 실행
4. 🔄 결과 분석 및 다른 모델과 비교
5. 🔄 Generation bias 분석 (필요시)

## 참고 자료

- [GPT-OSS-20B HuggingFace](https://huggingface.co/openai/gpt-oss-20b)
- [CENTaUR 원본 논문](https://arxiv.org/abs/2306.03917)
- [프로젝트 README](README.md)

