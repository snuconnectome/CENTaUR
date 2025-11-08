# CENTaUR dgx-spark 실험 빠른 시작 가이드

## ✅ 환경 준비 완료
- GPU: NVIDIA GB10 (Blackwell) ✅
- PyTorch: 2.10.0.dev20251105+cu128 ✅
- CUDA: 작동 확인 완료 ✅
- 의존성: transformers, peft, accelerate 설치 완료 ✅

## 실험 단계별 실행

### 1단계: 벤치마크 실행 (1-2시간)

```bash
# dgx-spark에 접속
ssh dgx-spark

# tmux 세션 시작 (필수 - 연결 끊겨도 계속 실행됨)
tmux new -s centaur_exp

# CENTaUR 디렉토리로 이동
cd ~/git/CENTaUR

# 벤치마크 실행
./benchmark_gpu.sh
```

**예상 결과**:
- 10 samples: ~1-2분
- 100 samples: ~10-20분
- GPU 메모리 프로파일 생성

**확인**:
```bash
ls -lh outputs/benchmark_*.npz
tail logs/gpu_profile.csv
```

### 2단계: Feature Extraction (각 2-4시간)

#### Qwen2.5-32B 모델

```bash
# tmux 세션에서 실행 (이미 시작했으면 생략)
cd ~/git/CENTaUR

# Qwen 모델 feature extraction
./run_qwen25_full.sh
```

**중간에 확인** (다른 터미널에서):
```bash
ssh dgx-spark
tmux attach -t centaur_exp  # 세션 재접속

# 또는
ssh dgx-spark 'tail -f ~/git/CENTaUR/logs/qwen25_base_extraction.log'
```

#### DeepSeek 모델

```bash
# Qwen 완료 후
./run_deepseek_full.sh
```

#### GPT-OSS-20B 모델

```bash
# 모델 다운로드 (처음 한 번만)
./download_gpt_oss.sh
# 또는 서버에서 SLURM으로:
# sbatch ko_centaur/scripts/download_gpt_oss_optimized.sh

# Feature extraction
./run_gpt_oss_full.sh
```

#### EXAONE-3.5-32B 모델 ⭐ (최우선 추천)

```bash
# Feature extraction (HuggingFace에서 자동 다운로드)
./run_exaone35_full.sh
```

**특징**:
- 한국어 최고 성능 (MMLU-Pro 81.8%)
- 전세계 오픈 모델 중 4위
- 현재 EXAONE-3.0-7.8B의 4배 큰 모델

#### Kimi K2 모델 (Multi-Agent 특화)

```bash
# Feature extraction (매우 큰 모델, 다운로드 시간 오래 걸림)
./run_kimi_k2_full.sh
```

**특징**:
- Multi-Agent 기능 특화
- 128K 토큰 컨텍스트 지원
- MoE 아키텍처 (1T 파라미터, 활성 32B)
- ⚠️ 매우 큰 모델 (저장공간 확인 필요)

**세션 detach**: `Ctrl+b, d` (백그라운드 실행 유지)

### 3단계: 100-Fold LOO Cross-Validation (각 3-4시간)

#### Qwen2.5 평가

```bash
cd ~/git/CENTaUR
./run_qwen25_loo_cv.sh
```

#### DeepSeek 평가

```bash
./run_deepseek_loo_cv.sh
```

#### GPT-OSS-20B 평가

```bash
./run_gpt_oss_loo_cv.sh
```

#### EXAONE-3.5-32B 평가 ⭐

```bash
./run_exaone35_loo_cv.sh
```

#### Kimi K2 평가

```bash
./run_kimi_k2_loo_cv.sh
```

### 4단계: 결과 확인

```bash
cd ~/git/CENTaUR
python generate_report.py
```

## tmux 세션 관리

```bash
# 세션 목록 보기
tmux ls

# 세션 재접속
tmux attach -t centaur_exp

# 세션에서 빠져나오기 (백그라운드 실행 유지)
# Ctrl+b, d

# 세션 종료 (실험 완료 후)
tmux kill-session -t centaur_exp
```

## 실험 진행 상황 모니터링

```bash
# GPU 사용량 실시간 모니터링
ssh dgx-spark 'watch -n 1 nvidia-smi'

# 로그 실시간 확인
ssh dgx-spark 'tail -f ~/git/CENTaUR/logs/qwen25_base_extraction.log'

# 생성된 파일 확인
ssh dgx-spark 'ls -lh ~/git/CENTaUR/outputs/'
```

## 전체 자동 실행 (12-17시간)

**옵션**: 모든 단계를 순차 실행하는 마스터 스크립트

```bash
cd ~/git/CENTaUR
cat > run_all_experiments.sh << 'SCRIPT'
#!/bin/bash
set -e

echo "=== CENTaUR 전체 실험 시작 ==="
date

# Phase 1: Benchmark
echo -e "\n### Phase 1: Benchmark ###"
./benchmark_gpu.sh

# Phase 2: Feature Extraction
echo -e "\n### Phase 2: Qwen2.5 Feature Extraction ###"
./run_qwen25_full.sh

echo -e "\n### Phase 2: DeepSeek Feature Extraction ###"
./run_deepseek_full.sh

# Phase 3: LOO CV
echo -e "\n### Phase 3: Qwen2.5 LOO CV ###"
./run_qwen25_loo_cv.sh

echo -e "\n### Phase 3: DeepSeek LOO CV ###"
./run_deepseek_loo_cv.sh

# Phase 4: Report
echo -e "\n### Phase 4: Final Report ###"
python generate_report.py

echo -e "\n=== 모든 실험 완료 ==="
date
SCRIPT

chmod +x run_all_experiments.sh

# tmux에서 실행
tmux new -s centaur_exp
./run_all_experiments.sh
# Ctrl+b, d로 detach
```

## 예상 타임라인

| 시작 시간 | 단계 | 소요 시간 | 종료 예상 |
|----------|------|----------|----------|
| 00:00 | Benchmark | 1-2시간 | 02:00 |
| 02:00 | Qwen Feature Extraction | 2-3시간 | 05:00 |
| 05:00 | DeepSeek Feature Extraction | 2-3시간 | 08:00 |
| 08:00 | Qwen LOO CV | 3-4시간 | 12:00 |
| 12:00 | DeepSeek LOO CV | 3-4시간 | 16:00 |
| 16:00 | Report | 0.5시간 | 16:30 |

**총 소요 시간**: 약 12-17시간

## 트러블슈팅

### GPU 메모리 부족 에러
```bash
# OOM 발생 시 quantization 확인
# 모든 스크립트는 이미 --use_quantization 플래그 포함
```

### 처리가 너무 느림
```bash
# GPU 사용 중인지 확인
nvidia-smi

# CPU 사용 중이면 PyTorch CUDA 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}')"
```

### tmux 세션이 사라짐
```bash
# 세션 복구 불가 - 처음부터 재시작 필요
# 향후 중간 체크포인트 저장 기능 추가 예정
```

## 다음 단계

실험 완료 후:
1. 결과 리포트 확인 (`python generate_report.py`)
2. NLL 값이 30K 이하면 성공 ✅
3. Fine-tuned 모델로 추가 실험 고려
4. 논문 작성 준비

## 문의

실험 중 문제 발생 시:
- 로그 파일 확인: `~/git/CENTaUR/logs/`
- GPU 상태 확인: `nvidia-smi`
- Python 환경 확인: `source venv/bin/activate && python --version`
