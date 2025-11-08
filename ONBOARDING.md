# CENTaUR 프로젝트 온보딩 가이드

이 문서는 CENTaUR 프로젝트에 새로 참여하는 개발자를 위한 종합 가이드입니다.

## 📋 목차

1. [프로젝트 개요](#프로젝트-개요)
2. [환경 설정](#환경-설정)
3. [프로젝트 구조](#프로젝트-구조)
4. [주요 워크플로우](#주요-워크플로우)
5. [개발 가이드](#개발-가이드)
6. [실험 실행](#실험-실행)
7. [참고 자료](#참고-자료)

---

## 프로젝트 개요

### CENTaUR란?

**CENTaUR** (Cognitive Embeddings for Natural Understanding & Representation)는 대규모 언어 모델(LLM)을 인지 모델로 변환하는 연구 프로젝트입니다.

- **원본 논문**: Binz, M., & Schulz, E. (2023). Turning large language models into cognitive models. ICLR 2023.
- **목적**: LLM의 hidden state를 인지 표현(cognitive representation)으로 사용하여 인간의 의사결정을 예측
- **현재 프로젝트**: 한국어 지원 LLM을 사용한 현대적 재구현 (Ko-CENTaUR)

### 핵심 개념

1. **Feature Extraction**: LLM의 마지막 레이어 hidden state 추출
2. **Binomial Regression**: 추출된 feature로 인간 선택 예측
3. **LOO Cross-Validation**: 100-fold 교차 검증으로 일반화 성능 평가
4. **NLL Metric**: Negative Log-Likelihood로 모델 성능 측정

### 평가 기준

- **Random Baseline**: ~120K NLL
- **LLaMA-65B (원본)**: ~30K NLL
- **목표**: 30K 이하 달성

---

## 환경 설정

### 로컬 개발 환경

#### 1. 저장소 클론

```bash
git clone git@github.com:snuconnectome/CENTaUR.git
cd CENTaUR
```

#### 2. Python 가상환경 생성

```bash
# Python 3.10+ 권장
python3 -m venv venv
source venv/bin/activate  # Linux/Mac
# 또는
venv\Scripts\activate  # Windows
```

#### 3. 의존성 설치

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

**주요 패키지**:
- `torch>=2.0.0` - PyTorch
- `transformers>=4.35.0` - HuggingFace Transformers
- `peft>=0.5.0` - Parameter-Efficient Fine-Tuning
- `accelerate>=0.24.0` - 분산 학습
- `bitsandbytes>=0.41.0` - 양자화 지원

#### 4. CUDA 설정 확인

```bash
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### 서버 환경 (dgx-spark)

#### 1. 서버 접속

```bash
ssh dgx-spark
```

#### 2. 환경 변수 설정

```bash
cd ~/git/CENTaUR
source setup_server_env.sh
```

또는 수동 설정:

```bash
export TMPDIR=/scratch/connectome/connectome1/ko-centaur/tmp
export HF_HOME=/scratch/connectome/connectome1/ko-centaur/cache
export TORCH_HOME=/scratch/connectome/connectome1/ko-centaur/models
```

#### 3. Conda 환경 활성화 (서버)

```bash
source /scratch/connectome/connectome1/miniconda3/bin/activate ko-centaur
```

### GPU 환경 확인

```bash
# GPU 상태 확인
nvidia-smi

# PyTorch CUDA 확인
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, Version: {torch.version.cuda}')"
```

---

## 프로젝트 구조

```
CENTaUR/
├── scripts/                    # 주요 실행 스크립트
│   ├── extract_centaur_features.py    # Feature extraction (핵심)
│   ├── fit_centaur_loo_cv.py          # 100-fold LOO CV
│   ├── submit_extract_*.sh            # SLURM 작업 제출
│   └── submit_fit_*.sh                # SLURM 평가 작업
│
├── ko_centaur/                # Ko-CENTaUR 모듈
│   ├── models/                # 모델 정의
│   ├── data/                  # 데이터 처리
│   ├── training/              # Fine-tuning 코드
│   ├── evaluation/            # 평가 코드
│   └── baselines/             # Baseline 모델
│
├── legacy/                    # 원본 CENTaUR 코드 (참고용)
│   ├── choices13k/            # Risky choice task
│   ├── HorizonTask/           # Sequential decision-making
│   └── ExperientialSymbolicTask/  # Learning task
│
├── outputs/                   # 생성된 feature 파일 (.npz)
├── logs/                      # 실행 로그
├── claudedocs/                # 상세 문서
│   ├── EVALUATION_METHODOLOGY_ANALYSIS.md
│   └── SERVER_DEPLOYMENT_GUIDE.md
│
├── docs/                      # 문서
│   ├── SETUP_GUIDE.md
│   ├── SLURM_GUIDE.md
│   └── KO_CENTAUR_WORKFLOW.md
│
├── requirements.txt           # Python 의존성
├── README.md                  # 프로젝트 개요
├── QUICKSTART.md              # 빠른 시작 가이드
└── ONBOARDING.md              # 이 문서
```

### 주요 디렉토리 설명

- **`scripts/`**: 프로덕션 실행 스크립트
- **`ko_centaur/`**: 모듈화된 Ko-CENTaUR 구현
- **`legacy/`**: 원본 CENTaUR 코드 (참고용, 구식)
- **`outputs/`**: 추출된 feature 저장소
- **`logs/`**: 실행 로그 및 디버깅 정보

---

## 주요 워크플로우

### 전체 파이프라인

```
1. Feature Extraction (GPU 필요)
   └─> LLM 모델 로드
   └─> Hidden state 추출
   └─> outputs/*.npz 저장

2. 100-fold LOO Cross-Validation (CPU 가능)
   └─> Feature 로드
   └─> Binomial Regression 학습
   └─> NLL 계산

3. 결과 분석
   └─> 리포트 생성
   └─> Baseline 비교
```

### 1. Feature Extraction

**목적**: LLM의 마지막 레이어 hidden state를 추출하여 인지 표현으로 사용

**스크립트**: `scripts/extract_centaur_features.py`

```bash
# 로컬 테스트 (10 샘플)
python scripts/extract_centaur_features.py \
    --model qwen25 \
    --n_samples 10 \
    --use_quantization

# 전체 실행 (서버)
./run_qwen25_full.sh
./run_deepseek_full.sh
```

**출력**: `outputs/{model}_features.npz`

### 2. LOO Cross-Validation

**목적**: 100-fold 교차 검증으로 모델 일반화 성능 평가

**스크립트**: `scripts/fit_centaur_loo_cv.py`

```bash
# 로컬 테스트
python scripts/fit_centaur_loo_cv.py \
    --model qwen25 \
    --n_folds 5  # 테스트용

# 전체 실행 (100-fold)
./run_qwen25_loo_cv.sh
./run_deepseek_loo_cv.sh
```

**출력**: NLL 값 및 평가 결과

### 3. 결과 분석

```bash
python generate_report.py
```

---

## 개발 가이드

### 코드 스타일

- **Python**: PEP 8 준수
- **타입 힌트**: 가능한 곳에 사용
- **문서화**: 함수/클래스 docstring 작성

### 주요 모듈

#### 1. Feature Extraction (`scripts/extract_centaur_features.py`)

```python
# 핵심 기능
- 모델 로드 (QLoRA 지원)
- Hidden state 추출
- 양자화 지원 (NF4)
- 배치 처리
```

#### 2. Binomial Regression (`scripts/fit_centaur_loo_cv.py`)

```python
# 핵심 기능
- LOO CV 구현
- Nested CV for regularization
- LBFGS 최적화
- NLL 계산
```

### 테스트

```bash
# 벤치마크 실행
./benchmark_gpu.sh

# 로컬 테스트
python scripts/extract_centaur_features.py --n_samples 10
python scripts/fit_centaur_loo_cv.py --n_folds 5
```

### 디버깅

```bash
# 상세 로깅
python scripts/extract_centaur_features.py --debug

# Feature 진단
python scripts/diagnose_features.py --model qwen25
```

---

## 실험 실행

### 로컬 개발/테스트

```bash
# 1. 벤치마크 (1-2분)
./benchmark_gpu.sh

# 2. Feature extraction 테스트 (10 샘플)
python scripts/extract_centaur_features.py \
    --model qwen25 \
    --n_samples 10 \
    --use_quantization

# 3. LOO CV 테스트 (5-fold)
python scripts/fit_centaur_loo_cv.py \
    --model qwen25 \
    --n_folds 5
```

### 서버 전체 실험 (dgx-spark)

#### tmux 세션 사용 (권장)

```bash
# 세션 시작
tmux new -s centaur_exp

# 실험 실행
cd ~/git/CENTaUR
./run_qwen25_full.sh

# Detach: Ctrl+b, d
# 재접속: tmux attach -t centaur_exp
```

#### 전체 파이프라인

```bash
# Phase 1: Feature Extraction
./run_qwen25_full.sh      # 2-3시간
./run_deepseek_full.sh    # 2-3시간

# Phase 2: LOO CV
./run_qwen25_loo_cv.sh    # 3-4시간
./run_deepseek_loo_cv.sh  # 3-4시간

# Phase 3: 리포트
python generate_report.py
```

**예상 소요 시간**: 약 12-17시간

### SLURM 작업 제출 (서버)

```bash
# Feature extraction
sbatch scripts/submit_extract_qwen25.sh
sbatch scripts/submit_extract_deepseek.sh

# LOO CV
sbatch scripts/submit_fit_qwen25.sh
sbatch scripts/submit_fit_deepseek.sh
```

작업 상태 확인:

```bash
squeue -u $USER
```

---

## 참고 자료

### 프로젝트 문서

- **[README.md](README.md)**: 프로젝트 개요 및 빠른 시작
- **[QUICKSTART.md](QUICKSTART.md)**: dgx-spark 실험 가이드
- **[CLAUDE.md](CLAUDE.md)**: 전체 프로젝트 문서 (원본 CENTaUR 포함)
- **[DGX_EXPERIMENT_GUIDE.md](DGX_EXPERIMENT_GUIDE.md)**: DGX 실험 상세 가이드

### 상세 문서 (`claudedocs/`)

- `EVALUATION_METHODOLOGY_ANALYSIS.md`: 평가 방법론 분석
- `SERVER_DEPLOYMENT_GUIDE.md`: 서버 배포 가이드
- `COMPREHENSIVE_FINDINGS_*.md`: 실험 결과 분석

### 외부 자료

- **원본 논문**: [Binz & Schulz (2023)](https://arxiv.org/abs/2306.03917)
- **Tutorial Paper**: [LLM4BeSci](https://osf.io/preprints/psyarxiv/f7stn)
- **Updated Implementation**: [Zak-Hussain/LLM4BeSci](https://github.com/Zak-Hussain/LLM4BeSci)

### 모델 정보

- **Qwen2.5-32B-Instruct**: QLoRA fine-tuned on Choices13k
- **DeepSeek-R1-Distill-Qwen-32B**: QLoRA fine-tuned on Choices13k

---

## 자주 묻는 질문 (FAQ)

### Q: GPU 메모리 부족 오류가 발생합니다.

A: `--use_quantization` 플래그를 사용하세요. 모든 프로덕션 스크립트는 이미 포함되어 있습니다.

```bash
python scripts/extract_centaur_features.py --use_quantization
```

### Q: 처리가 너무 느립니다.

A: GPU 사용 여부를 확인하세요:

```bash
nvidia-smi
python -c "import torch; print(torch.cuda.is_available())"
```

### Q: tmux 세션이 사라졌습니다.

A: 세션 목록 확인:

```bash
tmux ls
tmux attach -t centaur_exp
```

### Q: Feature extraction이 중간에 멈췄습니다.

A: 체크포인트 기능은 아직 구현되지 않았습니다. 처음부터 재시작해야 합니다.

### Q: 로컬과 서버 환경이 다릅니다.

A: `setup_server_env.sh`를 참고하여 환경 변수를 설정하세요.

---

## 다음 단계

1. ✅ 환경 설정 완료
2. ✅ 프로젝트 구조 이해
3. 🔄 로컬에서 벤치마크 실행
4. 🔄 서버에서 전체 실험 실행
5. 🔄 결과 분석 및 리포트 작성

---

## 문의 및 지원

- **저장소**: https://github.com/snuconnectome/CENTaUR
- **이슈**: GitHub Issues 사용
- **문서**: `docs/` 및 `claudedocs/` 디렉토리 참고

---

**마지막 업데이트**: 2025-11-08

