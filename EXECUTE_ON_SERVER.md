# 서버에서 즉시 실행하기

SSH 연결 문제로 로컬에서 자동 실행이 불가능합니다.
터미널을 열어서 직접 서버에 접속하여 실행해주세요.

---

## 방법 1: 자동 실행 스크립트 (권장)

**터미널을 열고 다음 명령어를 입력하세요**:

```bash
# 1. 서버 접속
ssh server

# 2. 프로젝트 디렉토리로 이동
cd /scratch/connectome/connectome1/ko-centaur

# 3. 최신 코드 받기
git pull origin feature/ko-centaur-llm-strategy

# 4. 자동 실행 스크립트 실행 (완료까지 4-8시간 소요)
bash RUN_NOW.sh
```

**이 스크립트가 자동으로**:
- ✅ 환경 설정 확인
- ✅ Qwen2.5 feature extraction 실행
- ✅ DeepSeek feature extraction 실행
- ✅ Qwen2.5 LOO CV 실행
- ✅ DeepSeek LOO CV 실행
- ✅ 최종 결과 출력

---

## 방법 2: 수동 단계별 실행

자동 스크립트가 문제 발생 시, 수동으로 단계별 실행:

### 1단계: 준비

```bash
ssh server
cd /scratch/connectome/connectome1/ko-centaur
git pull origin feature/ko-centaur-llm-strategy

source ~/.bashrc
conda activate centaur

mkdir -p data/features data/results logs
```

### 2단계: Feature Extraction

```bash
# Qwen2.5
sbatch scripts/submit_extract_qwen25.sh

# DeepSeek
sbatch scripts/submit_extract_deepseek.sh

# 작업 확인
squeue -u $USER

# 진행상황 모니터링 (Ctrl+C로 종료)
tail -f logs/extract_qwen25_*.out
```

**대기**: 각 작업이 완료될 때까지 1-2시간 (총 2-4시간)

### 3단계: Feature 검증

```bash
python -c "
import torch
qwen = torch.load('data/features/centaur_features_qwen25.pth')
deep = torch.load('data/features/centaur_features_deepseek.pth')
print(f'✅ Qwen2.5: {qwen[\"features\"].shape}')
print(f'✅ DeepSeek: {deep[\"features\"].shape}')
"
```

**예상 출력**:
```
✅ Qwen2.5: torch.Size([100, 3072])
✅ DeepSeek: torch.Size([100, 3072])
```

### 4단계: LOO Cross-Validation

```bash
# Qwen2.5
sbatch scripts/submit_fit_qwen25.sh

# DeepSeek
sbatch scripts/submit_fit_deepseek.sh

# 진행상황 모니터링
tail -f logs/fit_qwen25_*.out
```

**대기**: 각 작업이 완료될 때까지 1-2시간 (총 2-4시간)

### 5단계: 결과 확인

```bash
python -c "
import torch

qwen = torch.load('data/results/loo_cv_results_qwen25.pth')
deep = torch.load('data/results/loo_cv_results_deepseek.pth')

print('='*70)
print('Ko-CENTaUR RESULTS')
print('='*70)
print(f'\nQwen2.5-32B: {qwen[\"total_nll\"]:.0f} NLL')
print(f'DeepSeek-R1: {deep[\"total_nll\"]:.0f} NLL')
print(f'\nBenchmark:')
print(f'  Random: ~120,000 NLL')
print(f'  LLaMA-65B: ~30,000 NLL')
print('='*70)
"
```

---

## 방법 3: 백그라운드 실행 (터미널 종료해도 계속 실행)

```bash
# 서버 접속
ssh server
cd /scratch/connectome/connectome1/ko-centaur
git pull origin feature/ko-centaur-llm-strategy

# nohup으로 백그라운드 실행
nohup bash RUN_NOW.sh > pipeline_output.log 2>&1 &

# 프로세스 ID 확인
echo $!

# 터미널 종료 가능 (작업은 계속 진행됨)
exit

# 나중에 다시 접속하여 로그 확인
ssh server
cd /scratch/connectome/connectome1/ko-centaur
tail -f pipeline_output.log
```

---

## 모니터링 명령어

**작업 상태 확인**:
```bash
squeue -u $USER
```

**GPU 사용률 확인** (feature extraction 중):
```bash
watch -n 5 nvidia-smi
```

**로그 실시간 확인**:
```bash
tail -f logs/extract_qwen25_*.out
tail -f logs/fit_qwen25_*.out
```

**에러 확인**:
```bash
cat logs/extract_qwen25_*.err
cat logs/fit_qwen25_*.err
```

---

## 예상 소요 시간

| 단계 | 시간 |
|-----|------|
| Qwen2.5 Feature Extraction | 1-2시간 |
| DeepSeek Feature Extraction | 1-2시간 |
| Qwen2.5 LOO CV | 1-2시간 |
| DeepSeek LOO CV | 1-2시간 |
| **전체** | **4-8시간** |

---

## 문제 해결

**Q: "Permission denied" 에러**
```bash
# 스크립트 실행 권한 부여
chmod +x RUN_NOW.sh
chmod +x scripts/*.sh scripts/*.py
```

**Q: "conda: command not found"**
```bash
# bashrc 다시 로드
source ~/.bashrc
conda activate centaur
```

**Q: "GPU out of memory"**
```bash
# GPU 상태 확인
nvidia-smi

# 다른 프로세스가 GPU 사용 중이면 대기하거나
# 작은 샘플로 테스트
python scripts/extract_centaur_features.py --model qwen25 --n_samples 10
```

**Q: 작업이 너무 오래 걸림**
```bash
# 정상입니다! Feature extraction은 각 1-2시간 소요
# 진행상황 확인:
tail -n 50 logs/extract_qwen25_*.out
```

---

## 성공 확인

다음 4개 파일이 생성되면 성공:

```
✅ data/features/centaur_features_qwen25.pth
✅ data/features/centaur_features_deepseek.pth
✅ data/results/loo_cv_results_qwen25.pth
✅ data/results/loo_cv_results_deepseek.pth
```

---

**지금 바로 터미널을 열고 실행하세요!** 🚀
