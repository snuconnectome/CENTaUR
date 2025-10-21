# Ko-CENTaUR SLURM Job Submission Guide

SLURM (Simple Linux Utility for Resource Management) 시스템을 사용한 GPU 학습 가이드

## 클러스터 현황

### 노드 구성
- **node1**: 96 CPUs, 8× RTX GPUs (gpu:rtx:8)
- **node2**: 96 CPUs, GPU 없음
- **node3**: 96 CPUs, 8× GeForce GPUs (gpu:geforce:8)
- **node4**: 96 CPUs, GPU 없음

## SLURM 기본 명령어

### 클러스터 상태 확인
```bash
# 전체 노드 상태
sinfo

# GPU 상세 정보
sinfo -eO "CPUs:8,Memory:9,Gres:20,NodeAIOT:16,NodeList:50"

# 실행중인 작업 확인
squeue

# 내 작업만 확인
squeue -u $USER
```

### 작업 제출
```bash
# 배치 작업 제출
sbatch scripts/train_psych101_slurm.sh

# 제출 후 작업 ID 확인 (예: Submitted batch job 62390)
```

### 작업 모니터링
```bash
# 작업 상태 확인
squeue -j <JOB_ID>

# 실시간 로그 확인
tail -f /scratch/connectome/connectome1/ko-centaur/logs/slurm-<JOB_ID>.out

# 에러 로그 확인
tail -f /scratch/connectome/connectome1/ko-centaur/logs/slurm-<JOB_ID>.err
```

### 작업 제어
```bash
# 작업 취소
scancel <JOB_ID>

# 모든 내 작업 취소
scancel -u $USER

# 작업 홀드 (일시정지)
scontrol hold <JOB_ID>

# 작업 릴리즈 (재개)
scontrol release <JOB_ID>
```

## Ko-CENTaUR 학습 작업 제출

### 1. 서버 접속
```bash
ssh server
```

### 2. 데이터 확인
```bash
ls -lh /scratch/connectome/connectome1/ko-centaur/data/psych101_exaone_train.jsonl
```

### 3. 작업 스크립트 복사
```bash
# 로컬에서 서버로 복사
scp /Users/jiookcha/Documents/git/CENTaUR/scripts/train_psych101_slurm.sh server:/scratch/connectome/connectome1/ko-centaur/
scp /Users/jiookcha/Documents/git/CENTaUR/ko_centaur/training/train_psych101_full_slurm.py server:/scratch/connectome/connectome1/ko-centaur/
```

### 4. 작업 제출
```bash
cd /scratch/connectome/connectome1/ko-centaur
sbatch train_psych101_slurm.sh
```

출력 예시:
```
Submitted batch job 62390
```

### 5. 작업 모니터링
```bash
# 작업 상태
squeue -j 62390

# 실시간 로그
tail -f logs/slurm-62390.out

# GPU 사용량 확인 (작업이 실행중인 노드에서)
srun --jobid=62390 nvidia-smi
```

## 작업 스크립트 설정 (train_psych101_slurm.sh)

### SLURM 파라미터 설명
```bash
#SBATCH --job-name=ko-centaur-train    # 작업 이름
#SBATCH --partition=debug              # 파티션 (큐) 이름
#SBATCH --nodes=1                      # 노드 개수
#SBATCH --ntasks=1                     # 태스크 개수
#SBATCH --cpus-per-task=8              # 태스크당 CPU 코어 수
#SBATCH --gres=gpu:rtx:1               # GPU 요청 (rtx GPU 1개)
#SBATCH --mem=64G                      # 메모리
#SBATCH --time=12:00:00                # 최대 실행 시간 (12시간)
#SBATCH --output=logs/slurm-%j.out     # 표준 출력 파일
#SBATCH --error=logs/slurm-%j.err      # 에러 출력 파일
```

### GPU 종류 선택
- **RTX GPU (node1)**: `--gres=gpu:rtx:1`
- **GeForce GPU (node3)**: `--gres=gpu:geforce:1`
- **여러 GPU**: `--gres=gpu:rtx:2` (2개)

### 메모리 설정
- **최소 권장**: 64GB (현재 설정)
- **여유 있게**: 128GB
- **최대**: 각 노드 514GB까지 가능

## 학습 스크립트 설정 (train_psych101_full_slurm.py)

### 주요 변경 사항
1. **CUDA_VISIBLE_DEVICES 제거**: SLURM이 자동 설정
2. **device_map 변경**: `"auto"` → `{"": 0}` (단일 GPU 강제)
3. **메모리 최적화**:
   - BATCH_SIZE: 4 → 2
   - GRADIENT_ACCUMULATION_STEPS: 4 → 8 (효과적 배치 크기 유지)
   - MAX_SEQ_LENGTH: 2048 → 1024

### 예상 학습 시간
- **60,092 샘플, 3 에폭**
- **RTX 3090 (24GB) 기준**: 4-6 시간
- **총 스텝**: ~11,268 steps

### 체크포인트 저장
- **저장 주기**: 500 steps마다
- **저장 위치**: `/scratch/connectome/connectome1/ko-centaur/models/exaone-psych101-full/`
- **최대 저장 수**: 3개 (오래된 것 자동 삭제)

## 트러블슈팅

### 문제 1: 작업이 대기 상태 (PD)
```bash
squeue -j <JOB_ID>
# STATE가 PD (Pending)인 경우
```

**원인**:
- 모든 GPU가 사용중
- 요청한 리소스가 너무 큼
- 파티션 제한

**해결**:
```bash
# 대기 이유 확인
squeue -j <JOB_ID> --start

# GPU 가용성 확인
sinfo -eO "Gres:20,GresUsed:24,NodeList:50"
```

### 문제 2: OOM (Out of Memory) 에러
```bash
# 로그에서 "CUDA out of memory" 확인
tail -100 logs/slurm-<JOB_ID>.err
```

**해결 방법**:
1. **배치 크기 감소**: BATCH_SIZE를 1로 줄임
2. **시퀀스 길이 감소**: MAX_SEQ_LENGTH를 512로 줄임
3. **메모리 요청 증가**: `--mem=128G`
4. **GPU 개수 증가**: `--gres=gpu:rtx:2` (멀티 GPU)

### 문제 3: 작업이 시작하지 않음
```bash
# 작업 상세 정보
scontrol show job <JOB_ID>
```

**확인 사항**:
- Conda 환경 활성화 경로
- 데이터 파일 존재 여부
- 스크립트 실행 권한

### 문제 4: 학습 중단
```bash
# 마지막 체크포인트에서 재개
# train_psych101_full_slurm.py에 --resume_from_checkpoint 옵션 추가 필요
```

## 작업 완료 후 확인

### 1. 최종 모델 확인
```bash
ls -lh /scratch/connectome/connectome1/ko-centaur/models/exaone-psych101-full/
```

**필수 파일**:
- `adapter_model.bin` - LoRA 어댑터 가중치
- `adapter_config.json` - LoRA 설정
- `training_metrics.json` - 학습 메트릭

### 2. 학습 로그 분석
```bash
# 최종 loss 확인
grep "Training Complete" logs/slurm-<JOB_ID>.out -A 10

# 학습 시간 확인
grep "Total time:" logs/slurm-<JOB_ID>.out
```

### 3. GPU 사용률 확인
```bash
# 작업이 실행된 노드에서
grep "GPU" logs/slurm-<JOB_ID>.out
```

## 고급 사용

### 인터랙티브 세션 (디버깅용)
```bash
# GPU 1개 할당받아 인터랙티브 세션 시작
srun --partition=debug --gres=gpu:rtx:1 --cpus-per-task=8 --mem=64G --pty bash

# 세션 내에서 Python 스크립트 직접 실행 가능
python ko_centaur/training/train_psych101_full_slurm.py
```

### 배열 작업 (여러 실험 동시 실행)
```bash
# 5개의 다른 랜덤 시드로 학습
#SBATCH --array=1-5

# 스크립트 내에서 $SLURM_ARRAY_TASK_ID 사용
```

### 이메일 알림
```bash
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=your.email@domain.com
```

## 모범 사례

1. **작업 제출 전 확인**:
   ```bash
   # 데이터 존재 확인
   ls -lh /scratch/connectome/connectome1/ko-centaur/data/

   # GPU 가용성 확인
   sinfo -eO "Gres:20,GresUsed:24,NodeList:50"

   # 스크립트 문법 확인
   bash -n scripts/train_psych101_slurm.sh
   ```

2. **리소스 요청 최적화**:
   - 필요한 만큼만 요청 (다른 사용자 고려)
   - 시간 제한 보수적으로 설정 (12시간 → 8시간으로 충분할 수 있음)
   - GPU는 최소한으로 (1개면 충분)

3. **로그 관리**:
   ```bash
   # 오래된 로그 정리
   find logs/ -name "slurm-*.out" -mtime +30 -delete
   ```

4. **체크포인트 전략**:
   - SAVE_STEPS=500 (기본값, 약 30분마다)
   - 중단 시 재개 가능하도록 설정

## 참고 자료

- SLURM 공식 문서: https://slurm.schedmd.com/
- SLURM sbatch: https://slurm.schedmd.com/sbatch.html
- SLURM srun: https://slurm.schedmd.com/srun.html
- GPU 할당: https://slurm.schedmd.com/gres.html

## 빠른 참조 카드

```bash
# 작업 제출
sbatch train_psych101_slurm.sh

# 작업 상태
squeue -u $USER

# 작업 취소
scancel <JOB_ID>

# 로그 확인
tail -f logs/slurm-<JOB_ID>.out

# GPU 상태
sinfo -eO "Gres:20,GresUsed:24,NodeList:50"

# 인터랙티브 세션
srun --partition=debug --gres=gpu:rtx:1 --mem=64G --pty bash
```
