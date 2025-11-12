# NGC PyTorch 환경 설정 가이드

## 개요

CENTaUR 프로젝트는 이제 **NGC PyTorch를 기본 환경으로 사용**합니다.
GB10 GPU 호환성과 최적화된 성능을 위해 NVIDIA NGC 컨테이너를 사용합니다.

## 현재 상태

✅ **NGC PyTorch 24.08 설치 완료**
- Docker 이미지: `nvcr.io/nvidia/pytorch:24.08-py3`
- PyTorch: 2.5.0a0+nv24.08
- CUDA: 12.6
- GB10 GPU: 작동 확인

## 빠른 시작

### 1단계: 환경 설정 (한 번만 실행)

```bash
./scripts/setup_ngc_environment.sh
```

이 스크립트는:
- `~/.bashrc`에 NGC 환경 자동 로드 추가
- Native venv 처리 (rename/delete/keep)
- dgx-venv 처리 (CPU 전용이므로 삭제 권장)
- 편리한 alias 및 wrapper 스크립트 생성

### 2단계: Shell 재로드

```bash
source ~/.bashrc
# 또는
source activate-ngc.sh  # 빠른 활성화
```

### 3단계: 테스트

```bash
centaur-python --version
# 또는
centaur-shell  # Interactive shell
```

## 사용 방법

### Python 스크립트 실행

```bash
# NGC 컨테이너에서 Python 실행
centaur-python script.py --args

# 또는 wrapper 직접 사용
./ngc-python script.py --args
```

### 훈련 시작

```bash
# Qwen2.5-32B QLoRA 훈련
./scripts/train_qwen25_ngc.sh

# 백그라운드 실행
nohup ./scripts/train_qwen25_ngc.sh > logs/training.log 2>&1 &
```

### Interactive Shell

```bash
centaur-shell
# 컨테이너 안에서 작업
cd /workspace
python ko_centaur/training/train_qwen25_32b_qlora.py --config ...
```

## 생성된 파일

### Wrapper Scripts

- **`ngc-python`**: Python 스크립트를 NGC 컨테이너에서 실행
- **`ngc-python-interactive`**: Interactive NGC shell
- **`activate-ngc.sh`**: NGC 환경 빠른 활성화

### Training Scripts

- **`scripts/train_qwen25_ngc.sh`**: Qwen2.5-32B 훈련 (NGC 사용)
- **`scripts/train_qwen25_ngc_docker.sh`**: Docker 명령어 직접 사용 (동일 기능)

### Configuration

- **`.bashrc_centaur`**: NGC 환경 설정 (자동 로드됨)

## Shell Aliases

설정 후 사용 가능한 명령어:

```bash
centaur-python script.py    # NGC에서 Python 실행
centaur-shell               # Interactive NGC shell
centaur-run script.py       # 로깅과 함께 실행
cdcentaur                   # 프로젝트 디렉토리로 이동
```

## Native venv vs NGC PyTorch

| 환경 | PyTorch | CUDA | GB10 지원 | 사용 |
|------|---------|------|-----------|------|
| **venv** (native) | 2.5.1 | 12.4 | ❌ | 사용 중지 |
| **dgx-venv** | 2.9.0+cpu | ❌ | ❌ | 삭제 권장 |
| **NGC Docker** ⭐ | 2.5.0a0+nv | 12.6 | ✅ | **기본 사용** |

### venv 처리 옵션

1. **Rename** (권장): `venv` → `venv.backup`
   - 안전, 필요시 복구 가능
   - 디스크 공간 사용

2. **Delete**: `venv` 완전 삭제
   - 디스크 공간 확보
   - 복구 불가

3. **Keep**: 두 환경 유지
   - 수동으로 선택 사용
   - 실수로 native 사용 가능성

## 문제 해결

### NGC 컨테이너 재설치

```bash
echo "462773" | sudo -S docker pull nvcr.io/nvidia/pytorch:24.08-py3
```

### 권한 문제

```bash
# Docker 권한 추가
sudo usermod -aG docker $USER
# 로그아웃 후 재로그인
```

### GB10 경고 무시

```
WARNING: Detected NVIDIA GB10 GPU, which is not yet supported
```

이 경고는 무시해도 됩니다. **실제로는 작동합니다.**

### 패키지 설치

NGC 컨테이너에서 추가 패키지 필요 시:

```bash
centaur-shell
pip install package_name
```

또는 스크립트에서:

```bash
./ngc-python -m pip install package_name
```

## 훈련 체크리스트

- [ ] NGC 환경 설정 완료
- [ ] Shell 재로드 (`source ~/.bashrc`)
- [ ] GPU 확인 (`nvidia-smi`)
- [ ] NGC 테스트 (`centaur-python --version`)
- [ ] 훈련 시작 (`./scripts/train_qwen25_ngc.sh`)

## 다음 단계

1. ✅ NGC PyTorch 설치
2. ✅ 환경 설정
3. ✅ Wrapper scripts 생성
4. ⏳ **훈련 실행** ← 현재
5. ⏳ 특징 추출 (Feature extraction)
6. ⏳ LOO Cross-Validation

---

**마지막 업데이트**: 2025-11-09
**NGC PyTorch 버전**: 24.08
