# CENTaUR - NGC PyTorch 사용 가이드

## ✅ 설정 완료!

NGC PyTorch 환경이 성공적으로 설정되었습니다.

## 🚀 빠른 시작

### 훈련 실행

```bash
# Qwen2.5-32B QLoRA 훈련 시작
./scripts/train_qwen25_ngc.sh

# 백그라운드로 실행
nohup ./scripts/train_qwen25_ngc.sh > logs/training_$(date +%Y%m%d_%H%M%S).log 2>&1 &
```

### Python 스크립트 실행

```bash
# NGC 컨테이너에서 Python 실행
./ngc-python script.py

# 또는 환경 로드 후
source activate-ngc.sh
centaur-python script.py
```

### Interactive Shell

```bash
./ngc-python-interactive
# 또는
centaur-shell
```

## 📁 환경 구조

```
CENTaUR/
├── ngc-python              # Python wrapper (NGC)
├── ngc-python-interactive  # Interactive shell
├── activate-ngc.sh         # 환경 빠른 활성화
├── venv.backup/            # 백업된 native venv
├── scripts/
│   ├── train_qwen25_ngc.sh          # NGC 훈련 스크립트
│   └── setup_ngc_environment.sh     # 환경 설정
└── .bashrc_centaur         # NGC 환경 설정
```

## 🔧 사용 가능한 명령어

새 터미널에서 자동으로 로드됩니다 (~/.bashrc에 추가됨):

```bash
centaur-python script.py    # NGC에서 Python 실행
centaur-shell               # Interactive NGC shell
centaur-run script.py       # 로깅과 함께 실행
cdcentaur                   # 프로젝트 디렉토리로 이동
```

## ⚡ 환경 정보

- **NGC PyTorch**: 2.5.0a0+nv24.08
- **CUDA**: 12.6 (사용 가능)
- **GPU**: NVIDIA GB10 (작동 확인)
- **Container**: nvcr.io/nvidia/pytorch:24.08-py3

## 📝 다음 단계

1. ✅ NGC 환경 설정 완료
2. ✅ Native venv 백업 완료 (venv.backup)
3. ⏳ **훈련 실행** ← 다음
4. ⏳ 특징 추출
5. ⏳ LOO Cross-Validation

## ⚠️ 중요: 훈련 스크립트 필수 요구사항

### GPU 초기화 코드 반드시 포함!

**모든 QLoRA 훈련 스크립트는 main() 함수 시작 부분에 GPU 초기화 코드를 포함해야 합니다.**

이 코드가 없으면 다음 오류가 발생합니다:
```
TypeError: device() received an invalid combination of arguments - got (NoneType)
```

**필수 템플릿:**
```python
import os
import torch

def main():
    # GPU 초기화 (필수!)
    print("\n[0/6] GPU 사용 확인 및 강제 설정...")
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA를 사용할 수 없습니다!")

    num_gpus = torch.cuda.device_count()
    print(f"✅ CUDA 사용 가능: {num_gpus}개 GPU 감지")

    device = torch.device("cuda:0")
    print(f"   기본 디바이스: {device}")
    print("✅ GPU 사용 준비 완료")

    # 이후 tokenizer, 모델 로딩...
```

자세한 내용은 `CLAUDE.md`의 "Critical Implementation Requirements" 섹션 참조.

## 🔍 문제 해결

### 새 터미널에서 명령어가 없다면

```bash
source ~/.bashrc
# 또는
source activate-ngc.sh
```

### Native venv 복구

```bash
mv venv.backup venv
```

### NGC 컨테이너 재시작

```bash
echo "462773" | sudo -S docker pull nvcr.io/nvidia/pytorch:24.08-py3
```

## 📚 상세 가이드

더 자세한 내용은 `NGC_SETUP_GUIDE.md`를 참고하세요.

---

**준비 완료!** 이제 훈련을 시작하세요:

```bash
./scripts/train_qwen25_ngc.sh
```
