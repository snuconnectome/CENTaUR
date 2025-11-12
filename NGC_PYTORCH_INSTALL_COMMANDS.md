# NGC PyTorch 설치 명령어 (복사해서 실행)

## 빠른 설치

터미널에서 다음 명령어를 실행하세요:

```bash
# 1. NGC 로그인
echo "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" | \
  sudo docker login nvcr.io -u '$oauthtoken' --password-stdin

# 2. 컨테이너 다운로드 (시간이 걸릴 수 있음)
sudo docker pull nvcr.io/nvidia/pytorch:24.08-py3

# 3. 설치 확인
sudo docker images | grep pytorch

# 4. PyTorch 버전 테스트
sudo docker run --rm --gpus=all \
  nvcr.io/nvidia/pytorch:24.08-py3 \
  python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

## 컨테이너 실행

```bash
# 기본 실행
sudo docker run -it --gpus=all nvcr.io/nvidia/pytorch:24.08-py3

# 작업 디렉토리 마운트
sudo docker run -it --gpus=all \
  -v /home/juke/git/CENTaUR:/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:24.08-py3
```

## CENTaUR 프로젝트 실행

```bash
# 컨테이너에서 CENTaUR 실행
sudo docker run -it --gpus=all \
  -v /home/juke/git/CENTaUR:/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:24.08-py3 \
  bash -c "cd /workspace && python scripts/extract_centaur_features.py --model exaone35-base ..."
```

---

**API 키**: 포함됨 ✅
**작성일**: 2025-11-09

