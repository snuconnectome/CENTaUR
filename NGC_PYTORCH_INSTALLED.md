# NGC PyTorch 설치 완료 보고서

## 설치 정보

- **설치 날짜**: 2025-11-09
- **컨테이너**: nvcr.io/nvidia/pytorch:24.08-py3
- **설치 방법**: Docker 컨테이너
- **상태**: ✅ 설치 완료

## 설치 확인

컨테이너가 성공적으로 다운로드되었습니다.

## 사용 방법

### 기본 실행
```bash
echo "462773" | sudo -S docker run -it --gpus=all \
  nvcr.io/nvidia/pytorch:24.08-py3
```

### 작업 디렉토리 마운트
```bash
echo "462773" | sudo -S docker run -it --gpus=all \
  -v /home/juke/git/CENTaUR:/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:24.08-py3
```

### CENTaUR 프로젝트 실행
```bash
# EXAONE-3.5-32B Feature Extraction
echo "462773" | sudo -S docker run -it --gpus=all \
  -v /home/juke/git/CENTaUR:/workspace \
  -w /workspace \
  nvcr.io/nvidia/pytorch:24.08-py3 \
  bash -c "cd /workspace && python scripts/extract_centaur_features.py --model exaone35-base --dataset ko_centaur/data/choices13k_100.jsonl --output outputs/exaone35_features.npz"
```

## PyTorch 버전 확인

```bash
echo "462773" | sudo -S docker run --rm --gpus=all \
  nvcr.io/nvidia/pytorch:24.08-py3 \
  python -c "import torch; print(torch.__version__)"
```

## 다음 단계

1. ✅ NGC PyTorch 컨테이너 설치 완료
2. EXAONE-3.5-32B Feature Extraction을 컨테이너에서 실행
3. GPU 호환성 테스트
4. LOO CV 작업 실행

---

**설치 완료**: 2025-11-09
