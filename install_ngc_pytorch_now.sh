#!/bin/bash
# NGC PyTorch 설치 스크립트 (수동 실행용)
# 사용자가 직접 터미널에서 실행해야 합니다

set -e

NGC_API_KEY="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"

echo "=========================================="
echo "NGC PyTorch 설치 스크립트"
echo "=========================================="
echo ""

# 1. Docker 권한 확인
echo "1. Docker 권한 확인..."
DOCKER_CMD="docker"
if docker ps &>/dev/null; then
    echo "   ✅ Docker 권한 OK"
elif sudo docker ps &>/dev/null; then
    echo "   ⚠️  sudo 필요 (Docker 권한 없음)"
    DOCKER_CMD="sudo docker"
    echo "   sudo를 사용하여 진행합니다"
else
    echo "   ❌ Docker 접근 불가"
    echo "   해결 방법:"
    echo "   sudo usermod -aG docker $USER"
    echo "   (로그아웃 후 재로그인 필요)"
    exit 1
fi

# 2. NGC 레지스트리 로그인
echo ""
echo "2. NGC 레지스트리 로그인..."
echo "$NGC_API_KEY" | $DOCKER_CMD login nvcr.io -u '$oauthtoken' --password-stdin

if [ $? -eq 0 ]; then
    echo "   ✅ NGC 로그인 성공"
else
    echo "   ❌ NGC 로그인 실패"
    exit 1
fi

# 3. PyTorch 컨테이너 다운로드
echo ""
echo "3. PyTorch 컨테이너 다운로드 중..."
echo "   (시간이 걸릴 수 있습니다)"
$DOCKER_CMD pull nvcr.io/nvidia/pytorch:24.08-py3

if [ $? -eq 0 ]; then
    echo "   ✅ 컨테이너 다운로드 완료"
else
    echo "   ❌ 컨테이너 다운로드 실패"
    exit 1
fi

# 4. 설치 확인
echo ""
echo "4. 설치 확인..."
$DOCKER_CMD images | grep pytorch

# 5. PyTorch 버전 테스트
echo ""
echo "5. PyTorch 버전 테스트..."
$DOCKER_CMD run --rm --gpus=all \
  nvcr.io/nvidia/pytorch:24.08-py3 \
  python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"

echo ""
echo "=========================================="
echo "설치 완료!"
echo "=========================================="
echo ""
echo "컨테이너 실행 방법:"
echo "  $DOCKER_CMD run -it --gpus=all \\"
echo "    -v \$(pwd):/workspace \\"
echo "    nvcr.io/nvidia/pytorch:24.08-py3"
echo ""

