#!/bin/bash
# NGC PyTorch 설치 스크립트 (DGX Spark용)
# NVIDIA 공식 문서 기반

set -e

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

# 2. NGC API 키 확인
echo ""
echo "2. NGC API 키 확인..."
read -p "   NGC API 키를 입력하세요: " NGC_API_KEY

if [ -z "$NGC_API_KEY" ]; then
    echo "   ❌ API 키가 입력되지 않았습니다."
    echo "   NGC 웹사이트에서 API 키를 생성하세요: https://ngc.nvidia.com/"
    exit 1
fi

# 3. NGC 레지스트리 로그인
echo ""
echo "3. NGC 레지스트리 로그인..."
echo "$NGC_API_KEY" | $DOCKER_CMD login nvcr.io -u \$oauthtoken --password-stdin

if [ $? -eq 0 ]; then
    echo "   ✅ NGC 로그인 성공"
else
    echo "   ❌ NGC 로그인 실패"
    exit 1
fi

# 4. PyTorch 컨테이너 다운로드
echo ""
echo "4. PyTorch 컨테이너 다운로드 중..."
echo "   (시간이 걸릴 수 있습니다)"
$DOCKER_CMD pull nvcr.io/nvidia/pytorch:24.08-py3

if [ $? -eq 0 ]; then
    echo "   ✅ 컨테이너 다운로드 완료"
else
    echo "   ❌ 컨테이너 다운로드 실패"
    exit 1
fi

# 5. 설치 확인
echo ""
echo "5. 설치 확인..."
$DOCKER_CMD images | grep pytorch

echo ""
echo "=========================================="
echo "설치 완료!"
echo "=========================================="
echo ""
echo "컨테이너 실행 방법:"
echo "  docker run -it --gpus=all \\"
echo "    -v \$(pwd):/workspace \\"
echo "    nvcr.io/nvidia/pytorch:24.08-py3"
echo ""
echo "PyTorch 버전 확인:"
echo "  docker run --rm --gpus=all \\"
echo "    nvcr.io/nvidia/pytorch:24.08-py3 \\"
echo "    python -c 'import torch; print(torch.__version__)'"

