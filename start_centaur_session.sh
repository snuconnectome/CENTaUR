#!/bin/bash
# CENTaUR tmux 세션 시작 스크립트
# dgx-spark 서버에 연결하여 실험 시작

echo "🚀 CENTaUR 실험 세션 시작"
echo "================================"

# SSH로 서버에 연결하고 tmux 세션 생성/접속
ssh dgx-spark << 'ENDSSH'
# 프로젝트 디렉토리로 이동
cd ~/git/CENTaUR || cd /home/juke/git/CENTaUR

# 기존 centaur 세션이 있는지 확인
if tmux has-session -t centaur 2>/dev/null; then
    echo "✅ 기존 centaur 세션 발견"
    echo "세션에 연결하려면: tmux attach -t centaur"
    echo ""
    echo "현재 세션 목록:"
    tmux ls
else
    echo "📝 새로운 centaur 세션 생성 중..."
    # 새 세션 생성 (detached 상태로)
    tmux new-session -d -s centaur -c "$(pwd)"
    
    # 작업 디렉토리 설정
    tmux send-keys -t centaur "cd ~/git/CENTaUR || cd /home/juke/git/CENTaUR" C-m
    
    # 환경 설정
    tmux send-keys -t centaur "source ~/.bashrc 2>/dev/null || true" C-m
    
    # 현재 상태 확인
    tmux send-keys -t centaur "echo '=== CENTaUR 실험 세션 ==='" C-m
    tmux send-keys -t centaur "pwd" C-m
    tmux send-keys -t centaur "ls -lh outputs/*.npz 2>/dev/null || echo 'Feature 파일 확인 중...'" C-m
    
    echo "✅ centaur 세션 생성 완료"
    echo ""
    echo "세션에 연결하려면: tmux attach -t centaur"
fi

# 세션 정보 출력
echo ""
echo "📊 세션 정보:"
tmux list-sessions | grep centaur || echo "세션을 찾을 수 없습니다"
ENDSSH

echo ""
echo "================================"
echo "✅ 완료!"
echo ""
echo "다음 명령어로 세션에 연결하세요:"
echo "  ssh dgx-spark"
echo "  tmux attach -t centaur"

