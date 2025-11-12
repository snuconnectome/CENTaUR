#!/bin/bash
# dgx-spark 서버에서 실행할 스크립트
# tmux centaur 세션 생성 및 실험 시작

echo "🚀 CENTaUR 실험 세션 설정"
echo "================================"

# 프로젝트 디렉토리로 이동
cd ~/git/CENTaUR 2>/dev/null || cd /home/juke/git/CENTaUR 2>/dev/null || {
    echo "❌ 프로젝트 디렉토리를 찾을 수 없습니다"
    exit 1
}

echo "📁 작업 디렉토리: $(pwd)"

# 기존 centaur 세션 확인
if tmux has-session -t centaur 2>/dev/null; then
    echo "✅ 기존 centaur 세션 발견"
    echo ""
    echo "현재 세션 목록:"
    tmux ls
    echo ""
    echo "기존 세션에 연결하려면:"
    echo "  tmux attach -t centaur"
    echo ""
    read -p "기존 세션을 사용하시겠습니까? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "새 세션을 생성합니다..."
        tmux kill-session -t centaur 2>/dev/null
    else
        echo "기존 세션에 연결합니다..."
        tmux attach -t centaur
        exit 0
    fi
fi

# 새 세션 생성
echo "📝 새로운 centaur 세션 생성 중..."
tmux new-session -d -s centaur -c "$(pwd)"

# 환경 설정
tmux send-keys -t centaur "cd $(pwd)" C-m
tmux send-keys -t centaur "source ~/.bashrc 2>/dev/null || true" C-m

# 현재 상태 확인
tmux send-keys -t centaur "clear" C-m
tmux send-keys -t centaur "echo '=== CENTaUR 실험 세션 ==='" C-m
tmux send-keys -t centaur "echo '작업 디렉토리: $(pwd)'" C-m
tmux send-keys -t centaur "echo ''" C-m

# Feature 파일 확인
tmux send-keys -t centaur "echo '📊 Feature 파일 확인:'" C-m
tmux send-keys -t centaur "ls -lh outputs/*.npz 2>/dev/null || echo '  Feature 파일 없음'" C-m
tmux send-keys -t centaur "echo ''" C-m

# 실험 계획 안내
tmux send-keys -t centaur "echo '🎯 다음 단계 (우선순위):'" C-m
tmux send-keys -t centaur "echo '  1. Qwen2.5-32B Base LOO CV (Feature 있음)'" C-m
tmux send-keys -t centaur "echo '  2. DeepSeek-R1 Base LOO CV (Feature 있음)'" C-m
tmux send-keys -t centaur "echo '  3. EXAONE-3.5-32B Feature Extraction'" C-m
tmux send-keys -t centaur "echo ''" C-m

# 명령어 힌트
tmux send-keys -t centaur "echo '💡 실행 명령어 예시:'" C-m
tmux send-keys -t centaur "echo '  python scripts/fit_centaur_loo_cv_flexible.py \\'" C-m
tmux send-keys -t centaur "echo '      outputs/qwen25_base_features.npz \\'" C-m
tmux send-keys -t centaur "echo '      outputs/qwen25_base_nll_results.json \\'" C-m
tmux send-keys -t centaur "echo '      --model_name \"Qwen2.5-32B-Base\"'" C-m
tmux send-keys -t centaur "echo ''" C-m

echo "✅ centaur 세션 생성 완료!"
echo ""
echo "세션에 연결하려면:"
echo "  tmux attach -t centaur"
echo ""
echo "또는 바로 연결:"
read -p "지금 연결하시겠습니까? (y/n) " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    tmux attach -t centaur
else
    echo "나중에 연결: tmux attach -t centaur"
fi

