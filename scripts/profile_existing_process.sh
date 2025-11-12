#!/bin/bash
# 실행 중인 프로세스에 프로파일링을 연결하는 스크립트

if [ -z "$1" ]; then
    echo "Usage: $0 <PID>"
    echo "Example: $0 12345"
    exit 1
fi

PID=$1

echo "Profiling process $PID..."
echo "Note: This will attach nsys to the running process"

# 프로세스 확인
if ! ps -p $PID > /dev/null 2>&1; then
    echo "Error: Process $PID not found"
    exit 1
fi

# 프로파일링 시작 (attach 모드는 제한적이므로, 새 프로세스로 실행하는 것이 더 나음)
echo "For profiling existing processes, you need to:"
echo "1. Start the process with nsys profile"
echo "2. Or use nvprof for older CUDA versions"
echo ""
echo "Current process info:"
ps -p $PID -o pid,ppid,cmd

