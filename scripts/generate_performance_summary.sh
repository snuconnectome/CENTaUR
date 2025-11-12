#!/bin/bash
# GPU 성능 요약 리포트 생성

cd ~/git/CENTaUR

echo "=========================================="
echo "GPU Performance Summary Report"
echo "=========================================="
echo "Generated: $(date)"
echo ""

# 현재 실행 중인 작업 확인
echo "=== Currently Running Jobs ==="
ps aux | grep -E 'extract_centaur|fit_centaur_loo_cv' | grep -v grep | awk '{printf "  PID %-8s CPU %5.1f%% MEM %5.1f%% %s\n", $2, $3, $4, substr($0, index($0,$11))}'
echo ""

# GPU 상태
echo "=== GPU Status ==="
nvidia-smi --query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu --format=csv,noheader,nounits | \
    awk -F', ' '{printf "  GPU %s: %s\n", $1, $2; printf "    Utilization: GPU %s%%, Memory %s%%\n", $3, $4; printf "    Memory: %s MB / %s MB\n", $5, $6; printf "    Power: %s W, Temp: %s°C\n\n", $7, $8}'
echo ""

# 프로파일 파일 확인
echo "=== Profile Files ==="
if [ -d profiles ]; then
    ls -lh profiles/*.nsys-rep 2>/dev/null | awk '{printf "  %s (%s)\n", $9, $5}'
    echo ""
    
    # 각 프로파일 분석
    for profile in profiles/*.nsys-rep; do
        if [ -f "$profile" ]; then
            echo "--- $(basename $profile) ---"
            /usr/local/bin/nsys stats --report cuda_api_sum --force-export=true "$profile" 2>&1 | \
                grep -A 10 "CUDA API Summary" | head -8
            echo ""
        fi
    done
else
    echo "  No profiles directory found"
fi

# 결과 파일 확인
echo "=== Output Files ==="
if [ -d outputs ]; then
    echo "Feature files:"
    ls -lh outputs/*features*.npz 2>/dev/null | awk '{printf "  %s (%s, %s)\n", $9, $5, $6" "$7" "$8}'
    echo ""
    echo "Result files:"
    ls -lh outputs/*results*.json 2>/dev/null | awk '{printf "  %s (%s, %s)\n", $9, $5, $6" "$7" "$8}'
fi

echo ""
echo "=========================================="

