#!/bin/bash
# CENTaUR 실험 모니터링 및 완료 알림 스크립트
# 10분마다 체크하고, 작업 완료 시 이메일 발송

EMAIL="cha.jiook@gmail.com"
CHECK_INTERVAL=600  # 10분 = 600초
LOG_FILE="logs/monitor.log"
STATUS_FILE="logs/experiment_status.json"

# 작업 목록
declare -A EXPERIMENTS=(
    ["EXAONE-3.5-32B"]="extract_centaur_features.*exaone35"
    ["Qwen2.5-Base-LOO-CV"]="fit_centaur.*qwen25_base"
    ["DeepSeek-Base-LOO-CV"]="fit_centaur.*deepseek_base"
)

# 결과 파일 목록
declare -A RESULT_FILES=(
    ["EXAONE-3.5-32B"]="outputs/exaone35_features.npz"
    ["Qwen2.5-Base-LOO-CV"]="outputs/qwen25_base_nll_results.json"
    ["DeepSeek-Base-LOO-CV"]="outputs/deepseek_base_nll_results.json"
)

log_message() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"
}

check_process_running() {
    local exp_name=$1
    local pattern=$2
    ps aux | grep -E "$pattern" | grep -v grep | wc -l
}

check_result_exists() {
    local exp_name=$1
    local result_file=$2
    if [ -f "$result_file" ]; then
        echo "1"
    else
        echo "0"
    fi
}

send_email() {
    local subject=$1
    local body=$2
    
    # mail 명령어 사용 (시스템에 mailutils 설치 필요)
    if command -v mail &> /dev/null; then
        echo "$body" | mail -s "$subject" "$EMAIL"
        log_message "이메일 발송 완료: $subject"
    else
        # mail 명령어가 없으면 Python으로 시도
        python3 << EOF
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sys

try:
    msg = MIMEMultipart()
    msg['From'] = 'centaur@server'
    msg['To'] = '$EMAIL'
    msg['Subject'] = '$subject'
    msg.attach(MIMEText('$body', 'plain', 'utf-8'))
    
    # 로컬 SMTP 서버 사용 (설정 필요)
    # 또는 외부 SMTP 서버 사용
    # smtp = smtplib.SMTP('localhost')
    # smtp.send_message(msg)
    # smtp.quit()
    
    print("이메일 발송 시도 (SMTP 설정 필요)")
except Exception as e:
    print(f"이메일 발송 실패: {e}")
EOF
        log_message "이메일 발송 시도 (SMTP 설정 필요): $subject"
    fi
}

check_experiment_status() {
    local exp_name=$1
    local pattern=$2
    local result_file=$3
    
    local process_running=$(check_process_running "$exp_name" "$pattern")
    local result_exists=$(check_result_exists "$exp_name" "$result_file")
    
    if [ "$result_exists" == "1" ]; then
        echo "completed"
    elif [ "$process_running" -gt 0 ]; then
        echo "running"
    else
        echo "stopped"
    fi
}

generate_status_report() {
    local report="=== CENTaUR 실험 진행 상황 ===\n\n"
    report+="시간: $(date '+%Y-%m-%d %H:%M:%S')\n\n"
    
    for exp_name in "${!EXPERIMENTS[@]}"; do
        local pattern="${EXPERIMENTS[$exp_name]}"
        local result_file="${RESULT_FILES[$exp_name]}"
        local status=$(check_experiment_status "$exp_name" "$pattern" "$result_file")
        
        report+="$exp_name:\n"
        case $status in
            "completed")
                report+="  상태: ✅ 완료\n"
                if [ -f "$result_file" ]; then
                    local size=$(ls -lh "$result_file" | awk '{print $5}')
                    report+="  결과 파일: $result_file ($size)\n"
                    
                    # JSON 결과 파일인 경우 NLL 값 추출
                    if [[ "$result_file" == *.json ]]; then
                        local nll=$(python3 -c "import json; f=open('$result_file'); d=json.load(f); print(f\"NLL: {d.get('mean_nll', 'N/A')}\")" 2>/dev/null || echo "N/A")
                        report+="  $nll\n"
                    fi
                fi
                ;;
            "running")
                local pid=$(ps aux | grep -E "$pattern" | grep -v grep | awk '{print $2}' | head -1)
                local cpu=$(ps aux | grep -E "$pattern" | grep -v grep | awk '{print $3}' | head -1)
                local mem=$(ps aux | grep -E "$pattern" | grep -v grep | awk '{print $4}' | head -1)
                report+="  상태: 🔄 실행 중\n"
                report+="  PID: $pid\n"
                report+="  CPU: ${cpu}%\n"
                report+="  MEM: ${mem}%\n"
                ;;
            "stopped")
                report+="  상태: ⏸️  중지됨\n"
                ;;
        esac
        report+="\n"
    done
    
    # GPU 사용량 추가
    report+="=== GPU 상태 ===\n"
    if command -v nvidia-smi &> /dev/null; then
        report+="$(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total --format=csv,noheader | head -1)\n"
    fi
    
    echo -e "$report"
}

main() {
    log_message "모니터링 시작"
    
    # 초기 상태 저장
    declare -A last_status
    for exp_name in "${!EXPERIMENTS[@]}"; do
        last_status[$exp_name]="unknown"
    done
    
    while true; do
        log_message "상태 체크 시작"
        
        local status_changed=false
        
        for exp_name in "${!EXPERIMENTS[@]}"; do
            local pattern="${EXPERIMENTS[$exp_name]}"
            local result_file="${RESULT_FILES[$exp_name]}"
            local current_status=$(check_experiment_status "$exp_name" "$pattern" "$result_file")
            local last_stat="${last_status[$exp_name]}"
            
            if [ "$current_status" != "$last_stat" ]; then
                status_changed=true
                log_message "$exp_name 상태 변경: $last_stat -> $current_status"
                
                if [ "$current_status" == "completed" ]; then
                    local report=$(generate_status_report)
                    send_email "[CENTaUR] $exp_name 완료" "$report"
                    log_message "$exp_name 완료 알림 발송"
                fi
                
                last_status[$exp_name]=$current_status
            fi
        done
        
        # 상태 리포트 생성 및 저장
        local report=$(generate_status_report)
        echo -e "$report" > "$STATUS_FILE"
        
        if [ "$status_changed" == true ]; then
            log_message "상태 변경 감지됨"
        fi
        
        log_message "다음 체크까지 대기 중... (${CHECK_INTERVAL}초)"
        sleep "$CHECK_INTERVAL"
    done
}

# 스크립트 실행
main

