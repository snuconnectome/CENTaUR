#!/usr/bin/env python3
"""
CENTaUR 실험 모니터링 및 이메일 알림 스크립트
10분마다 체크하고, 작업 완료 시 이메일 발송
"""

import time
import subprocess
import os
import json
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path

EMAIL = "cha.jiook@gmail.com"
CHECK_INTERVAL = 600  # 10분 = 600초
LOG_FILE = "logs/monitor.log"
STATUS_FILE = "logs/experiment_status.json"

# SMTP 설정 (환경변수로 오버라이드 가능)
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USER = os.getenv("SMTP_USER", EMAIL)  # 기본값은 수신 이메일과 동일
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")  # 앱 비밀번호 필요
SMTP_FROM = os.getenv("SMTP_FROM", EMAIL)

# 작업 목록
EXPERIMENTS = {
    "EXAONE-3.5-32B": {
        "pattern": "extract_centaur_features.py.*exaone35",
        "result_file": "outputs/exaone35_features.npz"
    },
    "Qwen2.5-Base-LOO-CV": {
        "pattern": "fit_centaur_loo_cv_flexible.py.*qwen25_base",
        "result_file": "outputs/qwen25_base_nll_results.json"
    },
    "DeepSeek-Base-LOO-CV": {
        "pattern": "fit_centaur_loo_cv_flexible.py.*deepseek_base",
        "result_file": "outputs/deepseek_base_nll_results.json"
    }
}

def log_message(msg):
    """로그 메시지 기록"""
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    log_msg = f"[{timestamp}] {msg}\n"
    print(log_msg, end='')
    
    # 로그 디렉토리 생성
    os.makedirs(os.path.dirname(LOG_FILE), exist_ok=True)
    with open(LOG_FILE, 'a') as f:
        f.write(log_msg)

def check_process_running(pattern):
    """프로세스 실행 여부 확인"""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        # 패턴을 여러 부분으로 분리하여 모두 포함하는지 확인
        pattern_parts = pattern.split('.*')
        for line in result.stdout.split('\n'):
            if 'grep' in line:
                continue
            # 모든 패턴 부분이 포함되어 있는지 확인
            if all(part in line for part in pattern_parts if part):
                return True
        return False
    except:
        return False

def check_result_exists(result_file):
    """결과 파일 존재 여부 확인"""
    return os.path.exists(result_file)

def get_process_info(pattern):
    """프로세스 정보 가져오기"""
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True
        )
        lines = [line for line in result.stdout.split('\n') if pattern in line and 'grep' not in line]
        if lines:
            parts = lines[0].split()
            return {
                'pid': parts[1],
                'cpu': parts[2],
                'mem': parts[3]
            }
    except:
        pass
    return None

def send_email_smtp(subject, body):
    """SMTP를 사용한 이메일 발송"""
    if not SMTP_PASSWORD:
        log_message("⚠️  SMTP_PASSWORD 환경변수가 설정되지 않음")
        return False
    
    try:
        # 이메일 메시지 생성
        msg = MIMEMultipart()
        msg['From'] = SMTP_FROM
        msg['To'] = EMAIL
        msg['Subject'] = subject
        
        # 본문 추가
        msg.attach(MIMEText(body, 'plain', 'utf-8'))
        
        # SMTP 서버 연결 및 발송
        if SMTP_PORT == 587:
            # TLS 사용 (Gmail 등)
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
            server.starttls()
        elif SMTP_PORT == 465:
            # SSL 사용
            server = smtplib.SMTP_SSL(SMTP_SERVER, SMTP_PORT)
        else:
            # 일반 SMTP
            server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        log_message(f"✅ 이메일 발송 완료 (SMTP): {subject} -> {EMAIL}")
        return True
        
    except smtplib.SMTPAuthenticationError as e:
        log_message(f"❌ SMTP 인증 실패: {e}")
        log_message("💡 Gmail 사용 시: 앱 비밀번호를 생성하여 SMTP_PASSWORD에 설정하세요")
        return False
    except Exception as e:
        log_message(f"❌ SMTP 발송 실패: {e}")
        return False

def send_email_simple(subject, body):
    """간단한 이메일 발송 (여러 방법 시도)"""
    # 방법 1: mail 명령어 시도
    try:
        process = subprocess.Popen(
            ['mail', '-s', subject, EMAIL],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        process.communicate(input=body.encode('utf-8'))
        if process.returncode == 0:
            log_message(f"✅ 이메일 발송 완료 (mail): {subject} -> {EMAIL}")
            return True
    except FileNotFoundError:
        pass
    
    # 방법 2: sendmail 시도
    try:
        email_content = f"Subject: {subject}\n\n{body}"
        process = subprocess.Popen(
            ['sendmail', EMAIL],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        process.communicate(input=email_content.encode('utf-8'))
        if process.returncode == 0:
            log_message(f"✅ 이메일 발송 완료 (sendmail): {subject} -> {EMAIL}")
            return True
    except FileNotFoundError:
        pass
    
    # 방법 3: SMTP 사용 (Python)
    if send_email_smtp(subject, body):
        return True
    
    # 방법 4: 이메일 내용을 파일로 저장 (수동 발송 가능)
    email_file = f"logs/email_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    os.makedirs(os.path.dirname(email_file), exist_ok=True)
    with open(email_file, 'w', encoding='utf-8') as f:
        f.write(f"To: {EMAIL}\n")
        f.write(f"Subject: {subject}\n\n")
        f.write(body)
    
    log_message(f"⚠️  모든 이메일 발송 방법 실패")
    log_message(f"📧 이메일 내용 저장됨: {email_file}")
    log_message(f"📋 수동 발송 방법:")
    log_message(f"   1. SMTP 설정: export SMTP_PASSWORD='your_app_password'")
    log_message(f"   2. 또는: cat {email_file} | mail -s '{subject}' {EMAIL}")
    return False

def check_experiment_status(exp_name, pattern, result_file):
    """실험 상태 확인"""
    if check_result_exists(result_file):
        return "completed"
    elif check_process_running(pattern):
        return "running"
    else:
        return "stopped"

def generate_status_report():
    """상태 리포트 생성"""
    report = f"=== CENTaUR 실험 진행 상황 ===\n\n"
    report += f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    
    for exp_name, config in EXPERIMENTS.items():
        pattern = config["pattern"]
        result_file = config["result_file"]
        status = check_experiment_status(exp_name, pattern, result_file)
        
        report += f"{exp_name}:\n"
        
        if status == "completed":
            report += "  상태: ✅ 완료\n"
            if os.path.exists(result_file):
                size = os.path.getsize(result_file) / (1024 * 1024)  # MB
                report += f"  결과 파일: {result_file} ({size:.1f} MB)\n"
                
                # JSON 결과 파일인 경우 NLL 값 추출
                if result_file.endswith('.json'):
                    try:
                        with open(result_file) as f:
                            data = json.load(f)
                            nll = data.get('mean_nll', 'N/A')
                            report += f"  NLL: {nll}\n"
                    except:
                        pass
        
        elif status == "running":
            proc_info = get_process_info(pattern)
            if proc_info:
                report += "  상태: 🔄 실행 중\n"
                report += f"  PID: {proc_info['pid']}\n"
                report += f"  CPU: {proc_info['cpu']}%\n"
                report += f"  MEM: {proc_info['mem']}%\n"
            else:
                report += "  상태: 🔄 실행 중\n"
        
        else:
            report += "  상태: ⏸️  중지됨\n"
        
        report += "\n"
    
    # GPU 사용량 추가
    report += "=== GPU 상태 ===\n"
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used,memory.total', '--format=csv,noheader'],
            capture_output=True,
            text=True
        )
        if result.returncode == 0:
            report += result.stdout.strip() + "\n"
    except:
        report += "GPU 정보 확인 불가\n"
    
    return report

def main():
    """메인 함수"""
    log_message("모니터링 시작")
    
    # 초기 상태 저장
    last_status = {exp_name: "unknown" for exp_name in EXPERIMENTS.keys()}
    
    while True:
        log_message("상태 체크 시작")
        
        status_changed = False
        
        for exp_name, config in EXPERIMENTS.items():
            pattern = config["pattern"]
            result_file = config["result_file"]
            current_status = check_experiment_status(exp_name, pattern, result_file)
            last_stat = last_status[exp_name]
            
            if current_status != last_stat:
                status_changed = True
                log_message(f"{exp_name} 상태 변경: {last_stat} -> {current_status}")
                
                if current_status == "completed":
                    report = generate_status_report()
                    send_email_simple(f"[CENTaUR] {exp_name} 완료", report)
                    log_message(f"{exp_name} 완료 알림 발송")
                
                last_status[exp_name] = current_status
        
        # 상태 리포트 생성 및 저장
        report = generate_status_report()
        os.makedirs(os.path.dirname(STATUS_FILE), exist_ok=True)
        with open(STATUS_FILE, 'w') as f:
            f.write(report)
        
        if status_changed:
            log_message("상태 변경 감지됨")
        
        log_message(f"다음 체크까지 대기 중... ({CHECK_INTERVAL}초)")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()

