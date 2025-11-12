#!/usr/bin/env python3
"""
tmux centaur 세션 모니터링 및 이메일 알림 스크립트
10분마다 tmux 세션 로그를 모니터링하고 결과를 이메일로 전송
~/git/Emailer를 사용하여 이메일 전송
"""
import os
import sys
import time
import subprocess
from datetime import datetime
from pathlib import Path
import json

# Emailer 경로 추가
EMAILER_PATH = Path.home() / "git" / "Emailer"
sys.path.insert(0, str(EMAILER_PATH))

# 프로젝트 디렉토리
PROJECT_DIR = Path(__file__).parent.parent
LOGS_DIR = PROJECT_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# 모니터링 설정
CHECK_INTERVAL = 600  # 10분 (초)
TMUX_SESSION = "centaur"
EMAIL_TO = "cha.jiook@gmail.com"

def get_tmux_session_content():
    """tmux centaur 세션의 최근 내용 가져오기"""
    try:
        result = subprocess.run(
            ["tmux", "capture-pane", "-t", TMUX_SESSION, "-p"],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return result.stdout.strip()
        else:
            return f"❌ tmux 세션 '{TMUX_SESSION}' 접근 실패: {result.stderr}"
    except subprocess.TimeoutExpired:
        return "❌ tmux 세션 읽기 타임아웃"
    except Exception as e:
        return f"❌ 오류: {str(e)}"

def get_tmux_session_info():
    """tmux 세션 정보 가져오기"""
    try:
        result = subprocess.run(
            ["tmux", "ls"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            for line in lines:
                if TMUX_SESSION in line:
                    return line
            return f"❌ 세션 '{TMUX_SESSION}'을 찾을 수 없음"
        else:
            return f"❌ tmux 세션 목록 가져오기 실패"
    except Exception as e:
        return f"❌ 오류: {str(e)}"

def get_running_processes():
    """실행 중인 프로세스 확인"""
    try:
        result = subprocess.run(
            ["ps", "aux"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            relevant = []
            for line in lines:
                if any(keyword in line for keyword in ['train_qwen25', 'extract_centaur', 'monitor', 'fit_centaur']):
                    if 'python' in line or 'bash' in line:
                        parts = line.split()
                        if len(parts) > 10:
                            pid = parts[1]
                            cpu = parts[2]
                            mem = parts[3]
                            cmd = ' '.join(parts[10:])
                            relevant.append(f"  PID {pid}: CPU {cpu}%, MEM {mem}% - {cmd[:80]}")
            return relevant if relevant else ["  실행 중인 관련 프로세스 없음"]
        else:
            return ["  프로세스 정보 가져오기 실패"]
    except Exception as e:
        return [f"  오류: {str(e)}"]

def get_gpu_status():
    """GPU 상태 확인"""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu", 
             "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.strip().split('\n')
            gpu_info = []
            for i, line in enumerate(lines):
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 4:
                    gpu_info.append(f"  GPU {i}: 사용률 {parts[0]}%, 메모리 {parts[1]}MB/{parts[2]}MB, 온도 {parts[3]}°C")
            return gpu_info if gpu_info else ["  GPU 정보 없음"]
        else:
            return ["  GPU 정보 가져오기 실패"]
    except Exception as e:
        return [f"  GPU 확인 오류: {str(e)}"]

def get_recent_logs():
    """최근 로그 파일 확인"""
    try:
        log_files = []
        if LOGS_DIR.exists():
            for log_file in sorted(LOGS_DIR.glob("*.log"), key=lambda p: p.stat().st_mtime, reverse=True)[:5]:
                size = log_file.stat().st_size
                mtime = datetime.fromtimestamp(log_file.stat().st_mtime)
                log_files.append(f"  {log_file.name}: {size/1024:.1f}KB ({mtime.strftime('%Y-%m-%d %H:%M:%S')})")
        return log_files if log_files else ["  로그 파일 없음"]
    except Exception as e:
        return [f"  로그 확인 오류: {str(e)}"]

def generate_monitoring_report():
    """모니터링 리포트 생성"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = []
    report.append("=" * 70)
    report.append(f"tmux centaur 세션 모니터링 리포트")
    report.append(f"시간: {timestamp}")
    report.append("=" * 70)
    report.append("")
    
    # tmux 세션 정보
    report.append("📋 tmux 세션 정보:")
    session_info = get_tmux_session_info()
    report.append(f"  {session_info}")
    report.append("")
    
    # tmux 세션 내용 (최근 50줄)
    report.append("📝 tmux centaur 세션 내용 (최근 50줄):")
    report.append("-" * 70)
    session_content = get_tmux_session_content()
    lines = session_content.split('\n')
    recent_lines = lines[-50:] if len(lines) > 50 else lines
    for line in recent_lines:
        report.append(line[:100])  # 최대 100자로 제한
    report.append("-" * 70)
    report.append("")
    
    # 실행 중인 프로세스
    report.append("🔄 실행 중인 프로세스:")
    processes = get_running_processes()
    for proc in processes:
        report.append(proc)
    report.append("")
    
    # GPU 상태
    report.append("🎮 GPU 상태:")
    gpu_status = get_gpu_status()
    for gpu in gpu_status:
        report.append(gpu)
    report.append("")
    
    # 최근 로그
    report.append("📁 최근 로그 파일:")
    logs = get_recent_logs()
    for log in logs:
        report.append(log)
    report.append("")
    
    report.append("=" * 70)
    
    return '\n'.join(report)

def send_email_via_emailer(subject, body):
    """Emailer를 사용하여 이메일 전송"""
    try:
        # Emailer의 Gmail API 사용 시도
        from connectors.gmail_mcp import GmailConnector
        
        # Gmail API는 주로 읽기용이므로, SMTP를 직접 사용하는 방법 시도
        # 또는 Emailer의 다른 기능 사용
        print(f"⚠️  Emailer는 주로 읽기 전용입니다. SMTP로 전송합니다.")
        return send_email_via_smtp(subject, body)
    except ImportError:
        print(f"⚠️  Emailer 모듈을 찾을 수 없습니다. SMTP로 전송합니다.")
        return send_email_via_smtp(subject, body)
    except Exception as e:
        print(f"⚠️  Emailer 사용 실패: {e}. SMTP로 전송합니다.")
        return send_email_via_smtp(subject, body)

def send_email_via_smtp(subject, body):
    """SMTP를 사용하여 이메일 전송 (Emailer가 실패할 경우 대체)"""
    import smtplib
    from email.mime.text import MIMEText
    from email.mime.multipart import MIMEMultipart
    
    SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
    SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
    SMTP_USER = os.getenv("SMTP_USER", "cha.jiook@gmail.com")
    SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
    
    if not SMTP_PASSWORD:
        # 파일로 저장
        email_file = LOGS_DIR / f"email_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(email_file, 'w', encoding='utf-8') as f:
            f.write(f"To: {EMAIL_TO}\n")
            f.write(f"Subject: {subject}\n\n")
            f.write(body)
        print(f"⚠️  SMTP_PASSWORD 없음. 파일로 저장: {email_file}")
        return False
    
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = SMTP_USER
        msg['To'] = EMAIL_TO
        msg['Subject'] = subject
        
        part1 = MIMEText(body, 'plain', 'utf-8')
        msg.attach(part1)
        
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        print(f"✅ 이메일 전송 완료: {EMAIL_TO}")
        return True
    except Exception as e:
        print(f"❌ 이메일 전송 실패: {e}")
        # 파일로 저장
        email_file = LOGS_DIR / f"email_failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(email_file, 'w', encoding='utf-8') as f:
            f.write(f"To: {EMAIL_TO}\n")
            f.write(f"Subject: {subject}\n\n")
            f.write(body)
            f.write(f"\n\nError: {e}")
        return False

def main():
    """메인 루프"""
    print("=" * 70)
    print("tmux centaur 세션 모니터링 시작")
    print("=" * 70)
    print(f"세션: {TMUX_SESSION}")
    print(f"이메일 수신: {EMAIL_TO}")
    print(f"체크 간격: {CHECK_INTERVAL}초 (10분)")
    print(f"프로젝트 디렉토리: {PROJECT_DIR}")
    print("")
    
    # Emailer 경로 확인
    if EMAILER_PATH.exists():
        print(f"✅ Emailer 경로 확인: {EMAILER_PATH}")
    else:
        print(f"⚠️  Emailer 경로 없음: {EMAILER_PATH}")
        print("   SMTP로 전송합니다.")
    print("")
    
    check_count = 0
    
    try:
        while True:
            check_count += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"[{timestamp}] 체크 #{check_count} 시작...")
            
            # 모니터링 리포트 생성
            report = generate_monitoring_report()
            
            # 콘솔 출력
            print(report)
            print("")
            
            # 이메일 전송
            subject = f"[CENTaUR tmux] 모니터링 리포트 #{check_count} - {timestamp}"
            send_email_via_emailer(subject, report)
            
            print(f"다음 체크까지 대기 중... ({CHECK_INTERVAL}초)")
            print("")
            
            time.sleep(CHECK_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n모니터링 중지됨")
        final_report = generate_monitoring_report()
        subject = f"[CENTaUR tmux] 모니터링 종료 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        send_email_via_emailer(subject, final_report)
    except Exception as e:
        print(f"\n오류 발생: {e}")
        import traceback
        error_report = f"모니터링 오류 발생:\n\n{str(e)}\n\n{traceback.format_exc()}"
        send_email_via_emailer("[CENTaUR tmux] 모니터링 오류", error_report)

if __name__ == "__main__":
    main()

