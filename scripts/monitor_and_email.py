#!/usr/bin/env python3
"""
GPU 작업 모니터링 및 이메일 알림 스크립트
10분마다 진행상황을 이메일로 전송
"""
import os
import sys
import time
import subprocess
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from pathlib import Path
import json

# 이메일 설정
EMAIL_TO = "cha.jiook@gmail.com"
EMAIL_FROM = os.getenv("SMTP_FROM", "cha.jiook@gmail.com")
SMTP_SERVER = os.getenv("SMTP_SERVER", "smtp.gmail.com")
SMTP_PORT_STR = os.getenv("SMTP_PORT", "587")
try:
    SMTP_PORT = int(SMTP_PORT_STR)
except (ValueError, TypeError):
    SMTP_PORT = 587
SMTP_USER = os.getenv("SMTP_USER", "cha.jiook@gmail.com")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")

# 모니터링 설정
CHECK_INTERVAL = 600  # 10분 (초)
PROJECT_DIR = Path(__file__).parent.parent

def send_email(subject, body, html_body=None):
    """이메일 전송"""
    if not SMTP_PASSWORD:
        print("⚠️  SMTP_PASSWORD not set, saving email to file instead")
        email_file = PROJECT_DIR / "logs" / f"email_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        email_file.parent.mkdir(parents=True, exist_ok=True)
        with open(email_file, 'w') as f:
            f.write(f"To: {EMAIL_TO}\n")
            f.write(f"Subject: {subject}\n\n")
            f.write(body)
        print(f"   Email saved to: {email_file}")
        return False
    
    try:
        msg = MIMEMultipart('alternative')
        msg['From'] = EMAIL_FROM
        msg['To'] = EMAIL_TO
        msg['Subject'] = subject
        
        # Plain text
        part1 = MIMEText(body, 'plain', 'utf-8')
        msg.attach(part1)
        
        # HTML (if provided)
        if html_body:
            part2 = MIMEText(html_body, 'html', 'utf-8')
            msg.attach(part2)
        
        # Send email
        server = smtplib.SMTP(SMTP_SERVER, SMTP_PORT)
        server.starttls()
        server.login(SMTP_USER, SMTP_PASSWORD)
        server.send_message(msg)
        server.quit()
        
        print(f"✅ Email sent to {EMAIL_TO}")
        return True
    except Exception as e:
        print(f"❌ Failed to send email: {e}")
        # Save to file as backup
        email_file = PROJECT_DIR / "logs" / f"email_failed_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        email_file.parent.mkdir(parents=True, exist_ok=True)
        with open(email_file, 'w') as f:
            f.write(f"To: {EMAIL_TO}\n")
            f.write(f"Subject: {subject}\n\n")
            f.write(body)
            f.write(f"\n\nError: {e}")
        return False

def get_gpu_status():
    """GPU 상태 확인"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu', 
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            return result.stdout.strip()
        return "GPU status unavailable"
    except Exception as e:
        return f"GPU status error: {e}"

def get_running_jobs():
    """실행 중인 작업 확인"""
    jobs = []
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if result.returncode == 0:
            lines = result.stdout.split('\n')
            for line in lines:
                if 'extract_centaur' in line or 'fit_centaur_loo_cv' in line or 'train' in line.lower():
                    if 'python' in line and 'grep' not in line:
                        parts = line.split()
                        if len(parts) >= 11:
                            jobs.append({
                                'pid': parts[1],
                                'cpu': parts[2],
                                'mem': parts[3],
                                'cmd': ' '.join(parts[10:])[:100]
                            })
    except Exception as e:
        pass
    return jobs

def get_recent_logs(log_dir, n_lines=10):
    """최근 로그 확인"""
    log_dir = Path(log_dir)
    if not log_dir.exists():
        return "No log directory"
    
    logs = {}
    for log_file in sorted(log_dir.glob("*.log"), key=lambda x: x.stat().st_mtime, reverse=True)[:5]:
        try:
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
                logs[log_file.name] = '\n'.join(lines[-n_lines:])
        except Exception as e:
            logs[log_file.name] = f"Error reading: {e}"
    
    return logs

def get_output_files():
    """출력 파일 확인"""
    output_dir = PROJECT_DIR / "outputs"
    if not output_dir.exists():
        return {}
    
    files = {}
    for pattern in ["*.npz", "*.json", "*.pth"]:
        for f in output_dir.glob(pattern):
            stat = f.stat()
            files[f.name] = {
                'size': f"{stat.st_size / 1024 / 1024:.2f} MB",
                'modified': datetime.fromtimestamp(stat.st_mtime).strftime('%Y-%m-%d %H:%M:%S')
            }
    
    return files

def generate_status_report():
    """상태 리포트 생성"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    report = []
    report.append("=" * 70)
    report.append("CENTaUR GPU 작업 모니터링 리포트")
    report.append("=" * 70)
    report.append(f"시간: {timestamp}")
    report.append("")
    
    # GPU 상태
    report.append("--- GPU 상태 ---")
    gpu_status = get_gpu_status()
    for line in gpu_status.split('\n'):
        if line.strip():
            report.append(f"  {line}")
    report.append("")
    
    # 실행 중인 작업
    report.append("--- 실행 중인 작업 ---")
    jobs = get_running_jobs()
    if jobs:
        for job in jobs:
            report.append(f"  PID {job['pid']}: CPU {job['cpu']}%, MEM {job['mem']}%")
            report.append(f"    {job['cmd']}")
    else:
        report.append("  실행 중인 작업 없음")
    report.append("")
    
    # 최근 로그
    report.append("--- 최근 로그 (마지막 5줄) ---")
    logs = get_recent_logs(PROJECT_DIR / "logs")
    for log_name, log_content in list(logs.items())[:3]:
        report.append(f"  [{log_name}]")
        for line in log_content.split('\n')[-5:]:
            if line.strip():
                report.append(f"    {line}")
        report.append("")
    
    # 출력 파일
    report.append("--- 출력 파일 ---")
    files = get_output_files()
    if files:
        for fname, info in sorted(files.items(), key=lambda x: x[1]['modified'], reverse=True)[:5]:
            report.append(f"  {fname}: {info['size']} (수정: {info['modified']})")
    else:
        report.append("  출력 파일 없음")
    report.append("")
    
    # 프로파일 파일
    profile_dir = PROJECT_DIR / "profiles"
    if profile_dir.exists():
        report.append("--- 프로파일 파일 ---")
        profiles = list(profile_dir.glob("*.nsys-rep"))
        if profiles:
            for p in sorted(profiles, key=lambda x: x.stat().st_mtime, reverse=True)[:3]:
                size = p.stat().st_size / 1024
                mtime = datetime.fromtimestamp(p.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
                report.append(f"  {p.name}: {size:.1f} KB ({mtime})")
        else:
            report.append("  프로파일 파일 없음")
    report.append("")
    
    report.append("=" * 70)
    report.append(f"다음 체크: {datetime.fromtimestamp(time.time() + CHECK_INTERVAL).strftime('%Y-%m-%d %H:%M:%S')}")
    report.append("=" * 70)
    
    return '\n'.join(report)

def main():
    """메인 루프"""
    print("=" * 70)
    print("CENTaUR GPU 작업 모니터링 시작")
    print("=" * 70)
    print(f"이메일 수신: {EMAIL_TO}")
    print(f"체크 간격: {CHECK_INTERVAL}초 (10분)")
    print(f"프로젝트 디렉토리: {PROJECT_DIR}")
    print("")
    
    if not SMTP_PASSWORD:
        print("⚠️  경고: SMTP_PASSWORD 환경변수가 설정되지 않았습니다.")
        print("   이메일 대신 파일로 저장됩니다.")
        print("   설정 방법:")
        print("   export SMTP_PASSWORD='your_gmail_app_password'")
        print("")
    
    check_count = 0
    
    try:
        while True:
            check_count += 1
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            print(f"[{timestamp}] 체크 #{check_count} 시작...")
            
            # 상태 리포트 생성
            report = generate_status_report()
            
            # 콘솔 출력
            print(report)
            print("")
            
            # 이메일 전송
            subject = f"[CENTaUR] 작업 진행 상황 리포트 #{check_count} - {timestamp}"
            send_email(subject, report)
            
            print(f"다음 체크까지 대기 중... ({CHECK_INTERVAL}초)")
            print("")
            
            time.sleep(CHECK_INTERVAL)
            
    except KeyboardInterrupt:
        print("\n모니터링 중지됨")
        final_report = generate_status_report()
        subject = f"[CENTaUR] 모니터링 종료 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        send_email(subject, final_report)
    except Exception as e:
        print(f"\n오류 발생: {e}")
        error_report = f"모니터링 오류 발생:\n\n{str(e)}\n\n{traceback.format_exc()}"
        send_email("[CENTaUR] 모니터링 오류", error_report)

if __name__ == "__main__":
    import traceback
    main()

