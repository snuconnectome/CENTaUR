#!/home/juke/git/CENTaUR/dgx-venv/bin/python
"""
Ko-CENTaUR 파이프라인 모니터링 스크립트
1시간마다 진행상황을 이메일로 전송
"""

import os
import sys
import subprocess
import time
from datetime import datetime
from pathlib import Path

# Gmail API 사용 (Emailer 모듈은 send 기능이 없어서 직접 구현)


def get_pipeline_status():
    """파이프라인 상태 수집"""
    status = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'phase1_finetuning': {},
        'phase2_extraction': {},
        'phase3_loo_cv': {},
        'phase4_analysis': {},
        'gpu_status': {},
        'recent_logs': []
    }

    # Phase 1: Fine-tuning 확인
    qwen_path = Path('models/qwen25-32b-qlora')
    deepseek_path = Path('models/deepseek-r1-qlora')

    status['phase1_finetuning']['qwen25'] = '✅ 완료' if qwen_path.exists() else '❌ 대기'
    status['phase1_finetuning']['deepseek'] = '✅ 완료' if deepseek_path.exists() else '⏳ 진행 중'

    # Phase 2: Feature Extraction 확인
    features_dir = Path('data/features')
    expected_features = [
        'qwen25_base_features.pth',
        'qwen25_finetuned_features.pth',
        'deepseek_base_features.pth',
        'deepseek_finetuned_features.pth',
        'exaone35_base_features.pth',
        'gpt_oss_20b_base_features.pth',
        'kimi_k2_base_features.pth'
    ]

    completed_features = 0
    for feature_file in expected_features:
        if (features_dir / feature_file).exists():
            completed_features += 1

    status['phase2_extraction']['completed'] = completed_features
    status['phase2_extraction']['total'] = len(expected_features)
    status['phase2_extraction']['progress'] = f"{completed_features}/{len(expected_features)}"

    # Phase 3: LOO CV 확인
    results_dir = Path('data/results')
    expected_results = [
        'qwen25_base_loo_results.json',
        'qwen25_finetuned_loo_results.json',
        'deepseek_base_loo_results.json',
        'deepseek_finetuned_loo_results.json',
        'exaone35_base_loo_results.json',
        'gpt_oss_20b_base_loo_results.json',
        'kimi_k2_base_loo_results.json'
    ]

    completed_loo = 0
    for result_file in expected_results:
        if (results_dir / result_file).exists():
            completed_loo += 1

    status['phase3_loo_cv']['completed'] = completed_loo
    status['phase3_loo_cv']['total'] = len(expected_results)
    status['phase3_loo_cv']['progress'] = f"{completed_loo}/{len(expected_results)}"

    # Phase 4: Analysis 확인
    reports_dir = Path('reports')
    status['phase4_analysis']['completed'] = (reports_dir / 'final_analysis.md').exists()

    # GPU 상태
    try:
        gpu_output = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=index,name,temperature.gpu,utilization.gpu,utilization.memory,memory.used,memory.total',
            '--format=csv,noheader,nounits'
        ], text=True)

        gpu_parts = gpu_output.strip().split(', ')
        status['gpu_status'] = {
            'name': gpu_parts[1] if len(gpu_parts) > 1 else 'N/A',
            'temp': f"{gpu_parts[2]}°C" if len(gpu_parts) > 2 else 'N/A',
            'util': f"{gpu_parts[3]}%" if len(gpu_parts) > 3 else 'N/A',
            'mem_util': f"{gpu_parts[4]}%" if len(gpu_parts) > 4 else 'N/A',
            'mem_used': gpu_parts[5] if len(gpu_parts) > 5 else 'N/A',
            'mem_total': gpu_parts[6] if len(gpu_parts) > 6 else 'N/A'
        }
    except Exception as e:
        status['gpu_status'] = {'error': str(e)}

    # 최근 로그 (마지막 20줄)
    log_files = sorted(Path('logs').glob('full_pipeline_*.log'))
    if log_files:
        latest_log = log_files[-1]
        try:
            with open(latest_log, 'r') as f:
                lines = f.readlines()
                status['recent_logs'] = lines[-20:]
                status['log_file'] = str(latest_log)
        except Exception as e:
            status['recent_logs'] = [f"Error reading log: {e}"]

    return status


def format_email_body(status):
    """이메일 본문 생성 (HTML)"""
    body = f"""
<html>
<head>
<style>
body {{ font-family: 'Courier New', monospace; background-color: #f5f5f5; padding: 20px; }}
.container {{ background-color: white; padding: 20px; border-radius: 5px; max-width: 800px; margin: 0 auto; }}
h1 {{ color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 10px; }}
h2 {{ color: #34495e; margin-top: 20px; }}
.status {{ padding: 10px; margin: 10px 0; border-left: 4px solid #3498db; background-color: #ecf0f1; }}
.completed {{ border-left-color: #27ae60; }}
.inprogress {{ border-left-color: #f39c12; }}
.pending {{ border-left-color: #e74c3c; }}
.logs {{ background-color: #2c3e50; color: #ecf0f1; padding: 15px; border-radius: 5px; overflow-x: auto; font-size: 12px; }}
</style>
</head>
<body>
<div class="container">
<h1>Ko-CENTaUR 파이프라인 진행 상황</h1>
<p><strong>시간:</strong> {status['timestamp']}</p>

<h2>Phase 1: Fine-tuning</h2>
<div class="status {'completed' if status['phase1_finetuning']['qwen25'] == '✅ 완료' else 'pending'}">
    <strong>Qwen2.5-32B:</strong> {status['phase1_finetuning']['qwen25']}
</div>
<div class="status {'completed' if status['phase1_finetuning']['deepseek'] == '✅ 완료' else 'inprogress'}">
    <strong>DeepSeek-R1-32B:</strong> {status['phase1_finetuning']['deepseek']}
</div>

<h2>Phase 2: Feature Extraction</h2>
<div class="status">
    <strong>진행률:</strong> {status['phase2_extraction']['progress']} 완료
</div>

<h2>Phase 3: LOO Cross-Validation</h2>
<div class="status">
    <strong>진행률:</strong> {status['phase3_loo_cv']['progress']} 완료
</div>

<h2>Phase 4: Analysis</h2>
<div class="status {'completed' if status['phase4_analysis']['completed'] else 'pending'}">
    <strong>상태:</strong> {'✅ 완료' if status['phase4_analysis']['completed'] else '❌ 대기'}
</div>

<h2>GPU 상태</h2>
"""

    if 'error' not in status['gpu_status']:
        body += f"""
<div class="status">
    <strong>GPU:</strong> {status['gpu_status'].get('name', 'N/A')}<br>
    <strong>온도:</strong> {status['gpu_status'].get('temp', 'N/A')}<br>
    <strong>사용률:</strong> {status['gpu_status'].get('util', 'N/A')}<br>
    <strong>메모리 사용률:</strong> {status['gpu_status'].get('mem_util', 'N/A')}<br>
    <strong>메모리 사용:</strong> {status['gpu_status'].get('mem_used', 'N/A')} / {status['gpu_status'].get('mem_total', 'N/A')} MB
</div>
"""
    else:
        body += f"""
<div class="status pending">
    <strong>GPU 정보 수집 실패:</strong> {status['gpu_status']['error']}
</div>
"""

    body += f"""
<h2>최근 로그 (마지막 10줄)</h2>
<div class="logs">
"""

    if status['recent_logs']:
        log_html = '<br>'.join([''.join(line).strip() for line in status['recent_logs'][-10:]])
        body += log_html
    else:
        body += "로그 없음"

    body += f"""
</div>

<p style="margin-top: 20px; color: #7f8c8d; font-size: 12px;">
<strong>로그 파일:</strong> {status.get('log_file', 'N/A')}<br>
이 이메일은 자동으로 생성되었습니다.
</p>

</div>
</body>
</html>
"""

    return body


def send_email_via_gmail_api(subject, html_body, to_email):
    """Gmail API를 사용해서 이메일 전송"""
    try:
        from google.oauth2.credentials import Credentials
        from googleapiclient.discovery import build
        from email.mime.text import MIMEText
        from email.mime.multipart import MIMEMultipart
        import base64
        import pickle

        # Emailer의 토큰 사용
        token_path = Path.home() / 'git' / 'Emailer' / 'token.pickle'

        if not token_path.exists():
            print(f"❌ Gmail API 토큰 없음: {token_path}")
            return False

        with open(token_path, 'rb') as token:
            creds = pickle.load(token)

        service = build('gmail', 'v1', credentials=creds)

        # 이메일 작성
        message = MIMEMultipart('alternative')
        message['To'] = to_email
        message['Subject'] = subject

        # HTML 파트
        html_part = MIMEText(html_body, 'html')
        message.attach(html_part)

        # Base64 인코딩
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode('utf-8')

        # 전송
        send_message = service.users().messages().send(
            userId='me',
            body={'raw': raw_message}
        ).execute()

        print(f"✅ 이메일 전송 성공: {to_email}")
        print(f"   Message ID: {send_message['id']}")
        return True

    except Exception as e:
        print(f"❌ 이메일 전송 실패: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """메인 모니터링 루프"""
    to_email = 'cha.jiook@gmail.com'
    check_interval = 3600  # 1시간 (초)

    print("="*60)
    print("Ko-CENTaUR 파이프라인 모니터링 시작")
    print("="*60)
    print(f"이메일: {to_email}")
    print(f"체크 간격: {check_interval}초 (1시간)")
    print("="*60)
    print()

    iteration = 0

    while True:
        iteration += 1
        print(f"\n[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] 체크 #{iteration}")

        # 상태 수집
        status = get_pipeline_status()

        # 이메일 본문 생성
        subject = f"[Ko-CENTaUR] 파이프라인 진행 상황 #{iteration}"
        body = format_email_body(status)

        # 콘솔 출력
        print("현재 상태:")
        print(f"  Phase 1: Qwen ({status['phase1_finetuning']['qwen25']}), DeepSeek ({status['phase1_finetuning']['deepseek']})")
        print(f"  Phase 2: {status['phase2_extraction']['progress']} features")
        print(f"  Phase 3: {status['phase3_loo_cv']['progress']} LOO CV")
        print(f"  Phase 4: {'완료' if status['phase4_analysis']['completed'] else '대기'}")
        print(f"  GPU: {status['gpu_status'].get('util', 'N/A')} utilization")

        # 이메일 전송
        send_email_via_gmail_api(subject, body, to_email)

        # Phase 4가 완료되면 종료
        if status['phase4_analysis']['completed']:
            print("\n✅ 전체 파이프라인 완료! 모니터링 종료.")

            # 최종 이메일
            final_subject = "[Ko-CENTaUR] 🎉 전체 파이프라인 완료!"
            final_body = format_email_body(status)
            send_email_via_gmail_api(final_subject, final_body, to_email)
            break

        # 1시간 대기
        print(f"다음 체크까지 {check_interval}초 대기...")
        time.sleep(check_interval)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n⚠️ 모니터링 중단됨 (Ctrl+C)")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ 오류 발생: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
