#!/usr/bin/env python3
"""
GPU 성능 분석 스크립트
nsys 프로파일 결과를 분석하여 GPU 사용률, 메모리 사용량, 성능 지표를 생성
"""
import subprocess
import json
import sys
from pathlib import Path
from datetime import datetime

def analyze_profile(profile_path):
    """nsys 프로파일 파일 분석"""
    profile_path = Path(profile_path)
    if not profile_path.exists():
        print(f"Error: Profile file not found: {profile_path}")
        return None
    
    print(f"\n{'='*70}")
    print(f"Analyzing GPU Profile: {profile_path.name}")
    print(f"{'='*70}\n")
    
    results = {
        'profile_file': str(profile_path),
        'timestamp': datetime.now().isoformat(),
        'cuda_api': {},
        'gpu_memory': {},
        'gpu_kernels': {}
    }
    
    # CUDA API Summary
    print("1. CUDA API Summary")
    print("-" * 70)
    try:
        output = subprocess.run(
            ['/usr/local/bin/nsys', 'stats', '--report', 'cuda_api_sum', str(profile_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if output.returncode == 0:
            lines = output.stdout.split('\n')
            in_table = False
            api_calls = []
            
            for line in lines:
                if 'CUDA API Summary' in line:
                    in_table = True
                    continue
                if in_table and '---' in line:
                    continue
                if in_table and line.strip() and not line.startswith('Time'):
                    parts = line.split()
                    if len(parts) >= 3:
                        try:
                            time_pct = float(parts[0].rstrip('%'))
                            total_time = float(parts[1])
                            num_calls = int(parts[2])
                            api_name = ' '.join(parts[8:]) if len(parts) > 8 else parts[-1]
                            
                            api_calls.append({
                                'time_percent': time_pct,
                                'total_time_ns': total_time,
                                'num_calls': num_calls,
                                'name': api_name
                            })
                        except (ValueError, IndexError):
                            pass
            
            results['cuda_api'] = {
                'total_calls': sum(c['num_calls'] for c in api_calls),
                'top_apis': sorted(api_calls, key=lambda x: x['time_percent'], reverse=True)[:10]
            }
            
            for api in results['cuda_api']['top_apis'][:5]:
                print(f"  {api['name']:40s} {api['time_percent']:6.2f}% ({api['num_calls']} calls)")
        else:
            print(f"  Warning: Could not generate CUDA API summary")
            print(f"  Error: {output.stderr}")
    except Exception as e:
        print(f"  Error analyzing CUDA API: {e}")
    
    # GPU Memory Summary
    print("\n2. GPU Memory Usage")
    print("-" * 70)
    try:
        output = subprocess.run(
            ['/usr/local/bin/nsys', 'stats', '--report', 'cuda_gpu_mem_size_sum', str(profile_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if output.returncode == 0 and 'SKIPPED' not in output.stdout:
            lines = output.stdout.split('\n')
            for line in lines[:15]:
                if line.strip() and not line.startswith('Processing'):
                    print(f"  {line}")
        else:
            print("  No GPU memory data available")
    except Exception as e:
        print(f"  Error analyzing GPU memory: {e}")
    
    # GPU Kernel Summary
    print("\n3. GPU Kernel Execution")
    print("-" * 70)
    try:
        output = subprocess.run(
            ['/usr/local/bin/nsys', 'stats', '--report', 'cuda_gpu_kern_sum', str(profile_path)],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if output.returncode == 0 and 'SKIPPED' not in output.stdout:
            lines = output.stdout.split('\n')
            for line in lines[:15]:
                if line.strip() and not line.startswith('Processing'):
                    print(f"  {line}")
        else:
            print("  No GPU kernel data available (may be CPU-only or compatibility issue)")
    except Exception as e:
        print(f"  Error analyzing GPU kernels: {e}")
    
    return results

def generate_report(profiles_dir, output_file):
    """모든 프로파일 결과를 종합한 리포트 생성"""
    profiles_dir = Path(profiles_dir)
    output_file = Path(output_file)
    
    print(f"\n{'='*70}")
    print(f"GPU Performance Analysis Report")
    print(f"{'='*70}\n")
    
    profile_files = list(profiles_dir.glob("*.nsys-rep"))
    
    if not profile_files:
        print(f"No profile files found in {profiles_dir}")
        return
    
    all_results = {}
    
    for profile_file in profile_files:
        print(f"\nAnalyzing: {profile_file.name}")
        results = analyze_profile(profile_file)
        if results:
            all_results[profile_file.stem] = results
    
    # 리포트 저장
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"Report saved to: {output_file}")
    print(f"{'='*70}\n")
    
    # 요약 출력
    print("Summary:")
    print("-" * 70)
    for job_name, results in all_results.items():
        if 'cuda_api' in results and 'total_calls' in results['cuda_api']:
            total_calls = results['cuda_api']['total_calls']
            top_api = results['cuda_api']['top_apis'][0] if results['cuda_api']['top_apis'] else None
            print(f"  {job_name:30s} {total_calls:6d} CUDA API calls")
            if top_api:
                print(f"    Top API: {top_api['name']} ({top_api['time_percent']:.1f}%)")

def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print(f"  {sys.argv[0]} <profile_file.nsys-rep>")
        print(f"  {sys.argv[0]} --all <profiles_dir> [output.json]")
        sys.exit(1)
    
    if sys.argv[1] == '--all':
        profiles_dir = sys.argv[2] if len(sys.argv) > 2 else "profiles"
        output_file = sys.argv[3] if len(sys.argv) > 3 else "gpu_performance_report.json"
        generate_report(profiles_dir, output_file)
    else:
        profile_file = sys.argv[1]
        results = analyze_profile(profile_file)
        if results:
            print("\nAnalysis complete!")

if __name__ == "__main__":
    main()

