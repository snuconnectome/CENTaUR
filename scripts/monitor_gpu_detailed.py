#!/usr/bin/env python3
"""
GPU 모니터링 스크립트 - nvidia-smi와 nvidia-ml-py를 사용한 상세 모니터링
"""
import subprocess
import time
import sys
import os
from datetime import datetime

try:
    import pynvml
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False
    print("Warning: pynvml not available, using nvidia-smi only")

def get_gpu_info_nvml():
    """nvidia-ml-py를 사용한 GPU 정보"""
    if not NVML_AVAILABLE:
        return None
    
    try:
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        
        # Memory info
        mem_info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        
        # Utilization
        util = pynvml.nvmlDeviceGetUtilizationRates(handle)
        
        # Power
        try:
            power = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0  # mW to W
        except:
            power = None
        
        # Temperature
        try:
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
        except:
            temp = None
        
        # Processes
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        
        return {
            'memory_used_gb': mem_info.used / 1024**3,
            'memory_total_gb': mem_info.total / 1024**3,
            'memory_percent': (mem_info.used / mem_info.total) * 100,
            'gpu_util': util.gpu,
            'memory_util': util.memory,
            'power_w': power,
            'temp_c': temp,
            'processes': len(processes),
            'process_details': [
                {
                    'pid': p.pid,
                    'memory_mb': p.usedGpuMemory / 1024**2
                } for p in processes
            ]
        }
    except Exception as e:
        print(f"Error getting NVML info: {e}")
        return None

def get_gpu_info_smi():
    """nvidia-smi를 사용한 GPU 정보"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,compute_mode', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return None
        
        lines = result.stdout.strip().split('\n')
        gpu_info = []
        
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            if len(parts) >= 9:
                gpu_info.append({
                    'index': parts[0],
                    'name': parts[1],
                    'gpu_util': int(parts[2]) if parts[2].isdigit() else 0,
                    'memory_util': int(parts[3]) if parts[3].isdigit() else 0,
                    'memory_used_mb': int(parts[4]) if parts[4].isdigit() else 0,
                    'memory_total_mb': int(parts[5]) if parts[5].isdigit() else 0,
                    'power_w': float(parts[6]) if parts[6] != '[N/A]' else None,
                    'temp_c': int(parts[7]) if parts[7].isdigit() else None,
                    'compute_mode': parts[8]
                })
        
        return gpu_info
    except Exception as e:
        print(f"Error getting SMI info: {e}")
        return None

def get_processes_smi():
    """nvidia-smi를 사용한 프로세스 정보"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-compute-apps=pid,process_name,used_memory', '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            timeout=5
        )
        
        if result.returncode != 0:
            return []
        
        processes = []
        for line in result.stdout.strip().split('\n'):
            if line.strip():
                parts = [p.strip() for p in line.split(',')]
                if len(parts) >= 3:
                    processes.append({
                        'pid': parts[0],
                        'name': parts[1],
                        'memory_mb': int(parts[2]) if parts[2].isdigit() else 0
                    })
        
        return processes
    except Exception as e:
        return []

def monitor_once():
    """한 번의 모니터링 수행"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    print(f"\n{'='*80}")
    print(f"GPU Monitoring - {timestamp}")
    print(f"{'='*80}")
    
    # nvidia-smi 정보
    smi_info = get_gpu_info_smi()
    if smi_info:
        for gpu in smi_info:
            print(f"\nGPU {gpu['index']}: {gpu['name']}")
            print(f"  GPU Utilization: {gpu['gpu_util']}%")
            print(f"  Memory Utilization: {gpu['memory_util']}%")
            if gpu['memory_total_mb'] > 0:
                print(f"  Memory Used: {gpu['memory_used_mb']} MB / {gpu['memory_total_mb']} MB ({gpu['memory_used_mb']/gpu['memory_total_mb']*100:.1f}%)")
            else:
                print(f"  Memory Used: {gpu['memory_used_mb']} MB / {gpu['memory_total_mb']} MB (N/A)")
            if gpu['power_w']:
                print(f"  Power: {gpu['power_w']:.1f} W")
            if gpu['temp_c']:
                print(f"  Temperature: {gpu['temp_c']}°C")
            print(f"  Compute Mode: {gpu['compute_mode']}")
    
    # nvidia-ml-py 정보 (더 상세)
    if NVML_AVAILABLE:
        nvml_info = get_gpu_info_nvml()
        if nvml_info:
            print(f"\nDetailed Info (NVML):")
            print(f"  GPU Utilization: {nvml_info['gpu_util']}%")
            print(f"  Memory Utilization: {nvml_info['memory_util']}%")
            print(f"  Memory: {nvml_info['memory_used_gb']:.2f} GB / {nvml_info['memory_total_gb']:.2f} GB ({nvml_info['memory_percent']:.1f}%)")
            if nvml_info['power_w']:
                print(f"  Power: {nvml_info['power_w']:.1f} W")
            if nvml_info['temp_c']:
                print(f"  Temperature: {nvml_info['temp_c']}°C")
            print(f"  Compute Processes: {nvml_info['processes']}")
            if nvml_info['process_details']:
                print(f"  Process Details:")
                for proc in nvml_info['process_details']:
                    print(f"    PID {proc['pid']}: {proc['memory_mb']:.0f} MB")
    
    # 프로세스 정보
    processes = get_processes_smi()
    if processes:
        print(f"\nGPU Processes:")
        for proc in processes:
            print(f"  PID {proc['pid']}: {proc['name']} ({proc['memory_mb']} MB)")
    else:
        print(f"\nNo compute processes running on GPU")
    
    # EXAONE 관련 프로세스 확인
    try:
        result = subprocess.run(
            ['ps', 'aux'],
            capture_output=True,
            text=True,
            timeout=5
        )
        if 'exaone' in result.stdout.lower() or 'extract' in result.stdout.lower():
            print(f"\n⚠️  EXAONE/Extraction processes found in ps output")
            for line in result.stdout.split('\n'):
                if 'exaone' in line.lower() or 'extract' in line.lower():
                    print(f"  {line[:100]}")
    except:
        pass

def main():
    """메인 함수"""
    if len(sys.argv) > 1:
        try:
            interval = float(sys.argv[1])
        except:
            interval = 5.0
    else:
        interval = 5.0
    
    print(f"Starting GPU monitoring (interval: {interval}s)")
    print("Press Ctrl+C to stop")
    
    try:
        while True:
            monitor_once()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\n\nMonitoring stopped.")

if __name__ == "__main__":
    main()

