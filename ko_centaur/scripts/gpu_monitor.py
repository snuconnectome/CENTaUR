#!/usr/bin/env python3
"""
Real-time GPU monitoring script
Usage: python gpu_monitor.py --interval 1 --output gpu_stats.csv
"""

import subprocess
import time
import argparse
import csv
from datetime import datetime

def get_gpu_stats():
    """Get GPU statistics using nvidia-smi"""
    try:
        result = subprocess.check_output([
            'nvidia-smi',
            '--query-gpu=timestamp,index,name,utilization.gpu,utilization.memory,memory.used,memory.total,power.draw,temperature.gpu,clocks.sm,clocks.mem',
            '--format=csv,noheader,nounits'
        ]).decode('utf-8')

        lines = result.strip().split('\n')
        stats = []
        for line in lines:
            parts = [p.strip() for p in line.split(',')]
            stats.append({
                'timestamp': parts[0],
                'gpu_id': parts[1],
                'gpu_name': parts[2],
                'gpu_util': float(parts[3]) if parts[3] != '[N/A]' else 0,
                'mem_util': float(parts[4]) if parts[4] != '[N/A]' else 0,
                'mem_used': float(parts[5]) if parts[5] != '[N/A]' else 0,
                'mem_total': float(parts[6]) if parts[6] != '[N/A]' else 0,
                'power_draw': float(parts[7]) if parts[7] != '[N/A]' else 0,
                'temperature': float(parts[8]) if parts[8] != '[N/A]' else 0,
                'clock_sm': float(parts[9]) if parts[9] != '[N/A]' else 0,
                'clock_mem': float(parts[10]) if parts[10] != '[N/A]' else 0,
            })
        return stats
    except Exception as e:
        print(f"Error getting GPU stats: {e}")
        return []

def get_process_stats():
    """Get GPU process statistics"""
    try:
        result = subprocess.check_output([
            'nvidia-smi',
            '--query-compute-apps=pid,used_memory',
            '--format=csv,noheader,nounits'
        ]).decode('utf-8')

        lines = result.strip().split('\n')
        processes = []
        for line in lines:
            if line:
                parts = [p.strip() for p in line.split(',')]
                processes.append({
                    'pid': parts[0],
                    'mem_used': float(parts[1]) if parts[1] != '[N/A]' else 0
                })
        return processes
    except Exception as e:
        print(f"Error getting process stats: {e}")
        return []

def monitor(interval=1, output_file=None, duration=None):
    """Monitor GPU stats continuously"""

    fieldnames = ['timestamp', 'gpu_id', 'gpu_name', 'gpu_util', 'mem_util',
                  'mem_used', 'mem_total', 'power_draw', 'temperature',
                  'clock_sm', 'clock_mem', 'process_count', 'process_mem']

    csv_file = None
    csv_writer = None

    if output_file:
        csv_file = open(output_file, 'w', newline='')
        csv_writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        csv_writer.writeheader()
        print(f"📝 Logging to {output_file}")

    start_time = time.time()

    try:
        while True:
            stats = get_gpu_stats()
            processes = get_process_stats()

            for gpu_stat in stats:
                # Add process info
                gpu_stat['process_count'] = len(processes)
                gpu_stat['process_mem'] = sum(p['mem_used'] for p in processes)

                # Print to console
                print(f"\n[{gpu_stat['timestamp']}] GPU {gpu_stat['gpu_id']}: {gpu_stat['gpu_name']}")
                print(f"  GPU Util:  {gpu_stat['gpu_util']:>5.1f}%")
                print(f"  Mem Util:  {gpu_stat['mem_util']:>5.1f}%")
                print(f"  Memory:    {gpu_stat['mem_used']:>6.0f} / {gpu_stat['mem_total']:.0f} MB")
                print(f"  Power:     {gpu_stat['power_draw']:>5.1f} W")
                print(f"  Temp:      {gpu_stat['temperature']:>5.1f}°C")
                print(f"  SM Clock:  {gpu_stat['clock_sm']:>6.0f} MHz")
                print(f"  Processes: {gpu_stat['process_count']} ({gpu_stat['process_mem']:.0f} MB)")

                # Write to CSV
                if csv_writer:
                    csv_writer.writerow(gpu_stat)
                    csv_file.flush()

            # Check duration
            if duration and (time.time() - start_time) >= duration:
                print(f"\n✅ Monitoring completed ({duration}s)")
                break

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n🛑 Monitoring stopped")
    finally:
        if csv_file:
            csv_file.close()
            print(f"✅ Saved to {output_file}")

def analyze(csv_file):
    """Analyze GPU statistics from CSV file"""
    import pandas as pd
    import numpy as np

    df = pd.read_csv(csv_file)

    print("\n" + "="*60)
    print("GPU Statistics Summary")
    print("="*60)

    for gpu_id in df['gpu_id'].unique():
        gpu_data = df[df['gpu_id'] == gpu_id]

        print(f"\n📊 GPU {gpu_id}: {gpu_data['gpu_name'].iloc[0]}")
        print(f"  Duration: {len(gpu_data)} samples")

        print(f"\n  GPU Utilization:")
        print(f"    Mean:   {gpu_data['gpu_util'].mean():>5.1f}%")
        print(f"    Max:    {gpu_data['gpu_util'].max():>5.1f}%")
        print(f"    Min:    {gpu_data['gpu_util'].min():>5.1f}%")
        print(f"    Std:    {gpu_data['gpu_util'].std():>5.1f}%")

        print(f"\n  Memory Utilization:")
        print(f"    Mean:   {gpu_data['mem_util'].mean():>5.1f}%")
        print(f"    Max:    {gpu_data['mem_util'].max():>5.1f}%")
        print(f"    Peak:   {gpu_data['mem_used'].max():>6.0f} / {gpu_data['mem_total'].iloc[0]:.0f} MB")

        print(f"\n  Power Consumption:")
        print(f"    Mean:   {gpu_data['power_draw'].mean():>5.1f} W")
        print(f"    Max:    {gpu_data['power_draw'].max():>5.1f} W")

        print(f"\n  Temperature:")
        print(f"    Mean:   {gpu_data['temperature'].mean():>5.1f}°C")
        print(f"    Max:    {gpu_data['temperature'].max():>5.1f}°C")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GPU Monitoring Tool')
    parser.add_argument('--interval', type=float, default=1.0,
                        help='Sampling interval in seconds (default: 1.0)')
    parser.add_argument('--output', type=str,
                        help='Output CSV file path')
    parser.add_argument('--duration', type=int,
                        help='Monitoring duration in seconds (default: infinite)')
    parser.add_argument('--analyze', type=str,
                        help='Analyze existing CSV file')

    args = parser.parse_args()

    if args.analyze:
        analyze(args.analyze)
    else:
        monitor(interval=args.interval, output_file=args.output, duration=args.duration)
