#!/usr/bin/env python3
"""
NGC PyTorch GPU 성능 벤치마크
"""

import torch
import time
import numpy as np
from datetime import datetime

print("=" * 60)
print("NGC PyTorch GPU 성능 벤치마크")
print("=" * 60)
print(f"시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# GPU 정보
print("1. GPU 정보:")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   CUDA capability: {torch.cuda.get_device_capability(0)}")
    print(f"   GPU 메모리: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
    print(f"   PyTorch version: {torch.__version__}")
print()

# 벤치마크 1: 기본 행렬 연산
print("2. 기본 행렬 연산 벤치마크:")
if torch.cuda.is_available():
    sizes = [1000, 5000, 10000]
    for size in sizes:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        x = torch.randn(size, size, device='cuda')
        y = torch.randn(size, size, device='cuda')
        
        # 워밍업
        for _ in range(10):
            _ = torch.matmul(x, y)
        torch.cuda.synchronize()
        
        # 측정
        start = time.time()
        for _ in range(100):
            z = torch.matmul(x, y)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        gflops = (2 * size ** 3 * 100) / elapsed / 1e9
        print(f"   {size}x{size} 행렬 곱셈: {elapsed/100*1000:.2f} ms/iter, {gflops:.2f} GFLOPS")
print()

# 벤치마크 2: 메모리 대역폭
print("3. 메모리 대역폭 테스트:")
if torch.cuda.is_available():
    sizes = [1000, 10000, 100000]
    for size in sizes:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        x = torch.randn(size, size, device='cuda')
        
        # 워밍업
        for _ in range(10):
            _ = x * 2
        torch.cuda.synchronize()
        
        # 측정
        start = time.time()
        for _ in range(1000):
            y = x * 2
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        bytes_transferred = size * size * 4 * 2 * 1000  # float32, read+write
        bandwidth = bytes_transferred / elapsed / 1e9
        print(f"   {size}x{size} 텐서 연산: {bandwidth:.2f} GB/s")
print()

# 벤치마크 3: 실제 모델 추론 시뮬레이션
print("4. 모델 추론 시뮬레이션:")
if torch.cuda.is_available():
    # Feature extraction 시뮬레이션 (5120차원 hidden state)
    batch_sizes = [1, 10, 100]
    hidden_dim = 5120
    
    for batch_size in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        
        # 입력 시뮬레이션
        input_ids = torch.randint(0, 1000, (batch_size, 128), device='cuda')
        hidden_states = torch.randn(batch_size, 128, hidden_dim, device='cuda')
        
        # 워밍업
        for _ in range(10):
            _ = hidden_states.mean(dim=1)  # Last token hidden state
        torch.cuda.synchronize()
        
        # 측정
        start = time.time()
        for _ in range(100):
            features = hidden_states.mean(dim=1)
        torch.cuda.synchronize()
        elapsed = time.time() - start
        
        print(f"   Batch size {batch_size}: {elapsed/100*1000:.2f} ms/iter")
print()

# GPU 메모리 사용량
print("5. GPU 메모리 사용량:")
if torch.cuda.is_available():
    allocated = torch.cuda.memory_allocated(0) / 1e9
    reserved = torch.cuda.memory_reserved(0) / 1e9
    total = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"   할당됨: {allocated:.2f} GB")
    print(f"   예약됨: {reserved:.2f} GB")
    print(f"   전체: {total:.2f} GB")
    print(f"   사용률: {reserved/total*100:.1f}%")
print()

print("=" * 60)
print("벤치마크 완료")
print("=" * 60)

