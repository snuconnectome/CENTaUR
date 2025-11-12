#!/usr/bin/env python3
"""
간단한 GPU 작업을 수행하는 테스트 스크립트 (프로파일링용)
"""
import torch
import time

def test_gpu_operations():
    """GPU 연산 테스트"""
    print("Testing GPU profiling...")
    
    # GPU 사용 가능 여부 확인
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return
    
    device = torch.device("cuda:0")
    print(f"Using device: {device}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    
    # 간단한 행렬 연산
    print("\nPerforming matrix operations...")
    size = 2048
    
    # 행렬 생성 및 연산
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    
    # 여러 연산 수행
    for i in range(10):
        c = torch.matmul(a, b)
        c = torch.relu(c)
        c = torch.softmax(c, dim=0)
        torch.cuda.synchronize()
    
    print("Operations completed!")
    
    # 메모리 정보
    print(f"\nGPU Memory Allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
    print(f"GPU Memory Reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")

if __name__ == "__main__":
    test_gpu_operations()

