#!/usr/bin/env python3
"""
Kimi 모델 다운로드 스크립트 (올바른 모델 ID 사용)
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import time

def download_kimi_k2():
    """Kimi-K2-Instruct 다운로드 (올바른 모델 ID)"""
    model_id = "moonshotai/Kimi-K2-Instruct"
    print(f"\n{'='*60}")
    print(f"Kimi-K2-Instruct 다운로드 시작")
    print(f"Model ID: {model_id}")
    print(f"{'='*60}\n")
    
    try:
        # Tokenizer 먼저
        print("1. Tokenizer 다운로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print(f"   ✅ Tokenizer 완료 (vocab_size: {tokenizer.vocab_size})")
        
        # 모델 파일 다운로드
        print("\n2. 모델 파일 다운로드 중...")
        print("   ⚠️  매우 큰 모델입니다 (1T 파라미터)")
        print("   예상 시간: 수 시간 소요될 수 있습니다")
        
        snapshot_download(
            repo_id=model_id,
            resume_download=True,
            local_files_only=False,
        )
        print("   ✅ Kimi-K2-Instruct 다운로드 완료")
        return True
    except Exception as e:
        print(f"   ❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def download_kimi_linear():
    """Kimi-Linear-48B-A3B-Instruct 다운로드 (더 작은 대안)"""
    model_id = "moonshotai/Kimi-Linear-48B-A3B-Instruct"
    print(f"\n{'='*60}")
    print(f"Kimi-Linear-48B-A3B-Instruct 다운로드 시작")
    print(f"Model ID: {model_id}")
    print(f"{'='*60}\n")
    
    try:
        # Tokenizer 먼저
        print("1. Tokenizer 다운로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
        print(f"   ✅ Tokenizer 완료 (vocab_size: {tokenizer.vocab_size})")
        
        # 모델 파일 다운로드
        print("\n2. 모델 파일 다운로드 중...")
        print("   크기: 49B 파라미터 (Kimi-K2보다 작음)")
        
        snapshot_download(
            repo_id=model_id,
            resume_download=True,
            local_files_only=False,
        )
        print("   ✅ Kimi-Linear-48B-A3B-Instruct 다운로드 완료")
        return True
    except Exception as e:
        print(f"   ❌ 오류: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("Kimi 모델 다운로드 (올바른 모델 ID)")
    print("=" * 60)
    print(f"시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Kimi-K2-Instruct 다운로드
    k2_success = download_kimi_k2()
    
    # Kimi-Linear 다운로드 (선택적, 더 작은 대안)
    # linear_success = download_kimi_linear()
    
    print("\n" + "=" * 60)
    print("다운로드 완료")
    print("=" * 60)
    print(f"Kimi-K2-Instruct: {'✅ 성공' if k2_success else '❌ 실패'}")
    print(f"종료 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

