#!/usr/bin/env python3
"""
모델 다운로드 스크립트 (백그라운드 실행용)
GPT-OSS-20B와 Kimi K2 다운로드
"""

import os
import sys
from pathlib import Path
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer
import time

def download_gpt_oss():
    """GPT-OSS-20B 다운로드"""
    model_id = "openai/gpt-oss-20b"
    print(f"\n{'='*60}")
    print(f"GPT-OSS-20B 다운로드 시작")
    print(f"{'='*60}\n")
    
    try:
        # Tokenizer 먼저
        print("Tokenizer 다운로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"✅ Tokenizer 완료 (vocab_size: {tokenizer.vocab_size})")
        
        # 모델 파일 다운로드
        print("\n모델 파일 다운로드 중...")
        snapshot_download(
            repo_id=model_id,
            resume_download=True,
            local_files_only=False,
        )
        print("✅ GPT-OSS-20B 다운로드 완료")
        return True
    except Exception as e:
        print(f"❌ 오류: {e}")
        return False

def download_kimi_k2():
    """Kimi K2 다운로드 (인증 필요할 수 있음)"""
    model_id = "moonshot-ai/Kimi-K2-Instruct"
    print(f"\n{'='*60}")
    print(f"Kimi K2 다운로드 시작")
    print(f"{'='*60}\n")
    
    try:
        # Tokenizer 먼저
        print("Tokenizer 다운로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        print(f"✅ Tokenizer 완료")
        
        # 모델 파일 다운로드
        print("\n모델 파일 다운로드 중...")
        snapshot_download(
            repo_id=model_id,
            resume_download=True,
            local_files_only=False,
        )
        print("✅ Kimi K2 다운로드 완료")
        return True
    except Exception as e:
        print(f"⚠️  Kimi K2 다운로드 실패: {e}")
        print("   인증이 필요할 수 있습니다.")
        print("   HuggingFace 로그인: huggingface-cli login")
        return False

def main():
    print("=" * 60)
    print("모델 다운로드 시작")
    print("=" * 60)
    print(f"시작 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # GPT-OSS-20B 다운로드
    gpt_oss_success = download_gpt_oss()
    
    # Kimi K2 다운로드
    kimi_success = download_kimi_k2()
    
    print("\n" + "=" * 60)
    print("다운로드 완료")
    print("=" * 60)
    print(f"GPT-OSS-20B: {'✅ 성공' if gpt_oss_success else '❌ 실패'}")
    print(f"Kimi K2: {'✅ 성공' if kimi_success else '❌ 실패'}")
    print(f"종료 시간: {time.strftime('%Y-%m-%d %H:%M:%S')}")

if __name__ == "__main__":
    main()

