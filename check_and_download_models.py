#!/usr/bin/env python3
"""
모델 다운로드 상태 확인 및 백그라운드 다운로드 스크립트
"""

import os
import sys
import subprocess
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from huggingface_hub import snapshot_download
import torch

models_to_check = {
    'EXAONE-3.5-32B': {
        'model_id': 'LGAI-EXAONE/EXAONE-3.5-32B-Instruct',
        'trust_remote_code': True,
    },
    'GPT-OSS-20B': {
        'model_id': 'openai/gpt-oss-20b',
        'trust_remote_code': False,
    },
    'Kimi K2': {
        'model_id': 'moonshot-ai/Kimi-K2-Instruct',
        'trust_remote_code': False,
    },
}

def check_model_downloaded(model_id):
    """모델 다운로드 여부 확인"""
    cache_dir = os.path.expanduser("~/.cache/huggingface/hub")
    model_path = model_id.replace("/", "--")
    full_path = os.path.join(cache_dir, f"models--{model_path}")
    
    if os.path.exists(full_path):
        # 파일 크기 확인
        total_size = 0
        for root, dirs, files in os.walk(full_path):
            for file in files:
                filepath = os.path.join(root, file)
                if os.path.exists(filepath):
                    total_size += os.path.getsize(filepath)
        
        # 최소 크기 확인 (1MB 이상)
        if total_size > 1024 * 1024:
            return True, total_size / (1024**3)  # GB
    return False, 0

def download_model(name, model_id, trust_remote_code=False):
    """모델 다운로드"""
    print(f"\n{'='*60}")
    print(f"{name} 다운로드 시작")
    print(f"Model ID: {model_id}")
    print(f"{'='*60}\n")
    
    try:
        # Tokenizer만 먼저 다운로드 (빠름)
        print("1. Tokenizer 다운로드 중...")
        tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            trust_remote_code=trust_remote_code
        )
        print(f"   ✅ Tokenizer 다운로드 완료 (vocab_size: {tokenizer.vocab_size})")
        
        # 모델 파일 다운로드 (snapshot_download 사용)
        print("\n2. 모델 파일 다운로드 중...")
        print("   (시간이 오래 걸릴 수 있습니다)")
        
        snapshot_download(
            repo_id=model_id,
            cache_dir=None,  # 기본 캐시 디렉토리 사용
            ignore_patterns=["*.safetensors.index.json", "*.h5", "*.ot"],  # 불필요한 파일 제외
        )
        
        print(f"   ✅ {name} 다운로드 완료")
        return True
        
    except Exception as e:
        print(f"   ❌ 다운로드 실패: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    print("=" * 60)
    print("모델 다운로드 상태 확인 및 다운로드")
    print("=" * 60)
    print()
    
    # 다운로드 필요한 모델 확인
    models_to_download = []
    
    for name, config in models_to_check.items():
        model_id = config['model_id']
        is_downloaded, size = check_model_downloaded(model_id)
        
        if is_downloaded:
            print(f"✅ {name}: 이미 다운로드됨 ({size:.2f} GB)")
        else:
            print(f"❌ {name}: 다운로드 필요")
            models_to_download.append((name, config))
    
    print()
    
    # 다운로드 시작
    if models_to_download:
        print(f"다운로드할 모델: {len(models_to_download)}개\n")
        
        for name, config in models_to_download:
            success = download_model(
                name,
                config['model_id'],
                config.get('trust_remote_code', False)
            )
            
            if success:
                print(f"\n✅ {name} 다운로드 완료\n")
            else:
                print(f"\n❌ {name} 다운로드 실패\n")
    else:
        print("✅ 모든 모델이 이미 다운로드되어 있습니다!")
    
    print("=" * 60)
    print("완료")
    print("=" * 60)

if __name__ == "__main__":
    main()

