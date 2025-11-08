#!/usr/bin/env python3
"""
Download New Models for CENTaUR Experiments
============================================

Downloads recommended models from HuggingFace Hub to local directory.

Usage:
    python download_new_models.py --model exaone35
    python download_new_models.py --model gpt-oss-120b
    python download_new_models.py --model all --output /path/to/models
"""

import argparse
from pathlib import Path
from huggingface_hub import snapshot_download
import os

# Model configurations
MODELS = {
    "exaone35": {
        "repo_id": "LGAI-EXAONE/EXAONE-3.5-32B-Instruct",
        "name": "EXAONE-3.5-32B-Instruct",
        "size": "32B",
        "priority": 1,
        "description": "LG AI 최신 한국어 모델, 글로벌 상위권"
    },
    "gpt-oss-120b": {
        "repo_id": "openai/gpt-oss-120b",
        "name": "GPT-OSS-120B",
        "size": "120B",
        "priority": 1,
        "description": "OpenAI 최신 오픈소스, MoE 아키텍처"
    },
    "motif": {
        "repo_id": "moreh/Motif-102B",
        "name": "Motif-102B",
        "size": "102B",
        "priority": 2,
        "description": "한국어 특화 대규모 모델"
    },
    "gpt-neox": {
        "repo_id": "EleutherAI/gpt-neox-20b",
        "name": "GPT-NeoX-20B",
        "size": "20B",
        "priority": 2,
        "description": "EleutherAI 표준 기준 모델"
    },
    "polyglot-ko": {
        "repo_id": "EleutherAI/polyglot-ko-12.8b",
        "name": "Polyglot-Ko-12.8B",
        "size": "12.8B",
        "priority": 2,
        "description": "한국어 전용 중형 모델"
    },
    "gecko": {
        "repo_id": "sackoh/GECKO-7B",
        "name": "GECKO-7B",
        "size": "7B",
        "priority": 3,
        "description": "한영 이중언어 경량 모델"
    },
    "gpt-j": {
        "repo_id": "EleutherAI/gpt-j-6b",
        "name": "GPT-J-6B",
        "size": "6B",
        "priority": 3,
        "description": "EleutherAI 경량 모델"
    },
    "cerebras": {
        "repo_id": "cerebras/Cerebras-GPT-13B",
        "name": "Cerebras-GPT-13B",
        "size": "13B",
        "priority": 3,
        "description": "Cerebras 최적 학습 모델"
    }
}


def download_model(model_id: str, output_dir: Path, cache_dir: Path = None):
    """
    Download model from HuggingFace Hub.
    
    Args:
        model_id: Model identifier (key in MODELS dict)
        output_dir: Where to save the model
        cache_dir: HuggingFace cache directory (optional)
    """
    
    if model_id not in MODELS:
        print(f"❌ Unknown model: {model_id}")
        print(f"Available models: {', '.join(MODELS.keys())}")
        return False
    
    model_info = MODELS[model_id]
    repo_id = model_info["repo_id"]
    model_name = model_info["name"]
    
    print(f"\n{'='*70}")
    print(f"Downloading: {model_name}")
    print(f"{'='*70}")
    print(f"Repository:  {repo_id}")
    print(f"Size:        {model_info['size']}")
    print(f"Description: {model_info['description']}")
    print(f"Save to:     {output_dir / model_id}")
    print()
    
    try:
        # Create output directory
        output_path = output_dir / model_id
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Download with progress bar
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(output_path),
            cache_dir=str(cache_dir) if cache_dir else None,
            resume_download=True,
            local_dir_use_symlinks=False
        )
        
        print(f"\n✅ Successfully downloaded {model_name}")
        print(f"   Location: {output_path}")
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading {model_name}: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Download models for CENTaUR experiments",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Download priority 1 models (EXAONE-3.5, GPT-OSS-120B)
  python download_new_models.py --model exaone35
  python download_new_models.py --model gpt-oss-120b
  
  # Download all models
  python download_new_models.py --model all
  
  # Download to custom directory
  python download_new_models.py --model exaone35 --output /data/models
  
  # List available models
  python download_new_models.py --list
        """
    )
    
    parser.add_argument(
        "--model",
        type=str,
        help="Model to download (or 'all' for all models)"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="/home/connectome/connectome1/models",
        help="Output directory for models (default: /home/connectome/connectome1/models)"
    )
    
    parser.add_argument(
        "--cache",
        type=str,
        default=None,
        help="HuggingFace cache directory (default: None)"
    )
    
    parser.add_argument(
        "--list",
        action="store_true",
        help="List all available models and exit"
    )
    
    parser.add_argument(
        "--priority",
        type=int,
        choices=[1, 2, 3],
        help="Download all models of this priority level"
    )
    
    args = parser.parse_args()
    
    # List models
    if args.list:
        print("\n" + "="*70)
        print("Available Models")
        print("="*70)
        
        for priority in [1, 2, 3]:
            print(f"\n## Priority {priority} ##")
            for model_id, info in MODELS.items():
                if info["priority"] == priority:
                    print(f"  {model_id:15s} - {info['name']:25s} ({info['size']:>5s})")
                    print(f"                    {info['description']}")
        print()
        return
    
    output_dir = Path(args.output)
    cache_dir = Path(args.cache) if args.cache else None
    
    # Download models
    if args.model == "all":
        print("Downloading ALL models...")
        for model_id in MODELS.keys():
            download_model(model_id, output_dir, cache_dir)
            
    elif args.priority:
        print(f"Downloading all Priority {args.priority} models...")
        for model_id, info in MODELS.items():
            if info["priority"] == args.priority:
                download_model(model_id, output_dir, cache_dir)
                
    elif args.model:
        download_model(args.model, output_dir, cache_dir)
        
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
