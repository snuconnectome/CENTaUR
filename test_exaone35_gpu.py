#!/usr/bin/env python3
"""
EXAONE-3.5-32B GPU 사용 테스트 스크립트
작은 샘플로 모델 로딩 및 feature extraction 테스트
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from pathlib import Path
import sys

print("=" * 60)
print("EXAONE-3.5-32B GPU 사용 테스트")
print("=" * 60)

# 1. GPU 확인
print("\n1. GPU 상태 확인:")
print(f"   CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   CUDA version: {torch.version.cuda}")
    print(f"   CUDA capability: {torch.cuda.get_device_capability(0)}")
    print(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    device = torch.device('cuda')
else:
    print("   ❌ CUDA 사용 불가")
    device = torch.device('cpu')

# 2. 모델 정보
model_id = "LGAI-EXAONE/EXAONE-3.5-32B-Instruct"
print(f"\n2. 모델 정보:")
print(f"   Model ID: {model_id}")
print(f"   Device: {device}")

# 3. Tokenizer 로딩 테스트
print(f"\n3. Tokenizer 로딩 테스트:")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    print(f"   ✅ Tokenizer 로딩 성공")
    print(f"   Vocab size: {tokenizer.vocab_size}")
except Exception as e:
    print(f"   ❌ Tokenizer 로딩 실패: {e}")
    sys.exit(1)

# 4. 작은 모델로 GPU 메모리 테스트
print(f"\n4. GPU 메모리 사용 테스트:")
try:
    # 간단한 텐서로 GPU 메모리 확인
    test_tensor = torch.randn(1000, 1000, device=device)
    print(f"   ✅ GPU 메모리 할당 성공")
    print(f"   할당된 메모리: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
    del test_tensor
    torch.cuda.empty_cache()
except Exception as e:
    print(f"   ❌ GPU 메모리 테스트 실패: {e}")

# 5. Quantization 설정 테스트
print(f"\n5. Quantization 설정 테스트:")
try:
    from transformers import BitsAndBytesConfig
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    print(f"   ✅ BitsAndBytesConfig 생성 성공")
    print(f"   - 4-bit NF4 quantization")
    print(f"   - Compute dtype: bfloat16")
    print(f"   - Double quantization: True")
except Exception as e:
    print(f"   ❌ Quantization 설정 실패: {e}")

# 6. 실제 모델 로딩 테스트 (선택적 - 시간이 오래 걸릴 수 있음)
print(f"\n6. 모델 로딩 테스트 (선택적):")
print(f"   ⚠️  이 단계는 시간이 오래 걸릴 수 있습니다 (모델 다운로드 + 로딩)")
print(f"   계속하시겠습니까? (y/n): ", end="")
# 자동으로 'n' 선택 (실제 테스트 시에는 'y' 입력)
response = 'n'
print(response)

if response.lower() == 'y':
    try:
        print(f"   모델 다운로드 및 로딩 중...")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            torch_dtype=torch.bfloat16
        )
        print(f"   ✅ 모델 로딩 성공")
        print(f"   GPU 메모리 사용: {torch.cuda.memory_allocated(0) / 1e9:.2f} GB")
        
        # 간단한 추론 테스트
        print(f"\n7. 추론 테스트:")
        test_text = "안녕하세요"
        inputs = tokenizer(test_text, return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        print(f"   ✅ 추론 테스트 성공")
        print(f"   Output shape: {outputs.logits.shape}")
        
    except Exception as e:
        print(f"   ❌ 모델 로딩 실패: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"   모델 로딩 테스트 건너뜀")

print(f"\n{'=' * 60}")
print("테스트 완료")
print(f"{'=' * 60}")

