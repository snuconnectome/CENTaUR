#!/usr/bin/env python3
"""Test EXAONE-3.0-7.8B-Instruct model loading with QLoRA"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import os

print("=" * 60)
print("Ko-CENTaUR EXAONE Model Loading Test")
print("=" * 60)

# Model configuration
model_name = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
print(f"\nModel: {model_name}")
print(f"Cache directory: {os.environ.get('HF_HOME', 'Not set')}")
print(f"GPUs available: {torch.cuda.device_count()}")

# 4-bit quantization config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print("\n" + "=" * 60)
print("Loading tokenizer...")
print("=" * 60)

try:
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
    )
    print(f"✅ Tokenizer loaded successfully")
    print(f"   Vocab size: {len(tokenizer)}")
    
    # Test Korean tokenization
    test_text = "이것은 한국어 테스트입니다."
    tokens = tokenizer(test_text, return_tensors="pt")
    print(f"   Test tokenization: {test_text}")
    print(f"   Token count: {tokens['input_ids'].shape[1]}")
    
except Exception as e:
    print(f"❌ Tokenizer loading failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("Loading model (4-bit quantized)...")
print("=" * 60)

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"✅ Model loaded successfully")
    print(f"   Device map: {model.hf_device_map}")
    print(f"   Memory footprint: {model.get_memory_footprint() / 1024**3:.2f} GB")
    
except Exception as e:
    print(f"❌ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("Test generation...")
print("=" * 60)

try:
    # EXAONE uses chat template
    messages = [
        {"role": "system", "content": "You are EXAONE model from LG AI Research, a helpful assistant."},
        {"role": "user", "content": "한국의 수도는 어디인가요?"}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")
    
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            eos_token_id=tokenizer.eos_token_id,
            max_new_tokens=50,
            temperature=0.7,
            do_sample=True,
        )
    
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"Response: {response}")
    print(f"\n✅ Generation successful")
    
except Exception as e:
    print(f"❌ Generation failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("✅ All tests passed!")
print("=" * 60)
