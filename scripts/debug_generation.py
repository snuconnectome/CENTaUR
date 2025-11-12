#!/usr/bin/env python3
"""
Debug script to verify generation and hidden state extraction
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json

# Setup
base_model = "Qwen/Qwen2.5-32B-Instruct"
adapter = "/scratch/connectome/connectome1/ko-centaur/outputs/qwen25-32b-qlora"
dataset_path = "/scratch/connectome/connectome1/ko-centaur/data/choices13k_100.jsonl"

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model)

print("Loading model...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True
)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    quantization_config=bnb_config
)

model = PeftModel.from_pretrained(model, adapter)
model.eval()

# Load first sample
with open(dataset_path, "r") as f:
    sample = json.loads(f.readline())

prompt = sample["text"]
label = sample["choice"]

print(f"\n{'='*70}")
print(f"Sample 0: Expected choice = {'B' if label == 1 else 'A'}")
print(f"{'='*70}")
print(f"Prompt (last 100 chars): ...{prompt[-100:]}")

# Tokenize
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
print(f"\nPrompt length: {inputs['input_ids'].shape[1]} tokens")

# Generate with hidden states
with torch.no_grad():
    gen_outputs = model.generate(
        **inputs,
        max_new_tokens=1,
        temperature=0.0,
        do_sample=False,
        output_hidden_states=True,
        return_dict_in_generate=True
    )

# Check what was generated
generated_ids = gen_outputs.sequences[0]  # Full sequence including prompt
generated_token_id = generated_ids[-1].item()  # Last token (the one we generated)
generated_text = tokenizer.decode(generated_token_id)

print(f"\nGenerated token ID: {generated_token_id}")
print(f"Generated text: '{generated_text}'")
print(f"Full generated sequence: {tokenizer.decode(generated_ids)[-50:]}")

# Check hidden states structure
print(f"\n{'='*70}")
print(f"HIDDEN STATES STRUCTURE")
print(f"{'='*70}")
print(f"Type of gen_outputs.hidden_states: {type(gen_outputs.hidden_states)}")
print(f"Length (num generation steps): {len(gen_outputs.hidden_states)}")

if len(gen_outputs.hidden_states) > 0:
    first_step = gen_outputs.hidden_states[0]
    print(f"\nFirst generation step:")
    print(f"  Type: {type(first_step)}")
    print(f"  Length (num layers): {len(first_step)}")

    last_layer = first_step[-1]
    print(f"\n  Last layer:")
    print(f"    Shape: {last_layer.shape}")
    print(f"    (batch_size, seq_len, hidden_dim)")

    # Extract hidden state
    generated_token_hidden = last_layer[0, -1, :]
    print(f"\n  Extracted hidden state shape: {generated_token_hidden.shape}")
    print(f"  Mean: {generated_token_hidden.mean().item():.6f}")
    print(f"  Std: {generated_token_hidden.std().item():.6f}")

print(f"\n{'='*70}")
print("VERIFICATION:")
print(f"✓ Generated {len(gen_outputs.hidden_states)} token(s)")
print(f"✓ Token: '{generated_text}' (ID: {generated_token_id})")
print(f"✓ Hidden state extracted from position -1 of sequence")
print(f"{'='*70}\n")
