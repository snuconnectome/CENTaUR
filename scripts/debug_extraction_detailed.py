#!/usr/bin/env python3
"""
Detailed debugging of extraction methodology
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

# Load first 5 samples
with open(dataset_path, "r") as f:
    samples = [json.loads(line) for line in f][:5]

print(f"\n{'='*70}")
print("DETAILED EXTRACTION DEBUG")
print(f"{'='*70}\n")

generated_tokens = []
features = []

for idx, sample in enumerate(samples):
    prompt = sample["text"]
    label = sample["choice"]

    print(f"\nSample {idx}: Choice = {'B' if label == 1 else 'A'}")
    print(f"Prompt (last 50 chars): ...{prompt[-50:]}")

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        # Generate ONE token
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )

        # Check generated token
        generated_token_id = gen_outputs[0, -1].item()
        generated_text = tokenizer.decode(generated_token_id)
        generated_tokens.append((generated_token_id, generated_text))

        print(f"  Generated token ID: {generated_token_id}")
        print(f"  Generated text: '{generated_text}'")

        # Forward pass with full sequence
        full_outputs = model(
            input_ids=gen_outputs,
            output_hidden_states=True
        )

        # Extract hidden state
        last_layer_hidden = full_outputs.hidden_states[-1]
        generated_token_hidden = last_layer_hidden[0, -1, :]
        features.append(generated_token_hidden.cpu())

        print(f"  Hidden state shape: {generated_token_hidden.shape}")
        print(f"  Hidden state mean: {generated_token_hidden.mean().item():.6f}")
        print(f"  Hidden state std: {generated_token_hidden.std().item():.6f}")

# Analysis
print(f"\n{'='*70}")
print("GENERATION ANALYSIS")
print(f"{'='*70}")
print("\nGenerated tokens:")
for idx, (token_id, text) in enumerate(generated_tokens):
    print(f"  Sample {idx}: ID={token_id:6d}, Text='{text}'")

unique_tokens = set(token_id for token_id, _ in generated_tokens)
print(f"\nUnique token count: {len(unique_tokens)} / {len(generated_tokens)}")

if len(unique_tokens) == 1:
    print("⚠️  WARNING: All samples generated the SAME token!")
    print("   This explains the high similarity.")
elif len(unique_tokens) == len(generated_tokens):
    print("✓ All samples generated different tokens")
else:
    print(f"✓ {len(unique_tokens)} unique tokens across {len(generated_tokens)} samples")

# Compute pairwise similarity
features_tensor = torch.stack(features)
features_norm = features_tensor / (features_tensor.norm(dim=1, keepdim=True) + 1e-8)
similarity = torch.mm(features_norm, features_norm.t())
off_diag = similarity - torch.eye(5)
avg_sim = off_diag.abs().mean().item()

print(f"\n{'='*70}")
print("FEATURE SIMILARITY ANALYSIS")
print(f"{'='*70}")
print(f"Average pairwise similarity: {avg_sim:.4f} ({avg_sim*100:.2f}%)")
print(f"\nPairwise similarity matrix:")
for i in range(5):
    row_str = "  "
    for j in range(5):
        row_str += f"{similarity[i,j].item():.4f}  "
    print(row_str)

print(f"\n{'='*70}")
