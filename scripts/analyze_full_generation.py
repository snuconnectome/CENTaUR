#!/usr/bin/env python3
"""
Analyze generation patterns on full 100-sample dataset
Reports generation bias, accuracy, and token distribution
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
import json
from collections import Counter

# Setup
base_model = "Qwen/Qwen2.5-32B-Instruct"
adapter = "/scratch/connectome/connectome1/ko-centaur/outputs/qwen25-32b-qlora"
dataset_path = "/scratch/connectome/connectome1/ko-centaur/data/choices13k_100.jsonl"

print("="*70)
print("FULL DATASET GENERATION ANALYSIS")
print("="*70)

print("\n1. Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(base_model)

print("\n2. Loading model...")
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

print("\n3. Loading dataset...")
with open(dataset_path, "r") as f:
    data = [json.loads(line) for line in f]

print(f"   Total samples: {len(data)}")

print("\n4. Analyzing generation patterns...")
print("   (This will take ~2 minutes)")

generated_tokens = []
true_labels = []

for idx, sample in enumerate(data):
    prompt = sample["text"]
    label = sample["choice"]

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        gen_outputs = model.generate(
            **inputs,
            max_new_tokens=1,
            temperature=0.0,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
        )

    generated_token_id = gen_outputs[0, -1].item()
    generated_text = tokenizer.decode(generated_token_id)

    generated_tokens.append((generated_token_id, generated_text))
    true_labels.append(label)

    if (idx + 1) % 10 == 0:
        print(f"   Progress: {idx + 1}/{len(data)}")

print("\n" + "="*70)
print("RESULTS")
print("="*70)

# Count ground truth distribution
true_a_count = sum(1 for l in true_labels if l == 0)
true_b_count = sum(1 for l in true_labels if l == 1)

print(f"\nGround Truth Distribution:")
print(f"  Choice A: {true_a_count}/{len(data)} ({true_a_count/len(data)*100:.1f}%)")
print(f"  Choice B: {true_b_count}/{len(data)} ({true_b_count/len(data)*100:.1f}%)")

# Analyze generation distribution
token_counter = Counter([text for _, text in generated_tokens])
print(f"\nGeneration Distribution:")
for token, count in token_counter.most_common():
    print(f"  '{token}': {count}/{len(data)} ({count/len(data)*100:.1f}%)")

# Calculate generation bias
gen_a_count = sum(1 for _, text in generated_tokens if 'A' in text)
gen_b_count = sum(1 for _, text in generated_tokens if 'B' in text)
bias_pct = max(gen_a_count, gen_b_count) / len(data) * 100

print(f"\n**Generation Bias**: {bias_pct:.1f}% toward {'B' if gen_b_count > gen_a_count else 'A'}")

# Analyze accuracy
correct = sum(1 for i, (_, text) in enumerate(generated_tokens)
              if (true_labels[i] == 1 and 'B' in text) or
                 (true_labels[i] == 0 and 'A' in text))
print(f"\nGeneration Accuracy: {correct}/{len(data)} ({correct/len(data)*100:.1f}%)")

# Conditional distributions
print(f"\nConditional Generation Patterns:")
true_a_gen_a = sum(1 for i, (_, text) in enumerate(generated_tokens) if true_labels[i] == 0 and 'A' in text)
true_a_gen_b = sum(1 for i, (_, text) in enumerate(generated_tokens) if true_labels[i] == 0 and 'B' in text)
true_b_gen_a = sum(1 for i, (_, text) in enumerate(generated_tokens) if true_labels[i] == 1 and 'A' in text)
true_b_gen_b = sum(1 for i, (_, text) in enumerate(generated_tokens) if true_labels[i] == 1 and 'B' in text)

print(f"  True A → Generated A: {true_a_gen_a}/{true_a_count} ({true_a_gen_a/true_a_count*100:.1f}%)")
print(f"  True A → Generated B: {true_a_gen_b}/{true_a_count} ({true_a_gen_b/true_a_count*100:.1f}%)")
print(f"  True B → Generated A: {true_b_gen_a}/{true_b_count} ({true_b_gen_a/true_b_count*100:.1f}%)")
print(f"  True B → Generated B: {true_b_gen_b}/{true_b_count} ({true_b_gen_b/true_b_count*100:.1f}%)")

print("\n" + "="*70)
print("INTERPRETATION")
print("="*70)

if bias_pct > 70:
    print(f"\n⚠️  CRITICAL: Strong generation bias detected ({bias_pct:.1f}%)")
    print(f"   The model is biased toward generating one token.")
    print(f"   This explains:")
    print(f"   - High feature similarity (95.25%): Most features represent same token")
    print(f"   - Worse than random NLL (0.7623 vs 0.6931): Model defaults to majority class")
    print(f"\n   Next steps:")
    print(f"   1. Test base model (no fine-tuning) to check if fine-tuning caused bias")
    print(f"   2. Test alternative prompt formats")
    print(f"   3. Try different model architecture (DeepSeek, EXAONE)")
elif bias_pct > 60:
    print(f"\n⚠️  Moderate generation bias detected ({bias_pct:.1f}%)")
    print(f"   The model shows preference for one token, but not extreme.")
    print(f"   Consider testing:")
    print(f"   1. Alternative prompt formats")
    print(f"   2. Base model comparison")
else:
    print(f"\n✓ Generation appears balanced ({bias_pct:.1f}% bias)")
    print(f"  High similarity (95.25%) likely has other causes.")
    print(f"  Next steps:")
    print(f"  1. Test single-step extraction (output_hidden_states in generate)")
    print(f"  2. Investigate prompt format effects on hidden states")

if correct / len(data) < 0.6:
    print(f"\n⚠️  Low generation accuracy ({correct/len(data)*100:.1f}%)")
    print(f"   Model is not learning choice patterns from prompts.")
    print(f"   This indicates model is not suitable for CENTaUR in current form.")

print("\n" + "="*70 + "\n")
