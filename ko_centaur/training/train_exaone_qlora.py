#!/usr/bin/env python3
"""
Test EXAONE QLoRA training on GPU 2
"""

import torch
import json
import os

# Use GPU 2 (free)
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

print("=" * 60)
print("EXAONE QLoRA Training Test on GPU 2")
print("=" * 60)

# Configuration
model_name = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
data_path = "/scratch/connectome/connectome1/ko-centaur/data/psych101_exaone_train.jsonl"
output_dir = "/scratch/connectome/connectome1/ko-centaur/models/test_gpu2"
max_length = 512
sample_size = 50
batch_size = 1
gradient_accumulation = 4
learning_rate = 2e-4
num_epochs = 1

os.makedirs(output_dir, exist_ok=True)

print(f"\nGPU: {torch.cuda.get_device_name(0)}")
print(f"Samples: {sample_size}, Max length: {max_length}")

# Tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
print(f"✅ Tokenizer loaded")

# Load data
data = []
with open(data_path, 'r') as f:
    for i, line in enumerate(f):
        if i >= sample_size:
            break
        data.append(json.loads(line))

print(f"✅ Loaded {len(data)} samples")

# Tokenize
def tokenize_function(examples):
    texts = []
    for messages in examples['messages']:
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        texts.append(text)
    
    return tokenizer(texts, truncation=True, max_length=max_length, padding="max_length")

dataset_dict = {"messages": [d['messages'] for d in data]}
dataset = Dataset.from_dict(dataset_dict)
tokenized_dataset = dataset.map(tokenize_function, batched=True, batch_size=50, remove_columns=dataset.column_names)

print(f"✅ Dataset tokenized")

# QLoRA config
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.float16,
    bnb_4bit_use_double_quant=True,
)

print(f"\nLoading model on GPU 2 (CUDA:0 after CUDA_VISIBLE_DEVICES)...")
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.float16,
)

print(f"✅ Model loaded ({model.get_memory_footprint() / 1024**3:.2f} GB)")

# LoRA
model.gradient_checkpointing_enable()
model = prepare_model_for_kbit_training(model)

lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Data collator
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Training
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=batch_size,
    gradient_accumulation_steps=gradient_accumulation,
    learning_rate=learning_rate,
    num_train_epochs=num_epochs,
    logging_steps=2,
    save_steps=25,
    fp16=True,
    optim="paged_adamw_8bit",
    report_to="none",
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator,
)

print(f"\n" + "=" * 60)
print("Starting training...")
print("=" * 60)

try:
    trainer.train()
    print(f"\n✅ Training complete!")
    
    model.save_pretrained(f"{output_dir}/final")
    tokenizer.save_pretrained(f"{output_dir}/final")
    print(f"✅ Saved to {output_dir}/final")
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()
