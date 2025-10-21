#!/usr/bin/env python3
"""
Train Qwen2.5-32B-Instruct with QLoRA (4-bit quantization)

GPU Memory: ~18-22GB (4-bit quantization with LoRA)
CPU Memory: ~50GB during loading
Model size: 32B parameters
Architecture: Standard transformer decoder (not MoE)

Strategy: BitsAndBytesConfig NF4 quantization + LoRA adapters
- Proven compatibility with PyTorch 2.6.0
- No exotic dependencies (MXFP4, Triton issues)
- Korean language support confirmed
"""

import argparse
import yaml
import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training


def load_config(config_path):
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_dataset(data_path, tokenizer, max_length, validation_split):
    """Prepare and tokenize dataset"""
    dataset = load_dataset('json', data_files=data_path, split='train')

    def tokenize_function(examples):
        texts = []
        for text, choice in zip(examples['text'], examples['choice']):
            # Append choice (A or B) to the prompt
            texts.append(text + (" A" if choice == 0 else " B"))

        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        # Copy input_ids to labels and set padding tokens to -100
        tokenized['labels'] = [
            [(token if token != tokenizer.pad_token_id else -100) for token in ids]
            for ids in tokenized['input_ids']
        ]
        return tokenized

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset",
    )

    if validation_split > 0:
        split = tokenized_dataset.train_test_split(
            test_size=validation_split,
            seed=42
        )
        return split['train'], split['test']

    return tokenized_dataset, None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to training config YAML')
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(config['misc']['seed'])

    print("=" * 80)
    print("Qwen2.5-32B-Instruct - QLoRA Training with 4-bit Quantization")
    print("=" * 80)
    print(f"Model: 32B parameters (standard transformer, not MoE)")
    print(f"Quantization: NF4 4-bit (BitsAndBytes)")
    print(f"GPU Configuration: 4 GPUs (multi-GPU training)")
    print(f"GPU Memory Expected: 18-22GB per GPU")
    print(f"Training: LoRA adapters with 4-bit base model")
    print(f"Korean Language: Confirmed support")
    print("=" * 80)

    # ============================================================================
    # Step 1: Load Tokenizer
    # ============================================================================
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        trust_remote_code=config['model'].get('trust_remote_code', True)
    )

    # Qwen tokenizer setup
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print("✅ Tokenizer loaded")
    print(f"   - Pad token: {tokenizer.pad_token} (ID: {tokenizer.pad_token_id})")
    print(f"   - EOS token: {tokenizer.eos_token} (ID: {tokenizer.eos_token_id})")

    assert tokenizer.pad_token_id is not None
    assert tokenizer.eos_token_id is not None

    # ============================================================================
    # Step 2: Prepare Dataset
    # ============================================================================
    print("\n[2/6] Preparing dataset...")
    train_dataset, eval_dataset = prepare_dataset(
        config['data']['train_file'],
        tokenizer,
        config['data']['max_seq_length'],
        config['data'].get('validation_split', 0.1)
    )
    print(f"✅ Train: {len(train_dataset)}, Eval: {len(eval_dataset) if eval_dataset else 0}")

    # ============================================================================
    # Step 3: Configure 4-bit Quantization
    # ============================================================================
    print("\n[3/6] Configuring 4-bit quantization...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",  # NormalFloat 4-bit
        bnb_4bit_compute_dtype=torch.bfloat16,  # Compute in bf16
        bnb_4bit_use_double_quant=True,  # Double quantization for extra memory savings
    )

    print("✅ Quantization config created")
    print("   - Method: NF4 4-bit")
    print("   - Compute dtype: bfloat16")
    print("   - Double quantization: enabled")

    # ============================================================================
    # Step 4: Load Model with 4-bit Quantization
    # ============================================================================
    print("\n[4/6] Loading model in 4-bit (5-10 minutes)...")
    print("🔧 BitsAndBytes quantization (proven compatibility)")

    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        quantization_config=bnb_config,
        device_map="auto",  # Automatic multi-GPU distribution
        trust_remote_code=config['model'].get('trust_remote_code', True),
        torch_dtype=torch.bfloat16,
    )

    print("✅ Model loaded in 4-bit")
    print(f"   - Memory footprint: {model.get_memory_footprint() / 1024**3:.2f} GB")
    print(f"   - Device map: auto (multi-GPU)")

    # ============================================================================
    # Step 5: Prepare for k-bit Training and Apply LoRA
    # ============================================================================
    print("\n[5/6] Configuring LoRA adapters...")

    # Enable gradient checkpointing and prepare for k-bit training
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)
    print("✅ Model prepared for k-bit training")

    lora_config = LoraConfig(
        r=config['model']['lora']['r'],
        lora_alpha=config['model']['lora']['lora_alpha'],
        target_modules=config['model']['lora']['target_modules'],
        lora_dropout=config['model']['lora']['lora_dropout'],
        bias=config['model']['lora']['bias'],
        task_type=config['model']['lora']['task_type'],
    )

    model = get_peft_model(model, lora_config)
    print("✅ LoRA adapters applied")
    print(f"   - Rank: {config['model']['lora']['r']}")
    print(f"   - Alpha: {config['model']['lora']['lora_alpha']}")
    print(f"   - Target modules: {config['model']['lora']['target_modules']}")

    # Print trainable parameters
    model.print_trainable_parameters()

    # ============================================================================
    # Step 6: Training Arguments and Trainer
    # ============================================================================
    print("\n[6/6] Initializing Trainer...")
    training_args = TrainingArguments(
        output_dir=config['output']['output_dir'],
        num_train_epochs=config['training']['num_train_epochs'],
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],
        learning_rate=config['training']['learning_rate'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        warmup_ratio=config['training']['warmup_ratio'],
        weight_decay=config['training']['weight_decay'],
        max_grad_norm=config['training']['max_grad_norm'],
        fp16=config['training'].get('fp16', False),
        bf16=config['training'].get('bf16', True),
        logging_steps=config['training']['logging_steps'],
        save_strategy=config['training']['save_strategy'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        eval_strategy=config['training']['eval_strategy'],
        eval_steps=config['training']['eval_steps'],
        optim=config['training'].get('optim', 'paged_adamw_8bit'),  # Memory-efficient optimizer
        report_to=config['misc']['report_to'],
        run_name=config['misc']['run_name'],
        seed=config['misc']['seed'],
        data_seed=config['misc']['data_seed'],
        gradient_checkpointing=True,  # Always enabled for QLoRA
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )
    print("✅ Trainer initialized")
    print(f"   - Effective batch size: {config['training']['per_device_train_batch_size']} × {config['training']['gradient_accumulation_steps']} × 4 GPUs = {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps'] * 4}")

    # ============================================================================
    # Training
    # ============================================================================
    print("\n" + "=" * 80)
    print("Starting QLoRA training...")
    print("Monitor GPU memory: nvidia-smi dmon -s mu")
    print("Expected: 18-22GB per GPU (4 GPUs total)")
    print("=" * 80)

    trainer.train()

    print("\n" + "=" * 80)
    print("Saving LoRA adapters...")
    print("=" * 80)
    model.save_pretrained(config['output']['output_dir'])
    tokenizer.save_pretrained(config['output']['output_dir'])
    print("✅ Training complete!")
    print(f"LoRA adapters saved to: {config['output']['output_dir']}")


if __name__ == "__main__":
    main()
