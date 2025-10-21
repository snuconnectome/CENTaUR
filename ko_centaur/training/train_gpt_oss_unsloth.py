#!/usr/bin/env python3
"""
Train GPT-OSS-20B with Unsloth QLoRA
Method: 4-bit quantization + LoRA adapters with Unsloth optimization

Usage:
    python training/train_gpt_oss_unsloth.py \
        --config configs/training_gpt_oss.yaml
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict

import torch
import yaml
from datasets import Dataset, load_dataset
from transformers import (
    AutoTokenizer,
    BitsAndBytesConfig,
    TrainingArguments,
    set_seed,
)

# Unsloth imports
try:
    from unsloth import FastLanguageModel
    from unsloth import is_bfloat16_supported
    from trl import SFTTrainer
    UNSLOTH_AVAILABLE = True
except ImportError:
    print("Warning: Unsloth not available. Please install: pip install unsloth")
    UNSLOTH_AVAILABLE = False


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_dataset(
    data_path: str,
    max_length: int = 512,
    validation_split: float = 0.1
):
    """
    Load the choices13k dataset

    Format: {"text": "Which option would you choose? ...", "choice": 0/1}
    """
    # Load dataset from JSONL
    dataset = load_dataset('json', data_files=data_path, split='train')

    # Split into train and validation
    if validation_split > 0:
        split = dataset.train_test_split(
            test_size=validation_split,
            seed=42
        )
        return split['train'], split['test']
    else:
        return dataset, None


def formatting_func(examples):
    """Format examples for SFT training"""
    return {"text": examples["text"]}


def main():
    parser = argparse.ArgumentParser(description='Train GPT-OSS-20B with Unsloth QLoRA')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_gpt_oss.yaml',
        help='Path to training configuration YAML'
    )
    args = parser.parse_args()

    if not UNSLOTH_AVAILABLE:
        print("Error: Unsloth library is required. Install with: pip install unsloth")
        sys.exit(1)

    # Load configuration
    config = load_config(args.config)

    # Set seed for reproducibility
    set_seed(config['misc']['seed'])

    # Initialize tokenizer
    print(f"\n{'='*60}")
    print("Loading tokenizer...")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        revision=config['model']['revision'],
        trust_remote_code=config['model']['trust_remote_code']
    )

    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Prepare dataset
    print(f"\n{'='*60}")
    print("Preparing dataset...")
    print(f"{'='*60}")
    train_dataset, eval_dataset = prepare_dataset(
        data_path=config['data']['train_file'],
        max_length=config['data']['max_seq_length'],
        validation_split=config['data']['validation_split']
    )

    print(f"Train samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Validation samples: {len(eval_dataset)}")

    # Load model with Unsloth optimization
    print(f"\n{'='*60}")
    print("Loading GPT-OSS-20B with Unsloth QLoRA...")
    print(f"{'='*60}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['model']['name'],
        max_seq_length=config['lora']['max_seq_length'],
        dtype=None,  # Auto-detect
        load_in_4bit=config['model']['load_in_4bit'],
        # Trust remote code
        trust_remote_code=config['model']['trust_remote_code'],
    )

    # Add LoRA adapters with Unsloth
    print(f"\n{'='*60}")
    print("Adding LoRA adapters...")
    print(f"LoRA rank: {config['lora']['r']}")
    print(f"LoRA alpha: {config['lora']['lora_alpha']}")
    print(f"Target modules: {config['lora']['target_modules']}")
    print(f"{'='*60}")

    model = FastLanguageModel.get_peft_model(
        model,
        r=config['lora']['r'],
        lora_alpha=config['lora']['lora_alpha'],
        lora_dropout=config['lora']['lora_dropout'],
        target_modules=config['lora']['target_modules'],
        bias=config['lora']['bias'],
        use_gradient_checkpointing=config['lora']['use_gradient_checkpointing'],
        random_state=config['lora']['random_state'],
        use_rslora=False,  # Rank stabilized LoRA
        loftq_config=None,  # LoftQ quantization
    )

    # Training arguments
    print(f"\n{'='*60}")
    print("Setting up training arguments...")
    print(f"{'='*60}")

    training_args = TrainingArguments(
        # Output
        output_dir=config['output']['output_dir'],
        overwrite_output_dir=config['output']['overwrite_output_dir'],

        # Training hyperparameters
        per_device_train_batch_size=config['training']['per_device_train_batch_size'],
        per_device_eval_batch_size=config['training']['per_device_eval_batch_size'],
        gradient_accumulation_steps=config['training']['gradient_accumulation_steps'],

        learning_rate=config['training']['learning_rate'],
        lr_scheduler_type=config['training']['lr_scheduler_type'],
        warmup_ratio=config['training']['warmup_ratio'],

        num_train_epochs=config['training']['num_train_epochs'],
        max_steps=config['training']['max_steps'],

        # Optimization
        optim=config['training']['optim'],
        weight_decay=config['training']['weight_decay'],
        adam_beta1=config['training']['adam_beta1'],
        adam_beta2=config['training']['adam_beta2'],
        adam_epsilon=config['training']['adam_epsilon'],
        max_grad_norm=config['training']['max_grad_norm'],

        # Precision
        fp16=config['training']['fp16'] and not is_bfloat16_supported(),
        bf16=config['training']['bf16'] and is_bfloat16_supported(),

        # Logging
        logging_steps=config['training']['logging_steps'],
        report_to=config['misc']['report_to'],
        run_name=config['misc']['run_name'],

        # Checkpointing
        save_strategy=config['training']['save_strategy'],
        save_steps=config['training']['save_steps'],
        save_total_limit=config['training']['save_total_limit'],
        save_safetensors=config['output']['save_safetensors'],

        # Evaluation
        evaluation_strategy=config['training']['evaluation_strategy'],
        eval_steps=config['training']['eval_steps'],

        # Data loading
        dataloader_num_workers=config['data']['dataloader_num_workers'],
        dataloader_pin_memory=config['data']['dataloader_pin_memory'],

        # Reproducibility
        seed=config['misc']['seed'],
        data_seed=config['misc']['data_seed'],
    )

    # Initialize SFTTrainer (Supervised Fine-Tuning)
    print(f"\n{'='*60}")
    print("Initializing SFTTrainer with Unsloth...")
    print(f"{'='*60}")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        dataset_text_field="text",
        max_seq_length=config['lora']['max_seq_length'],
        dataset_num_proc=4,
        packing=False,  # Can make training faster
        args=training_args,
    )

    # Resume from checkpoint if specified
    resume_from = config['output'].get('resume_from_checkpoint')

    # Train
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"Model: GPT-OSS-20B (20B parameters)")
    print(f"Method: Unsloth QLoRA (4-bit + LoRA)")
    print(f"Memory: ~14GB per GPU")
    print(f"Effective batch size: {config['training']['per_device_train_batch_size']} * {config['training']['gradient_accumulation_steps']}")
    print(f"{'='*60}\n")

    trainer.train(resume_from_checkpoint=resume_from)

    # Save LoRA adapters
    print(f"\n{'='*60}")
    print("Saving LoRA adapters...")
    print(f"{'='*60}")

    # Save adapters only (not full model)
    model.save_pretrained(config['output']['output_dir'])
    tokenizer.save_pretrained(config['output']['output_dir'])

    # Optionally merge and save full model
    if config['output'].get('merge_adapter_on_save', False):
        print("Merging LoRA adapters with base model...")
        model = FastLanguageModel.get_peft_model(
            model,
            merge_and_unload=True,
        )
        merged_path = os.path.join(config['output']['output_dir'], 'merged')
        model.save_pretrained(merged_path)
        tokenizer.save_pretrained(merged_path)
        print(f"Merged model saved to: {merged_path}")

    # Save training configuration
    config_save_path = os.path.join(config['output']['output_dir'], 'training_config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"LoRA adapters saved to: {config['output']['output_dir']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
