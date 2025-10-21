#!/usr/bin/env python3
"""
Train EXAONE 4.0-32B with DeepSpeed ZeRO-3
Method: Distributed training with CPU offloading across 7 GPUs

Usage:
    deepspeed --num_gpus=7 training/train_exaone40_deepspeed.py \
        --config configs/training_exaone40.yaml \
        --deepspeed configs/deepspeed_zero3_32b.json
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
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
    set_seed,
)


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def prepare_dataset(
    data_path: str,
    tokenizer,
    max_length: int = 512,
    validation_split: float = 0.1
):
    """
    Load and tokenize the choices13k dataset

    Format: {"text": "Which option would you choose? ...", "choice": 0/1}
    """
    # Load dataset from JSONL
    dataset = load_dataset('json', data_files=data_path, split='train')

    def tokenize_function(examples):
        # Tokenize the text prompts
        tokenized = tokenizer(
            examples['text'],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

        # Add labels (same as input_ids for causal LM)
        tokenized['labels'] = tokenized['input_ids'].copy()

        return tokenized

    # Tokenize dataset
    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing dataset"
    )

    # Split into train and validation
    if validation_split > 0:
        split = tokenized_dataset.train_test_split(
            test_size=validation_split,
            seed=42
        )
        return split['train'], split['test']
    else:
        return tokenized_dataset, None


def main():
    parser = argparse.ArgumentParser(description='Train EXAONE 4.0-32B with DeepSpeed')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/training_exaone40.yaml',
        help='Path to training configuration YAML'
    )
    parser.add_argument('--local_rank', type=int, default=-1, help='Local rank for distributed training')

    # Parse known args first (DeepSpeed adds additional args)
    args, _ = parser.parse_known_args()

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
        tokenizer=tokenizer,
        max_length=config['data']['max_seq_length'],
        validation_split=config['data']['validation_split']
    )

    print(f"Train samples: {len(train_dataset)}")
    if eval_dataset:
        print(f"Validation samples: {len(eval_dataset)}")

    # Training arguments must be created before model loading so DeepSpeed
    # can intercept the from_pretrained call and shard weights during init.
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
        fp16=config['training']['fp16'],
        bf16=config['training']['bf16'],

        # Gradient checkpointing
        gradient_checkpointing=config['training']['gradient_checkpointing'],
        gradient_checkpointing_kwargs=config['training']['gradient_checkpointing_kwargs'],

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

        # DeepSpeed
        deepspeed=config['training']['deepspeed'],

        # Distributed
        local_rank=config['misc']['local_rank'],
        ddp_find_unused_parameters=config['misc']['ddp_find_unused_parameters'],

        # Data loading
        dataloader_num_workers=config['data']['dataloader_num_workers'],
        dataloader_pin_memory=config['data']['dataloader_pin_memory'],

        # Reproducibility
        seed=config['misc']['seed'],
        data_seed=config['misc']['data_seed'],
    )

    # Load model with DeepSpeed ZeRO-3
    print(f"\n{'='*60}")
    print("Loading EXAONE 4.0-32B model...")
    print("Note: With DeepSpeed ZeRO-3, model loading is distributed")
    print(f"{'='*60}")

    # Model configuration
    model_config = AutoConfig.from_pretrained(
        config['model']['name'],
        revision=config['model']['revision'],
        trust_remote_code=config['model']['trust_remote_code']
    )

    # Load model
    # Note: DeepSpeed will handle distribution and offloading
    model = AutoModelForCausalLM.from_pretrained(
        config['model']['name'],
        config=model_config,
        revision=config['model']['revision'],
        trust_remote_code=config['model']['trust_remote_code'],
        torch_dtype=torch.bfloat16 if config['training']['bf16'] else torch.float16,
    )

    # Enable gradient checkpointing for memory efficiency
    if config['training']['gradient_checkpointing']:
        model.gradient_checkpointing_enable()

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False  # Causal LM, not masked LM
    )

    # Initialize Trainer
    print(f"\n{'='*60}")
    print("Initializing Trainer with DeepSpeed...")
    print(f"{'='*60}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Resume from checkpoint if specified
    resume_from = config['output'].get('resume_from_checkpoint')

    # Train
    print(f"\n{'='*60}")
    print("Starting training...")
    print(f"Model: EXAONE 4.0-32B (32B parameters)")
    print(f"Method: DeepSpeed ZeRO-3 with CPU offloading")
    print(f"GPUs: 7x RTX 3090 (24GB each)")
    print(f"Effective batch size: {config['training']['per_device_train_batch_size']} * {config['training']['gradient_accumulation_steps']} * 7 = {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps'] * 7}")
    print(f"{'='*60}\n")

    trainer.train(resume_from_checkpoint=resume_from)

    # Save final model
    print(f"\n{'='*60}")
    print("Saving final model...")
    print(f"{'='*60}")
    trainer.save_model()
    tokenizer.save_pretrained(config['output']['output_dir'])

    # Save training configuration
    config_save_path = os.path.join(config['output']['output_dir'], 'training_config.yaml')
    with open(config_save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"Model saved to: {config['output']['output_dir']}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
