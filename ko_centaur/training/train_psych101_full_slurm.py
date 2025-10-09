#!/usr/bin/env python3
"""
Full-Scale Psych-101 Training Script for Ko-CENTaUR (SLURM Version)

Trains EXAONE-3.0-7.8B-Instruct on complete Psych-101 dataset (60,092 samples)
using QLoRA (4-bit quantization + LoRA adapters) for memory efficiency.

SLURM-optimized version: Uses single GPU allocated by SLURM via --gres
Expected training time: 4-6 hours on single RTX 3090 (24GB)
"""

import os
import sys
import torch
import time
from pathlib import Path
from datetime import datetime
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import json

# =============================================================================
# Configuration
# =============================================================================

# SLURM handles GPU allocation - do NOT set CUDA_VISIBLE_DEVICES
# SLURM sets this automatically via --gres=gpu:rtx:1

# Set HuggingFace cache to /scratch to avoid home directory space issues
os.environ["HF_HOME"] = "/scratch/connectome/connectome1/ko-centaur/.cache/huggingface"
os.environ["TRANSFORMERS_CACHE"] = "/scratch/connectome/connectome1/ko-centaur/.cache/huggingface"

# Model configuration
MODEL_NAME = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
OUTPUT_DIR = "/scratch/connectome/connectome1/ko-centaur/models/exaone-psych101-full"
LOG_DIR = "/scratch/connectome/connectome1/ko-centaur/logs"
DATA_FILE = "/scratch/connectome/connectome1/ko-centaur/data/psych101_exaone_train.jsonl"
CACHE_DIR = "/scratch/connectome/connectome1/ko-centaur/.cache/huggingface"

# Resume from checkpoint
RESUME_FROM_CHECKPOINT = "/scratch/connectome/connectome1/ko-centaur/models/exaone-psych101-full/checkpoint-3000"

# Training hyperparameters - MEMORY OPTIMIZED for RTX A5000 24GB (Option A)
# Maintains effective batch size = 8 while reducing peak memory usage
# Previous config (batch_size=2, grad_accum=4, seq_len=1024) caused OOM after step 3000
BATCH_SIZE = 1  # Reduced from 2 to lower peak memory per forward pass
GRADIENT_ACCUMULATION_STEPS = 8  # Increased from 4 to maintain effective batch size = 8
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
MAX_SEQ_LENGTH = 512  # Reduced from 1024 to reduce activation memory (sufficient for Psych-101)

# Batch size = 1 safety mechanisms (critical for stable training)
WARMUP_STEPS = 500  # Extended from 100 to stabilize AdamW momentum with batch=1 gradient noise
MAX_GRAD_NORM = 1.0  # Gradient clipping to prevent spikes from high-variance batch=1 updates
LOGGING_STEPS = 50
SAVE_STEPS = 500

# LoRA configuration
LORA_R = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# =============================================================================
# Helper Functions
# =============================================================================

def setup_logging():
    """Create log directory and return log file path"""
    log_dir = Path(LOG_DIR)
    log_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"train_psych101_full_slurm_{timestamp}.log"

    return log_file

def log_message(message, log_file=None):
    """Print and optionally write to log file"""
    print(message)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(message + '\n')

def format_psych101_prompt(example):
    """
    Format Psych-101 samples into EXAONE instruction template

    Supports two formats:
    1. Original Psych-101: {'prompt': str, 'choice': int}
    2. Preprocessed EXAONE: {'messages': [{'role': str, 'content': str}]}

    EXAONE template:
    [|system|]You are EXAONE model from LG AI Research...[|endofturn|]
    [|user|]{question}[|endofturn|]
    [|assistant|]{answer}[|endofturn|]
    """

    # Check if already in messages format (preprocessed)
    if 'messages' in example:
        messages = example['messages']
        formatted_parts = []

        for msg in messages:
            role = msg['role']
            content = msg['content']

            if role == 'system':
                formatted_parts.append(f"[|system|]{content}[|endofturn|]")
            elif role == 'user':
                formatted_parts.append(f"[|user|]{content}[|endofturn|]")
            elif role == 'assistant':
                formatted_parts.append(f"[|assistant|]{content}[|endofturn|]")

        formatted_text = "\n".join(formatted_parts)
        return {"text": formatted_text}

    # Original Psych-101 format
    system_prompt = "You are EXAONE model from LG AI Research, a helpful assistant."
    question = example['prompt']

    # Extract the correct answer from prompt or use provided choice
    if 'choice' in example:
        answer = str(example['choice'])
    else:
        answer = "1"  # Default fallback

    formatted_text = (
        f"[|system|]{system_prompt}[|endofturn|]\n"
        f"[|user|]{question}[|endofturn|]\n"
        f"[|assistant|]{answer}[|endofturn|]"
    )

    return {"text": formatted_text}

def print_trainable_parameters(model, log_file=None):
    """Print the number of trainable parameters"""
    trainable_params = 0
    all_param = 0

    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    message = (
        f"Trainable params: {trainable_params:,} || "
        f"All params: {all_param:,} || "
        f"Trainable%: {100 * trainable_params / all_param:.2f}%"
    )

    log_message(message, log_file)

# =============================================================================
# Main Training Pipeline
# =============================================================================

def main():
    start_time = time.time()
    log_file = setup_logging()

    log_message("="*70, log_file)
    log_message("Ko-CENTaUR: Full Psych-101 Training (SLURM - RESUME)", log_file)
    log_message("="*70, log_file)
    log_message(f"Start time: {datetime.now()}", log_file)
    log_message(f"Model: {MODEL_NAME}", log_file)
    log_message(f"Output directory: {OUTPUT_DIR}", log_file)
    log_message(f"Resume from: {RESUME_FROM_CHECKPOINT}", log_file)
    log_message(f"Log file: {log_file}", log_file)
    log_message(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not set (SLURM managed)')}", log_file)

    # GPU info
    if torch.cuda.is_available():
        num_gpus = torch.cuda.device_count()
        log_message(f"\nGPU Configuration:", log_file)
        log_message(f"  Available GPUs: {num_gpus}", log_file)
        for i in range(num_gpus):
            gpu_name = torch.cuda.get_device_name(i)
            gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
            log_message(f"  GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)", log_file)
        if num_gpus > 1:
            log_message(f"  Training mode: DDP (Distributed Data Parallel)", log_file)
        else:
            log_message(f"  Training mode: Single GPU", log_file)

    # -------------------------------------------------------------------------
    # 1. Load Dataset
    # -------------------------------------------------------------------------
    log_message("\n[1/6] Loading Psych-101 dataset...", log_file)

    try:
        # Load preprocessed JSONL file
        dataset = load_dataset("json", data_files={"train": DATA_FILE})
        log_message(f"✅ Dataset loaded successfully", log_file)
        log_message(f"   Train samples: {len(dataset['train'])}", log_file)

        # Format dataset (handles both messages and prompt formats)
        log_message("   Formatting prompts...", log_file)
        formatted_dataset = dataset.map(
            format_psych101_prompt,
            remove_columns=[col for col in dataset['train'].column_names if col != 'text'],
            desc="Formatting prompts"
        )

        log_message(f"✅ Dataset formatted", log_file)

    except Exception as e:
        log_message(f"❌ Error loading dataset: {e}", log_file)
        import traceback
        log_message(traceback.format_exc(), log_file)
        return

    # -------------------------------------------------------------------------
    # 2. Load Tokenizer
    # -------------------------------------------------------------------------
    log_message("\n[2/6] Loading tokenizer...", log_file)

    try:
        tokenizer = AutoTokenizer.from_pretrained(
            MODEL_NAME,
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )

        # Set padding token if not present
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        log_message(f"✅ Tokenizer loaded", log_file)
        log_message(f"   Vocab size: {len(tokenizer)}", log_file)

    except Exception as e:
        log_message(f"❌ Error loading tokenizer: {e}", log_file)
        return

    # -------------------------------------------------------------------------
    # 3. Tokenize Dataset
    # -------------------------------------------------------------------------
    log_message("\n[3/6] Tokenizing dataset...", log_file)

    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=MAX_SEQ_LENGTH,
            padding="max_length"
        )

    try:
        tokenized_dataset = formatted_dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing"
        )

        log_message(f"✅ Dataset tokenized", log_file)

    except Exception as e:
        log_message(f"❌ Error tokenizing dataset: {e}", log_file)
        return

    # -------------------------------------------------------------------------
    # 4. Load Model with 8-bit Quantization (More Stable)
    # -------------------------------------------------------------------------
    log_message("\n[4/6] Loading model with 8-bit quantization...", log_file)

    # 8-bit quantization config (more stable than 4-bit with CUDA 11.8)
    bnb_config = BitsAndBytesConfig(
        load_in_8bit=True,
        llm_int8_threshold=6.0,  # Default threshold for outlier detection
    )

    try:
        # Single GPU: explicit device placement
        device_map = {"": 0}

        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map=device_map,
            trust_remote_code=True,
            cache_dir=CACHE_DIR
        )

        log_message(f"✅ Model loaded", log_file)
        log_message(f"   Device map: {model.hf_device_map}", log_file)

        # Prepare model for training
        model = prepare_model_for_kbit_training(model)
        log_message(f"✅ Model prepared for k-bit training", log_file)

    except Exception as e:
        log_message(f"❌ Error loading model: {e}", log_file)
        import traceback
        log_message(traceback.format_exc(), log_file)
        return

    # -------------------------------------------------------------------------
    # 5. Add LoRA Adapters
    # -------------------------------------------------------------------------
    log_message("\n[5/6] Adding LoRA adapters...", log_file)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        task_type="CAUSAL_LM"
    )

    try:
        model = get_peft_model(model, lora_config)
        log_message(f"✅ LoRA adapters added", log_file)
        print_trainable_parameters(model, log_file)

    except Exception as e:
        log_message(f"❌ Error adding LoRA: {e}", log_file)
        return

    # -------------------------------------------------------------------------
    # 6. Training
    # -------------------------------------------------------------------------
    log_message("\n[6/6] Starting training...", log_file)

    # Training arguments (Single GPU configuration)
    # Note: Using bf16 instead of fp16 for better stability with BitsAndBytes
    # Gradient checkpointing enabled via prepare_model_for_kbit_training for memory efficiency
    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        warmup_steps=WARMUP_STEPS,
        max_grad_norm=MAX_GRAD_NORM,  # Gradient clipping for batch=1 stability
        logging_steps=LOGGING_STEPS,
        save_steps=SAVE_STEPS,
        save_total_limit=3,
        bf16=True,  # Using bf16 instead of fp16 for stability
        optim="paged_adamw_8bit",
        logging_dir=f"{OUTPUT_DIR}/logs",
        report_to="none",
        remove_unused_columns=False,
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )

    # Train
    try:
        log_message(f"\nTraining configuration:", log_file)
        log_message(f"  Epochs: {NUM_EPOCHS}", log_file)
        log_message(f"  Batch size: {BATCH_SIZE}", log_file)
        log_message(f"  Gradient accumulation: {GRADIENT_ACCUMULATION_STEPS}", log_file)
        log_message(f"  Effective batch size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}", log_file)
        log_message(f"  Learning rate: {LEARNING_RATE}", log_file)
        log_message(f"  Warmup steps: {WARMUP_STEPS} (extended for batch=1 stability)", log_file)
        log_message(f"  Max grad norm: {MAX_GRAD_NORM} (gradient clipping)", log_file)
        log_message(f"  Max sequence length: {MAX_SEQ_LENGTH}", log_file)
        log_message(f"  Total samples: {len(tokenized_dataset['train'])}", log_file)

        # Log GPU memory info
        if torch.cuda.is_available():
            log_message(f"\nGPU Memory:", log_file)
            log_message(f"  Allocated: {torch.cuda.memory_allocated(0) / 1024**3:.2f} GB", log_file)
            log_message(f"  Reserved: {torch.cuda.memory_reserved(0) / 1024**3:.2f} GB", log_file)
            log_message(f"  Total: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB", log_file)

        # Resume from checkpoint if specified
        if RESUME_FROM_CHECKPOINT and Path(RESUME_FROM_CHECKPOINT).exists():
            log_message(f"\n🔄 Resuming training from checkpoint: {RESUME_FROM_CHECKPOINT}", log_file)
            train_result = trainer.train(resume_from_checkpoint=RESUME_FROM_CHECKPOINT)
        else:
            log_message(f"\n⚠️ Checkpoint not found, starting from scratch", log_file)
            train_result = trainer.train()

        log_message(f"\n✅ Training completed!", log_file)

        # Save model
        trainer.save_model()
        log_message(f"✅ Model saved to: {OUTPUT_DIR}", log_file)

        # Save training metrics
        metrics_file = Path(OUTPUT_DIR) / "training_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(train_result.metrics, f, indent=2)
        log_message(f"✅ Metrics saved to: {metrics_file}", log_file)

    except Exception as e:
        log_message(f"❌ Error during training: {e}", log_file)
        import traceback
        log_message(traceback.format_exc(), log_file)
        return

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    elapsed_time = time.time() - start_time
    hours = int(elapsed_time // 3600)
    minutes = int((elapsed_time % 3600) // 60)

    log_message("\n" + "="*70, log_file)
    log_message("Training Complete", log_file)
    log_message("="*70, log_file)
    log_message(f"Total time: {hours}h {minutes}m", log_file)
    log_message(f"Final loss: {train_result.training_loss:.4f}", log_file)
    log_message(f"Model saved to: {OUTPUT_DIR}", log_file)
    log_message(f"Log file: {log_file}", log_file)
    log_message("="*70, log_file)

if __name__ == "__main__":
    main()
