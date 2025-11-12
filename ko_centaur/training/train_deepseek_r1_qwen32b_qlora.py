#!/usr/bin/env python3
"""
Train DeepSeek-R1-Distill-Qwen-32B with QLoRA (4-bit quantization)

GPU Memory: ~18-22GB (4-bit quantization with LoRA)
CPU Memory: ~50GB during loading
Model size: 32B parameters
Architecture: Qwen2.5-32B based (distilled from 671B DeepSeek-R1)
Performance: Outperforms OpenAI o1-mini on reasoning benchmarks

Strategy: BitsAndBytesConfig NF4 quantization + LoRA adapters
- Same architecture as Qwen2.5-32B (proven compatibility)
- Enhanced reasoning capabilities from DeepSeek-R1 distillation
- Korean language support confirmed
- Benchmark results: AIME 2024: 86.0 (vs o1-mini's 63.6)
"""

import argparse
import os
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
    print("DeepSeek-R1-Distill-Qwen-32B - QLoRA Training with 4-bit Quantization")
    print("=" * 80)
    print(f"Model: 32B parameters (distilled from 671B DeepSeek-R1)")
    print(f"Base Architecture: Qwen2.5-32B")
    print(f"Quantization: NF4 4-bit (BitsAndBytes)")
    print(f"GPU Configuration: 4 GPUs (multi-GPU training)")
    print(f"GPU Memory Expected: 18-22GB per GPU")
    print(f"Training: LoRA adapters with 4-bit base model")
    print(f"Korean Language: Confirmed support")
    print(f"Performance: Outperforms OpenAI o1-mini (AIME 2024: 86.0 vs 63.6)")
    print("=" * 80)

    # ============================================================================
    # GPU 사용 강제 확인
    # ============================================================================
    print("\n[0/6] GPU 사용 확인 및 강제 설정...")
    if not torch.cuda.is_available():
        raise RuntimeError("❌ CUDA를 사용할 수 없습니다! GPU가 필요합니다.")

    num_gpus = torch.cuda.device_count()
    print(f"✅ CUDA 사용 가능: {num_gpus}개 GPU 감지")

    for i in range(num_gpus):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_mem = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   GPU {i}: {gpu_name} ({gpu_mem:.1f} GB)")

    # CUDA_VISIBLE_DEVICES가 설정되어 있으면 사용, 없으면 모든 GPU 사용
    cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', None)
    if cuda_visible:
        print(f"   CUDA_VISIBLE_DEVICES: {cuda_visible}")
    else:
        print(f"   CUDA_VISIBLE_DEVICES: 설정 안됨 (모든 GPU 사용)")

    # 기본 디바이스를 GPU로 설정
    device = torch.device("cuda:0")
    print(f"   기본 디바이스: {device}")
    print("✅ GPU 사용 준비 완료")

    # ============================================================================
    # Step 1: Load Tokenizer
    # ============================================================================
    print("\n[1/6] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        config['model']['name'],
        trust_remote_code=config['model'].get('trust_remote_code', True)
    )

    # Qwen tokenizer setup (DeepSeek-R1-Distill-Qwen uses same tokenizer)
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
    print("\n[4/6] Loading DeepSeek-R1-Distill-Qwen-32B in 4-bit (5-10 minutes)...")
    print("🏆 Reasoning-enhanced model distilled from 671B DeepSeek-R1")
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
    print(f"   - Reasoning capability: Enhanced (o1-mini level)")

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
    print(f"   - Effective batch size: {config['training']['per_device_train_batch_size']} × {config['training']['gradient_accumulation_steps']} × {torch.cuda.device_count()} GPUs = {config['training']['per_device_train_batch_size'] * config['training']['gradient_accumulation_steps'] * torch.cuda.device_count()}")

    # 최종 GPU 사용 확인
    print("\n🔍 최종 GPU 사용 확인...")
    if not torch.cuda.is_available():
        raise RuntimeError("❌ 오류: 학습 시작 전 CUDA 사용 불가!")

    # 모델 파라미터의 주요 디바이스 확인
    device_counts = {}
    for param in model.parameters():
        dev_type = param.device.type
        device_counts[dev_type] = device_counts.get(dev_type, 0) + 1

    print(f"   파라미터 디바이스 분포: {device_counts}")

    # GPU에 파라미터가 있는지 확인
    if 'cuda' not in device_counts or device_counts['cuda'] == 0:
        raise RuntimeError("❌ 오류: 모델 파라미터가 GPU에 없습니다! 학습을 진행할 수 없습니다.")

    # GPU 파라미터가 대부분인지 확인
    total_params = sum(device_counts.values())
    gpu_ratio = device_counts.get('cuda', 0) / total_params if total_params > 0 else 0

    if gpu_ratio < 0.5:
        print(f"   ⚠️  경고: GPU 파라미터 비율이 낮습니다 ({gpu_ratio*100:.1f}%)")
        print(f"   ⚠️  학습 속도가 느려질 수 있습니다.")
    else:
        print(f"   ✅ GPU 파라미터 비율: {gpu_ratio*100:.1f}%")

    # 주요 디바이스 확인 (가장 많은 파라미터가 있는 디바이스)
    main_device = max(device_counts.items(), key=lambda x: x[1])[0]
    if main_device != 'cuda':
        print(f"   ⚠️  경고: 주요 파라미터가 {main_device}에 있습니다.")
        print(f"   ⚠️  하지만 학습은 GPU에서 진행됩니다 (Trainer가 자동 처리).")
    else:
        print(f"   ✅ 주요 파라미터가 GPU에 있습니다.")

    print(f"   ✅ 사용 가능한 GPU: {torch.cuda.device_count()}개")
    print("✅ GPU 사용 확인 완료 - 학습 시작합니다!")
    print("   (일부 파라미터가 CPU에 있어도 학습은 GPU에서 진행됩니다)")

    # ============================================================================
    # Training
    # ============================================================================
    print("\n" + "=" * 80)
    print("Starting DeepSeek-R1-Distill-Qwen-32B QLoRA training...")
    print("Monitor GPU memory: nvidia-smi dmon -s mu")
    print(f"Expected: 18-22GB per GPU ({torch.cuda.device_count()} GPUs total)")
    print("Model Performance: AIME 2024: 86.0 (beats o1-mini's 63.6)")
    print("=" * 80)

    trainer.train()

    print("\n" + "=" * 80)
    print("Saving LoRA adapters...")
    print("=" * 80)
    model.save_pretrained(config['output']['output_dir'])
    tokenizer.save_pretrained(config['output']['output_dir'])
    print("✅ Training complete!")
    print(f"LoRA adapters saved to: {config['output']['output_dir']}")
    print(f"🏆 Reasoning-enhanced model ready for CENTaUR workflow!")


if __name__ == "__main__":
    main()
