#!/usr/bin/env python3
"""
CENTaUR Feature Extraction Script (FIXED)
==========================================

Extracts hidden states from fine-tuned models following the original
Binz & Schulz (2023) methodology for cognitive modeling.

CRITICAL: Original CENTaUR methodology (verified from legacy/choices13k/query.py):
- Uses generate() to produce ONE token ("1" or "2")
- Extracts hidden state from the GENERATED token, NOT the last prompt token
- This captures the model's choice representation, not just prompt encoding

Implementation:
- Uses model.generate(max_new_tokens=1, output_hidden_states=True)
- Extracts last-layer hidden state (5120-dim for Qwen2.5/DeepSeek)
- Suitable for downstream binomial regression

Previous bug (FIXED in this version):
- Was using forward() pass and extracting last PROMPT token
- This caused features to be too similar (89% pairwise similarity)
- Fine-tuned models performed worse than random baseline

Usage:
    python extract_centaur_features.py --model qwen25 --n_samples 10
    python extract_centaur_features.py --model deepseek --dataset custom.jsonl
"""

import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from pathlib import Path
import time


def extract_hidden_states(
    base_model_path: str,
    adapter_path: str,
    dataset_path: str,
    output_path: str,
    model_name: str,
    n_samples: int = None,
    is_local: bool = False,
    use_quantization: bool = True,
    use_adapter: bool = True
):
    """
    Extract last-layer hidden states from model (with or without fine-tuning).

    Following original CENTaUR methodology (CORRECTED):
    1. Load base model (+ optional LoRA adapter)
    2. Generate ONE token ("1" or "2") for each choice prompt
    3. Extract hidden state from the GENERATED token (not prompt!)
    4. Save features for downstream regression

    CRITICAL FIX: Previous version extracted last PROMPT token, which caused
    features to be too similar. We now extract from GENERATED token to capture
    the model's actual choice representation.

    Args:
        base_model_path: HuggingFace model ID or local path
        adapter_path: Path to LoRA adapter directory (None for base models)
        dataset_path: Path to JSONL dataset (text + choice fields)
        output_path: Where to save extracted features
        model_name: Descriptive name for logging
        n_samples: Limit processing (None = all samples)
        is_local: Whether base model is local (for trust_remote_code)
        use_quantization: Use NF4 quantization (saves memory)
        use_adapter: Whether to load LoRA adapter (False for base models)
    """

    print(f"\n{'='*70}")
    print(f"CENTaUR Feature Extraction")
    print(f"Model: {model_name}")
    print(f"{'='*70}\n")

    start_time = time.time()

    # ============================================================
    # 1. Configure Quantization (Optional)
    # ============================================================

    bnb_config = None
    if use_quantization:
        print("1. Configuring NF4 quantization (4-bit)...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )
        print("   ✓ 4-bit NF4 quantization enabled")
        print("   ✓ Double quantization: True")
        print("   ✓ Compute dtype: bfloat16")
    else:
        print("1. Quantization disabled (full precision)")

    # ============================================================
    # 2. Load Tokenizer
    # ============================================================

    print(f"\n2. Loading tokenizer...")
    print(f"   Source: {base_model_path}")

    tokenizer_kwargs = {}
    if is_local:
        tokenizer_kwargs["trust_remote_code"] = True

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, **tokenizer_kwargs)
    print(f"   ✓ Tokenizer loaded")
    print(f"   Vocab size: {len(tokenizer)}")

    # ============================================================
    # 3. Load Base Model
    # ============================================================

    print(f"\n3. Loading base model...")
    print(f"   Source: {base_model_path}")

    model_kwargs = {
        "device_map": "auto",
        "torch_dtype": torch.bfloat16
    }

    if use_quantization:
        model_kwargs["quantization_config"] = bnb_config

    if is_local:
        model_kwargs["trust_remote_code"] = True

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        **model_kwargs
    )

    print(f"   ✓ Base model loaded")
    print(f"   Device map: auto-distributed")
    if use_quantization:
        print(f"   Memory mode: 4-bit quantized (~21GB GPU)")
    else:
        print(f"   Memory mode: Full precision (~40GB+ GPU)")

    # ============================================================
    # 4. Load LoRA Adapter (Optional)
    # ============================================================

    if use_adapter and adapter_path:
        print(f"\n4. Loading LoRA adapter...")
        print(f"   Source: {adapter_path}")

        model = PeftModel.from_pretrained(model, adapter_path)
        model.eval()  # Set to evaluation mode

        print(f"   ✓ LoRA adapter loaded")
        print(f"   Model ready for inference (eval mode)")
    else:
        print(f"\n4. Using base model (no adapter)")
        model.eval()  # Set to evaluation mode
        print(f"   ✓ Base model ready for inference (eval mode)")

    # ============================================================
    # 5. Load Dataset
    # ============================================================

    print(f"\n5. Loading dataset...")
    print(f"   Source: {dataset_path}")

    with open(dataset_path, "r") as f:
        data = [json.loads(line) for line in f]

    if n_samples:
        data = data[:n_samples]
        print(f"   ⚠️  Limited to first {n_samples} samples")

    print(f"   ✓ Dataset loaded")
    print(f"   Total samples: {len(data)}")

    # Inspect first sample
    if data:
        first_sample = data[0]
        print(f"\n   Sample format check:")
        print(f"   - Text field: {len(first_sample['text'])} chars")
        print(f"   - Choice label: {first_sample['choice']}")
        print(f"   - Preview: {first_sample['text'][:100]}...")

    # ============================================================
    # 6. Extract Hidden States (Core Loop)
    # ============================================================

    print(f"\n6. Extracting hidden states...")
    print(f"   {'─'*66}")

    all_features = []
    all_labels = []
    extraction_start = time.time()

    for idx, item in enumerate(tqdm(data, desc="   Extracting", ncols=70)):
        prompt = item["text"]
        label = item["choice"]  # 0 for A, 1 for B

        # Tokenize prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # CRITICAL FIX: Generate ONE token + forward pass for hidden state
        # Original CENTaUR: llama.generate() then extract llama.generator.model.hl
        # Modern approach: generate token, then forward pass with full sequence
        with torch.no_grad():
            # Step 1: Generate ONE token (deterministic)
            gen_outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                temperature=0.0,  # Deterministic
                do_sample=False,  # Greedy decoding
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

            # Step 2: Forward pass with FULL sequence (prompt + generated token)
            # to get hidden state of the GENERATED token
            full_outputs = model(
                input_ids=gen_outputs,
                output_hidden_states=True
            )

            # Extract hidden state from LAST position (the generated token)
            # full_outputs.hidden_states: tuple of (n_layers+1,)
            # Each element: (batch_size, seq_len, hidden_dim)
            # seq_len = prompt_len + 1 (includes generated token)
            last_layer_hidden = full_outputs.hidden_states[-1]  # Last layer
            generated_token_hidden = last_layer_hidden[0, -1, :]  # Last position = generated token

        all_features.append(generated_token_hidden.cpu())
        all_labels.append(label)

        # Progress update every 10 samples
        if (idx + 1) % 10 == 0:
            elapsed = time.time() - extraction_start
            rate = (idx + 1) / elapsed
            remaining = (len(data) - idx - 1) / rate if rate > 0 else 0
            print(f"   [{idx + 1:>4}/{len(data)}] | "
                  f"Rate: {rate:.1f} samples/s | "
                  f"ETA: {remaining/60:.1f} min")

    # Stack features
    features = torch.stack(all_features)  # (n_samples, hidden_dim)
    labels = torch.tensor(all_labels)     # (n_samples,)

    extraction_time = time.time() - extraction_start

    print(f"\n   ✓ Feature extraction complete!")
    print(f"   {'─'*66}")
    print(f"   Features shape: {features.shape}")
    print(f"   Labels shape:   {labels.shape}")
    print(f"   Hidden dim:     {features.shape[1]}")
    print(f"   Extraction time: {extraction_time/60:.1f} minutes")
    print(f"   Average rate:   {len(data)/extraction_time:.2f} samples/s")

    # ============================================================
    # 7. Save Features
    # ============================================================

    print(f"\n7. Saving features...")

    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save with metadata for downstream analysis
    torch.save({
        "features": features,           # (n_samples, hidden_dim)
        "labels": labels,               # (n_samples,)
        "model_name": model_name,
        "base_model_path": base_model_path,
        "adapter_path": adapter_path,
        "dataset_path": dataset_path,
        "n_samples": len(data),
        "hidden_dim": features.shape[1],
        "extraction_time": extraction_time,
        "quantized": use_quantization,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
    }, output_path)

    print(f"   ✓ Features saved to:")
    print(f"     {output_path}")
    print(f"   File size: {Path(output_path).stat().st_size / 1024 / 1024:.1f} MB")

    # ============================================================
    # 8. Summary
    # ============================================================

    total_time = time.time() - start_time

    print(f"\n{'='*70}")
    print(f"Extraction Complete!")
    print(f"{'='*70}")
    print(f"Total time:     {total_time/60:.1f} minutes")
    print(f"Samples:        {len(data)}")
    print(f"Hidden dim:     {features.shape[1]}")
    print(f"Output:         {output_path}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract CENTaUR features from fine-tuned models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Extract from Qwen2.5-QLoRA (fine-tuned, first 10 samples)
  python extract_centaur_features.py --model qwen25 --n_samples 10

  # Extract from Qwen2.5-Base (no fine-tuning, all samples)
  python extract_centaur_features.py --model qwen25-base

  # Extract from DeepSeek-QLoRA (fine-tuned)
  python extract_centaur_features.py --model deepseek

  # Extract from DeepSeek-Base (no fine-tuning)
  python extract_centaur_features.py --model deepseek-base

  # Extract from EXAONE base model
  python extract_centaur_features.py --model exaone-base

  # Extract from Ko-CENTaUR (EXAONE + Psych-101)
  python extract_centaur_features.py --model ko-centaur

  # Extract from GPT-OSS-20B (OpenAI open-source GPT)
  python extract_centaur_features.py --model gpt-oss --n_samples 10

  # Extract from EXAONE-3.5-32B (Latest Korean model, top performance)
  python extract_centaur_features.py --model exaone35-base --n_samples 10

  # Extract from Kimi K2 (Multi-Agent optimized, 128K context)
  python extract_centaur_features.py --model kimi-k2 --n_samples 10

  # Custom dataset and output
  python extract_centaur_features.py --model qwen25-base \\
      --dataset /path/to/custom.jsonl \\
      --output /path/to/output.pth
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "qwen25", "qwen25-base",
            "deepseek", "deepseek-base",
            "exaone-base", "exaone35-base", "ko-centaur",
            "gpt-oss", "gpt-oss-base", "gpt-oss-120b",
            "motif", "gpt-neox", "polyglot-ko",
            "gecko", "gpt-j", "cerebras",
            "kimi-k2", "kimi-k2-base"
        ],
        help="Model to extract features from"
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset JSONL (default: server path for choices13k_100)"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for features (default: server data directory)"
    )

    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all)"
    )

    parser.add_argument(
        "--no-quantization",
        action="store_true",
        help="Disable 4-bit quantization (requires more GPU memory)"
    )

    args = parser.parse_args()

    # ============================================================
    # Model Configurations
    # ============================================================

    dataset_default = "/scratch/connectome/connectome1/ko-centaur/data/choices13k_100.jsonl"

    if args.model == "qwen25":
        base_model = "Qwen/Qwen2.5-32B-Instruct"
        adapter = "models/qwen25-32b-qlora"
        output_default = "data/features/centaur_features_qwen25.pth"
        name = "Qwen2.5-32B-QLoRA"
        is_local = False
        use_adapter = True

    elif args.model == "qwen25-base":
        base_model = "Qwen/Qwen2.5-32B-Instruct"
        adapter = None
        output_default = "data/features/centaur_features_qwen25_base.pth"
        name = "Qwen2.5-32B-Base"
        is_local = False
        use_adapter = False

    elif args.model == "deepseek":
        base_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        adapter = "/scratch/connectome/connectome1/ko-centaur/outputs/deepseek-r1-qwen32b-qlora"
        output_default = "/scratch/connectome/connectome1/ko-centaur/data/features/centaur_features_deepseek.pth"
        name = "DeepSeek-R1-32B-QLoRA"
        is_local = True
        use_adapter = True

    elif args.model == "deepseek-base":
        base_model = "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
        adapter = None
        output_default = "/scratch/connectome/connectome1/ko-centaur/data/features/centaur_features_deepseek_base.pth"
        name = "DeepSeek-R1-32B-Base"
        is_local = True
        use_adapter = False

    elif args.model == "exaone-base":
        base_model = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
        adapter = None
        output_default = "/scratch/connectome/connectome1/ko-centaur/data/features/centaur_features_exaone_base.pth"
        name = "EXAONE-3.0-7.8B-Base"
        is_local = False
        use_adapter = False

    elif args.model == "ko-centaur":
        # Ko-CENTaUR: EXAONE + Psych-101 fine-tuned
        base_model = "LGAI-EXAONE/EXAONE-3.0-7.8B-Instruct"
        adapter = "/scratch/connectome/connectome1/ko-centaur/models/exaone-psych101-full/checkpoint-22536"
        output_default = "/scratch/connectome/connectome1/ko-centaur/data/features/centaur_features_kocentaur.pth"
        name = "Ko-CENTaUR (EXAONE+Psych-101)"
        is_local = False
        use_adapter = True

    elif args.model == "gpt-oss":
        # GPT-OSS-20B (OpenAI open-source GPT model)
        base_model = "/home/connectome/connectome1/models/gpt-oss-20b"
        adapter = None
        output_default = "/scratch/connectome/connectome1/ko-centaur/data/features/centaur_features_gpt_oss.pth"
        name = "GPT-OSS-20B"
        is_local = True
        use_adapter = False

    elif args.model == "gpt-oss-base":
        # GPT-OSS-20B Base (same as gpt-oss, explicit base variant)
        base_model = "/home/connectome/connectome1/models/gpt-oss-20b"
        adapter = None
        output_default = "/scratch/connectome/connectome1/ko-centaur/data/features/centaur_features_gpt_oss_base.pth"
        name = "GPT-OSS-20B-Base"
        is_local = True
        use_adapter = False

    elif args.model == "exaone35-base":
        # EXAONE-3.5-32B (LG AI Research, latest version)
        base_model = "LGAI-EXAONE/EXAONE-3.5-32B-Instruct"
        adapter = None
        output_default = "/scratch/connectome/connectome1/ko-centaur/data/features/centaur_features_exaone35_base.pth"
        name = "EXAONE-3.5-32B-Base"
        is_local = True  # Custom code requires trust_remote_code=True
        use_adapter = False

    elif args.model == "gpt-oss-120b":
        # GPT-OSS-120B (OpenAI, MoE architecture)
        base_model = "/home/connectome/connectome1/models/gpt-oss-120b"
        adapter = None
        output_default = "/scratch/connectome/connectome1/ko-centaur/data/features/centaur_features_gpt_oss_120b.pth"
        name = "GPT-OSS-120B"
        is_local = True
        use_adapter = False

    elif args.model == "motif":
        # Motif-102B (Moreh, Korean-specialized)
        base_model = "/home/connectome/connectome1/models/motif"
        adapter = None
        output_default = "/scratch/connectome/connectome1/ko-centaur/data/features/centaur_features_motif.pth"
        name = "Motif-102B"
        is_local = True
        use_adapter = False

    elif args.model == "gpt-neox":
        # GPT-NeoX-20B (EleutherAI)
        base_model = "/home/connectome/connectome1/models/gpt-neox"
        adapter = None
        output_default = "/scratch/connectome/connectome1/ko-centaur/data/features/centaur_features_gpt_neox.pth"
        name = "GPT-NeoX-20B"
        is_local = True
        use_adapter = False

    elif args.model == "polyglot-ko":
        # Polyglot-Ko-12.8B (EleutherAI, Korean-only)
        base_model = "/home/connectome/connectome1/models/polyglot-ko"
        adapter = None
        output_default = "/scratch/connectome/connectome1/ko-centaur/data/features/centaur_features_polyglot_ko.pth"
        name = "Polyglot-Ko-12.8B"
        is_local = True
        use_adapter = False

    elif args.model == "gecko":
        # GECKO-7B (Korean-English bilingual)
        base_model = "/home/connectome/connectome1/models/gecko"
        adapter = None
        output_default = "/scratch/connectome/connectome1/ko-centaur/data/features/centaur_features_gecko.pth"
        name = "GECKO-7B"
        is_local = True
        use_adapter = False

    elif args.model == "gpt-j":
        # GPT-J-6B (EleutherAI, lightweight)
        base_model = "/home/connectome/connectome1/models/gpt-j"
        adapter = None
        output_default = "/scratch/connectome/connectome1/ko-centaur/data/features/centaur_features_gpt_j.pth"
        name = "GPT-J-6B"
        is_local = True
        use_adapter = False

    elif args.model == "cerebras":
        # Cerebras-GPT-13B (Cerebras)
        base_model = "/home/connectome/connectome1/models/cerebras"
        adapter = None
        output_default = "/scratch/connectome/connectome1/ko-centaur/data/features/centaur_features_cerebras.pth"
        name = "Cerebras-GPT-13B"
        is_local = True
        use_adapter = False

    elif args.model == "kimi-k2":
        # Kimi K2 Instruct (Moonshot AI, MoE architecture, Multi-Agent optimized)
        # 올바른 모델 ID: moonshotai (not moonshot-ai)
        base_model = "moonshotai/Kimi-K2-Instruct"
        adapter = None
        output_default = "/scratch/connectome/connectome1/ko-centaur/data/features/centaur_features_kimi_k2.pth"
        name = "Kimi-K2-Instruct"
        is_local = True  # Custom code requires trust_remote_code=True
        use_adapter = False

    elif args.model == "kimi-k2-base":
        # Kimi K2 Base (Moonshot AI, MoE architecture)
        base_model = "moonshotai/Kimi-K2-Base"
        adapter = None
        output_default = "/scratch/connectome/connectome1/ko-centaur/data/features/centaur_features_kimi_k2_base.pth"
        name = "Kimi-K2-Base"
        is_local = True  # Custom code requires trust_remote_code=True
        use_adapter = False

    # Use defaults or command-line overrides
    dataset = args.dataset if args.dataset else dataset_default
    output = args.output if args.output else output_default
    use_quant = not args.no_quantization

    # ============================================================
    # Run Extraction
    # ============================================================

    extract_hidden_states(
        base_model_path=base_model,
        adapter_path=adapter,
        dataset_path=dataset,
        output_path=output,
        model_name=name,
        n_samples=args.n_samples,
        is_local=is_local,
        use_quantization=use_quant,
        use_adapter=use_adapter
    )
