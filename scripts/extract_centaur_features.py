#!/usr/bin/env python3
"""
CENTaUR Feature Extraction Script
==================================

Extracts hidden states from fine-tuned models following the original
Binz & Schulz (2023) methodology for cognitive modeling.

Key differences from token probability approach:
- Extracts last-layer hidden states (4096-dim) instead of token logits
- Uses forward() pass, not generate()
- Suitable for downstream binomial regression

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
    use_quantization: bool = True
):
    """
    Extract last-layer hidden states from fine-tuned model.

    Following original CENTaUR methodology:
    1. Load base model + LoRA adapter
    2. Forward pass with prompts (NO generation)
    3. Extract hidden state from last token position
    4. Save features for downstream regression

    Args:
        base_model_path: HuggingFace model ID or local path
        adapter_path: Path to LoRA adapter directory
        dataset_path: Path to JSONL dataset (text + choice fields)
        output_path: Where to save extracted features
        model_name: Descriptive name for logging
        n_samples: Limit processing (None = all samples)
        is_local: Whether base model is local (for trust_remote_code)
        use_quantization: Use NF4 quantization (saves memory)
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
    # 4. Load LoRA Adapter
    # ============================================================

    print(f"\n4. Loading LoRA adapter...")
    print(f"   Source: {adapter_path}")

    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()  # Set to evaluation mode

    print(f"   ✓ LoRA adapter loaded")
    print(f"   Model ready for inference (eval mode)")

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

        # Forward pass (NO generation, just hidden states)
        with torch.no_grad():
            outputs = model(
                **inputs,
                output_hidden_states=True
            )

            # Extract last layer, last token position
            # outputs.hidden_states: tuple of (n_layers+1,)
            # Each element: (batch_size, seq_len, hidden_dim)
            last_layer_hidden = outputs.hidden_states[-1]  # (1, seq_len, hidden_dim)
            last_token_hidden = last_layer_hidden[0, -1, :]  # (hidden_dim,)

        all_features.append(last_token_hidden.cpu())
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
  # Extract from Qwen2.5 (first 10 samples)
  python extract_centaur_features.py --model qwen25 --n_samples 10

  # Extract from DeepSeek (all samples)
  python extract_centaur_features.py --model deepseek

  # Custom dataset and output
  python extract_centaur_features.py --model qwen25 \\
      --dataset /path/to/custom.jsonl \\
      --output /path/to/output.pth
        """
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["qwen25", "deepseek"],
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

    if args.model == "qwen25":
        base_model = "Qwen/Qwen2.5-32B-Instruct"
        adapter = "/scratch/connectome/connectome1/ko-centaur/outputs/qwen25-32b-qlora"
        dataset_default = "/scratch/connectome/connectome1/ko-centaur/data/choices13k_100.jsonl"
        output_default = "/scratch/connectome/connectome1/ko-centaur/data/features/centaur_features_qwen25.pth"
        name = "Qwen2.5-32B-QLoRA"
        is_local = False

    elif args.model == "deepseek":
        base_model = "/home/connectome/connectome1/models/deepseek-r1-distill-qwen-32b"
        adapter = "/scratch/connectome/connectome1/ko-centaur/outputs/deepseek-r1-qwen32b-qlora"
        dataset_default = "/scratch/connectome/connectome1/ko-centaur/data/choices13k_100.jsonl"
        output_default = "/scratch/connectome/connectome1/ko-centaur/data/features/centaur_features_deepseek.pth"
        name = "DeepSeek-R1-32B-QLoRA"
        is_local = True

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
        use_quantization=use_quant
    )
