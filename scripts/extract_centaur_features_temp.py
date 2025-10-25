#!/usr/bin/env python3
"""
Extract hidden states from fine-tuned models for CENTaUR evaluation
Following original Binz & Schulz (2023) methodology
"""

import argparse
import json
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from tqdm import tqdm
from pathlib import Path

def extract_features(
    base_model_path: str,
    adapter_path: str,
    dataset_path: str,
    output_path: str,
    model_name: str,
    n_samples: int = None,
    is_local: bool = False
):
    print(f"\n{'='*60}")
    print(f"CENTaUR Feature Extraction: {model_name}")
    print(f"{'='*60}")

    # NF4 quantization config (save memory)
    print(f"\n1. Configuring 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True
    )

    # Load tokenizer
    print(f"\n2. Loading tokenizer from {base_model_path}")
    tokenizer_kwargs = {}
    if is_local:
        tokenizer_kwargs["trust_remote_code"] = True
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, **tokenizer_kwargs)

    # Load base model with quantization
    print(f"\n3. Loading base model with NF4 quantization...")
    model_kwargs = {
        "quantization_config": bnb_config,
        "device_map": "auto",
        "torch_dtype": torch.bfloat16
    }
    if is_local:
        model_kwargs["trust_remote_code"] = True

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        **model_kwargs
    )

    # Load LoRA adapter (NO merge to save memory)
    print(f"\n4. Loading LoRA adapter from {adapter_path}")
    model = PeftModel.from_pretrained(model, adapter_path)
    model.eval()

    print(f"\nModel loaded (quantized + LoRA, no merge)")
    print(f"Model device: {model.device}")

    # Load dataset
    print(f"\n5. Loading dataset from {dataset_path}")
    with open(dataset_path, "r") as f:
        data = [json.loads(line) for line in f]

    if n_samples:
        data = data[:n_samples]

    print(f"Dataset size: {len(data)} samples")

    # Extract features
    print(f"\n6. Extracting hidden states...")
    all_features = []
    all_labels = []

    for idx, item in enumerate(tqdm(data, desc="Extracting features")):
        prompt = item["text"]
        label = item["choice"]  # 0 for A, 1 for B

        # Tokenize
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

        # Forward pass
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=1,
                output_hidden_states=True,
                return_dict_in_generate=True
            )

            # Extract last layer hidden state from last token
            last_hidden_state = outputs.hidden_states[0][-1]  # First generation step, last layer
            last_token_hidden = last_hidden_state[0, -1, :]  # (hidden_dim,)

        all_features.append(last_token_hidden.cpu())
        all_labels.append(label)

        # Progress update every 10 samples
        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx + 1}/{len(data)} samples")

    # Stack features
    features = torch.stack(all_features)  # (n_samples, hidden_dim)
    labels = torch.tensor(all_labels)     # (n_samples,)

    print(f"\n7. Feature extraction complete!")
    print(f"   Features shape: {features.shape}")
    print(f"   Labels shape: {labels.shape}")
    print(f"   Hidden dim: {features.shape[1]}")

    # Save
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    torch.save({
        "features": features,
        "labels": labels,
        "model_name": model_name,
        "base_model_path": base_model_path,
        "adapter_path": adapter_path,
        "dataset_path": dataset_path,
        "n_samples": len(data),
        "hidden_dim": features.shape[1]
    }, output_path)

    print(f"\n✅ Features saved to: {output_path}")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract CENTaUR features from fine-tuned models"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["qwen25", "deepseek"],
        help="Model to extract features from"
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=None,
        help="Number of samples to process (default: all)"
    )
    args = parser.parse_args()

    # Model configurations
    if args.model == "qwen25":
        base_model = "Qwen/Qwen2.5-32B-Instruct"
        adapter = "/scratch/connectome/connectome1/ko-centaur/outputs/qwen25-32b-qlora"
        output = "/scratch/connectome/connectome1/ko-centaur/data/centaur_features_qwen25.pth"
        name = "Qwen2.5-32B-QLoRA"
        is_local = False
    elif args.model == "deepseek":
        base_model = "/home/connectome/connectome1/models/deepseek-r1-distill-qwen-32b"
        adapter = "/scratch/connectome/connectome1/ko-centaur/outputs/deepseek-r1-qwen32b-qlora"
        output = "/scratch/connectome/connectome1/ko-centaur/data/centaur_features_deepseek.pth"
        name = "DeepSeek-R1-32B-QLoRA"
        is_local = True

    dataset = "/scratch/connectome/connectome1/ko-centaur/data/choices13k_100.jsonl"

    extract_features(
        base_model_path=base_model,
        adapter_path=adapter,
        dataset_path=dataset,
        output_path=output,
        model_name=name,
        n_samples=args.n_samples,
        is_local=is_local
    )
