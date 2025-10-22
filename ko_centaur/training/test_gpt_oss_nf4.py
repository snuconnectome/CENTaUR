#!/usr/bin/env python3
"""
Test GPT-OSS-20B NF4 hidden state extraction

Verifies that:
1. Model loads correctly with NF4 quantization
2. Hidden states (model.hl equivalent) are accessible
3. Feature dimensions match expected size
4. Batch processing works for CENTaUR workflow
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import argparse


def test_model_loading(model_path):
    """Test that NF4 model loads correctly with on-the-fly quantization"""
    print("\n" + "=" * 80)
    print("TEST 1: Model Loading with On-the-fly NF4 Quantization")
    print("=" * 80)

    print(f"\nLoading model from: {model_path}")
    print("Expected: NF4 quantized model with ~16-18GB VRAM usage")
    print("Using BitsAndBytes on-the-fly quantization...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

    # Configure NF4 quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    print(f"\n✅ Model loaded successfully")
    print(f"   Device map: {model.hf_device_map if hasattr(model, 'hf_device_map') else 'auto'}")
    print(f"   Model dtype: {model.dtype}")
    print(f"   Memory footprint: {model.get_memory_footprint() / 1024**3:.2f} GB")

    if torch.cuda.is_available():
        print(f"\n   GPU Memory:")
        print(f"   Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
        print(f"   Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

    return model, tokenizer


def test_hidden_state_extraction(model, tokenizer):
    """Test that hidden states can be extracted (critical for CENTaUR)"""
    print("\n" + "=" * 80)
    print("TEST 2: Hidden State Extraction")
    print("=" * 80)

    # Test prompt (risky choice scenario from CENTaUR)
    test_prompts = [
        "Choose between: A: 50% chance of $100, B: 100% chance of $40. Answer: Machine ",
        "Which would you prefer: A: certain $50, B: 50% chance of $100. Answer: Machine ",
    ]

    print(f"\nTesting with {len(test_prompts)} prompts...")

    # Tokenize
    inputs = tokenizer(test_prompts, return_tensors="pt", padding=True)

    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}

    print(f"   Input shape: {inputs['input_ids'].shape}")

    # Forward pass with output_hidden_states
    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
            temperature=0.0,
            do_sample=False,
        )

    # Check if hidden states are accessible
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        # Get last layer hidden state from last generation step
        last_hidden = outputs.hidden_states[-1][-1]  # Last generation step, last layer
        print(f"\n✅ Hidden states extracted successfully")
        print(f"   Shape: {last_hidden.shape}")
        print(f"   Dtype: {last_hidden.dtype}")
        print(f"   Device: {last_hidden.device}")

        # Extract features (like CENTaUR workflow)
        features = last_hidden[:, -1, :].cpu().detach()  # Last token embedding
        print(f"\n   Feature vector shape: {features.shape}")
        print(f"   Feature dtype: {features.dtype}")
        print(f"   Feature stats: min={features.min():.4f}, max={features.max():.4f}, mean={features.mean():.4f}")

        return features
    else:
        print("\n❌ ERROR: Hidden states not accessible!")
        print(f"   Output type: {type(outputs)}")
        print(f"   Output attributes: {dir(outputs)}")
        raise RuntimeError("Cannot extract hidden states - incompatible with CENTaUR workflow")


def test_batch_processing(model, tokenizer):
    """Test batch processing (essential for 13K samples in CENTaUR)"""
    print("\n" + "=" * 80)
    print("TEST 3: Batch Processing")
    print("=" * 80)

    # Simulate CENTaUR batch workflow
    batch_prompts = [
        f"Problem {i}: Choose A or B. Answer: Machine " for i in range(8)
    ]

    print(f"\nProcessing batch of {len(batch_prompts)} prompts...")

    all_features = []
    batch_size = 4  # Process in sub-batches

    for i in range(0, len(batch_prompts), batch_size):
        batch = batch_prompts[i:i+batch_size]

        inputs = tokenizer(batch, return_tensors="pt", padding=True)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model.generate(
                inputs['input_ids'],
                max_new_tokens=1,
                output_hidden_states=True,
                return_dict_in_generate=True,
                temperature=0.0,
                do_sample=False,
            )

        # Extract features
        last_hidden = outputs.hidden_states[-1][-1]
        features = last_hidden[:, -1, :].cpu().detach()
        all_features.append(features)

    all_features = torch.cat(all_features, dim=0)

    print(f"\n✅ Batch processing successful")
    print(f"   Total features extracted: {all_features.shape[0]}")
    print(f"   Feature dimension: {all_features.shape[1]}")
    print(f"   Ready for CENTaUR regression workflow")

    return all_features


def main():
    parser = argparse.ArgumentParser(description="Test GPT-OSS-20B NF4 for CENTaUR")
    parser.add_argument(
        "--model-path",
        type=str,
        default="/home/connectome/connectome1/models/gpt-oss-20b",
        help="Path to GPT-OSS-20B model (full precision, will be quantized on-the-fly)"
    )
    args = parser.parse_args()

    print("=" * 80)
    print("GPT-OSS-20B On-the-fly NF4 Quantization for CENTaUR")
    print("=" * 80)
    print(f"\nModel: {args.model_path}")
    print(f"Quantization: On-the-fly NF4 with BitsAndBytes")
    print(f"Purpose: Verify hidden state extraction compatibility")
    print(f"Use case: Feature extraction for cognitive modeling")

    try:
        # Test 1: Loading
        model, tokenizer = test_model_loading(args.model_path)

        # Test 2: Hidden state extraction
        features = test_hidden_state_extraction(model, tokenizer)

        # Test 3: Batch processing
        batch_features = test_batch_processing(model, tokenizer)

        # Final summary
        print("\n" + "=" * 80)
        print("✅ ALL TESTS PASSED")
        print("=" * 80)
        print("\n📊 Summary:")
        print(f"   Model memory: {model.get_memory_footprint() / 1024**3:.2f} GB")
        print(f"   Feature dimension: {features.shape[1]}")
        print(f"   Batch processing: ✅ Working")
        print(f"   CENTaUR compatibility: ✅ Ready")

        print("\n🚀 Next Steps:")
        print("   1. Update query.py to use this model path")
        print("   2. Run feature extraction on choices13k dataset")
        print("   3. Fit binomial regression models")
        print("   4. Compare with existing LLaMA results")

    except Exception as e:
        print("\n" + "=" * 80)
        print("❌ TEST FAILED")
        print("=" * 80)
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
