#!/bin/bash
#SBATCH --job-name=deepseek-r1-qwen32b-test
#SBATCH --partition=debug
#SBATCH --nodelist=node3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:geforce:1
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/connectome/connectome1/ko-centaur/logs/deepseek_r1_qwen32b_test_%j.out
#SBATCH --error=/scratch/connectome/connectome1/ko-centaur/logs/deepseek_r1_qwen32b_test_%j.err

echo "=========================================="
echo "DeepSeek-R1-Distill-Qwen-32B NF4 Validation Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

echo "✅ Model: DeepSeek-R1-Distill-Qwen-32B"
echo "   Parameters: 32B"
echo "   Performance: Outperforms OpenAI o1-mini"
echo "   Expected VRAM: ~20GB with NF4 quantization"
echo ""

# Verify model directory exists
MODEL_DIR="/home/connectome/connectome1/models/deepseek-r1-distill-qwen-32b"
if [ ! -d "$MODEL_DIR" ]; then
    echo "❌ ERROR: Model directory not found: $MODEL_DIR"
    echo "Please run download script first"
    exit 1
fi

# Check if model files are complete
if [ ! -f "$MODEL_DIR/config.json" ]; then
    echo "❌ ERROR: Model files incomplete (config.json missing)"
    echo "Download may still be in progress"
    exit 1
fi

echo "✅ Model directory verified: $MODEL_DIR"
echo ""

# Activate conda environment
echo "Activating conda environment..."
source /scratch/connectome/connectome1/miniconda3/etc/profile.d/conda.sh
conda activate ko-centaur

# Verify environment
echo ""
echo "Python environment:"
python --version
echo ""

echo "Library versions:"
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import bitsandbytes; print(f'BitsAndBytes: {bitsandbytes.__version__}')"
echo ""

# GPU status
echo "GPU Status:"
nvidia-smi --query-gpu=index,name,memory.total,memory.free --format=csv
echo ""

# Run validation test
echo "=========================================="
echo "Running NF4 Validation Test"
echo "Expected: Model loads with ~20GB VRAM"
echo "=========================================="
echo ""

cd /scratch/connectome/connectome1/ko-centaur

# Python test script
python << 'ENDPYTHON'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print("=" * 80)
print("DeepSeek-R1-Distill-Qwen-32B NF4 Loading Test")
print("Model: 32B parameters (distilled from 671B DeepSeek-R1)")
print("Performance: Outperforms OpenAI o1-mini")
print("=" * 80)

model_path = "/home/connectome/connectome1/models/deepseek-r1-distill-qwen-32b"

print(f"\n[1/4] Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"✅ Tokenizer loaded")
except Exception as e:
    print(f"❌ Tokenizer loading failed: {e}")
    exit(1)

print(f"\n[2/4] Configuring NF4 quantization...")
print(f"Expected VRAM: ~20GB")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
print(f"✅ Quantization config created")

print(f"\n[3/4] Loading model in NF4 (this may take 3-5 minutes)...")
try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
    )

    print(f"\n✅ Model loaded successfully!")
    print(f"Memory footprint: {model.get_memory_footprint() / 1024**3:.2f} GB")
    print(f"\nGPU Memory Usage:")
    print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

except torch.cuda.OutOfMemoryError as e:
    print(f"\n❌ Out of Memory")
    print(f"Model requires more VRAM than available")
    exit(1)
except Exception as e:
    print(f"\n❌ Model loading failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print(f"\n[4/4] Testing hidden state extraction...")
try:
    # Test prompt (reasoning task)
    test_prompt = "What is 15 * 24? Let's think step by step:"

    inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        outputs = model.generate(
            inputs['input_ids'],
            max_new_tokens=1,
            output_hidden_states=True,
            return_dict_in_generate=True,
            temperature=0.0,
            do_sample=False,
        )

    # Extract hidden states (critical for CENTaUR)
    if hasattr(outputs, 'hidden_states') and outputs.hidden_states:
        last_hidden = outputs.hidden_states[-1][-1]
        features = last_hidden[:, -1, :].cpu().detach()

        print(f"✅ Hidden states extracted successfully")
        print(f"   Feature vector shape: {features.shape}")
        print(f"   Feature dtype: {features.dtype}")
        print(f"   Ready for CENTaUR workflow!")
    else:
        print(f"❌ Hidden states not accessible")
        exit(1)

except Exception as e:
    print(f"❌ Hidden state extraction failed: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("✅ ALL TESTS PASSED")
print("=" * 80)
print("\n📊 Summary:")
print(f"   Model: DeepSeek-R1-Distill-Qwen-32B")
print(f"   Memory: {model.get_memory_footprint() / 1024**3:.2f} GB")
print(f"   Feature dimension: {features.shape[1]}")
print(f"   Hidden state extraction: ✅ Working")
print(f"   CENTaUR compatibility: ✅ Ready")
print(f"   Performance: Outperforms o1-mini on benchmarks")

print("\n🚀 Next Steps:")
print("   1. Use this model for CENTaUR workflow")
print("   2. Extract features for choices13k dataset")
print("   3. Compare with GPT-OSS-20B and Qwen2.5-32B")
print("   4. Benchmark reasoning capabilities")
ENDPYTHON

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Validation test PASSED"
    echo "Model is ready for CENTaUR workflow"
    echo ""
    echo "Performance highlights (from official benchmarks):"
    echo "   - AIME 2024: 86.0 (vs o1-mini's 63.6)"
    echo "   - AIME 2025: 76.3"
    echo "   - GPQA Diamond: 61.1"
    echo "   - Better reasoning than o1-mini!"
else
    echo "❌ Validation test failed"
    echo "Check error log for details"
fi
echo "=========================================="

exit $EXIT_CODE
