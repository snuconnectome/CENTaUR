#!/bin/bash
#SBATCH --job-name=deepseek-v32-test
#SBATCH --partition=debug
#SBATCH --nodelist=node3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:geforce:4
#SBATCH --mem=200G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/connectome/connectome1/ko-centaur/logs/deepseek_v32_test_%j.out
#SBATCH --error=/scratch/connectome/connectome1/ko-centaur/logs/deepseek_v32_test_%j.err

echo "=========================================="
echo "DeepSeek-V3.2-Exp 8-bit Validation Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# WARNING about model size
echo "⚠️  WARNING: DeepSeek-V3.2-Exp is 685B parameters"
echo "   8-bit quantization requires ~685GB VRAM"
echo "   With 4 GPUs (96GB total): WILL NOT FIT"
echo "   This test will likely FAIL with OOM"
echo "   Consider alternative approaches:"
echo "   - Use smaller model for CENTaUR workflow"
echo "   - Use model API instead of local inference"
echo "   - Use quantized versions if available"
echo ""

# Verify model directory exists
MODEL_DIR="/scratch/connectome/connectome1/ko-centaur/models/deepseek-v3.2-exp"
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

# Run validation test with 8-bit quantization (more conservative than 4-bit)
echo "=========================================="
echo "Running 8-bit Validation Test"
echo "Expected: OOM error due to model size"
echo "=========================================="
echo ""

cd /scratch/connectome/connectome1/ko-centaur

# Python test script
python << 'ENDPYTHON'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print("=" * 80)
print("DeepSeek-V3.2-Exp 8-bit Loading Test")
print("Model: 685B parameters")
print("=" * 80)

model_path = "/scratch/connectome/connectome1/ko-centaur/models/deepseek-v3.2-exp"

print(f"\n[1/3] Loading tokenizer...")
try:
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    print(f"✅ Tokenizer loaded")
except Exception as e:
    print(f"❌ Tokenizer loading failed: {e}")
    exit(1)

print(f"\n[2/3] Configuring 8-bit quantization...")
print(f"Expected VRAM: ~685GB (WILL NOT FIT on 96GB total)")
bnb_config = BitsAndBytesConfig(
    load_in_8bit=True,
    llm_int8_threshold=6.0,
)
print(f"✅ Quantization config created")

print(f"\n[3/3] Attempting to load model in 8-bit...")
print(f"WARNING: This will likely fail with OOM")
print(f"Available GPU memory: ~96GB total across 4 GPUs")
print(f"Required memory: ~685GB for 8-bit")

try:
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
        torch_dtype=torch.float16,
        max_memory={0: "20GB", 1: "20GB", 2: "20GB", 3: "20GB"},  # Conservative limits
    )

    print(f"\n✅ UNEXPECTED SUCCESS!")
    print(f"Model loaded in 8-bit quantization")
    print(f"Memory footprint: {model.get_memory_footprint() / 1024**3:.2f} GB")
    print(f"\nGPU Memory Usage:")
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}:")
        print(f"  Allocated: {torch.cuda.memory_allocated(i) / 1024**3:.2f} GB")
        print(f"  Reserved: {torch.cuda.memory_reserved(i) / 1024**3:.2f} GB")

except torch.cuda.OutOfMemoryError as e:
    print(f"\n❌ EXPECTED FAILURE: Out of Memory")
    print(f"Model is too large for available hardware")
    print(f"\nRecommendations:")
    print(f"1. Use smaller models (Qwen2.5-32B, GPT-OSS-20B)")
    print(f"2. Use DeepSeek API for inference")
    print(f"3. Use pre-quantized versions if available")
    print(f"4. Consider distributed inference across multiple nodes")
    exit(1)
except Exception as e:
    print(f"\n❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 80)
print("✅ Validation test PASSED (unlikely)")
print("=" * 80)
ENDPYTHON

EXIT_CODE=$?

echo ""
echo "=========================================="
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Validation test passed (unexpected)"
    echo "Model is ready for CENTaUR workflow"
else
    echo "❌ Validation test failed (expected)"
    echo "Model too large for available hardware"
    echo ""
    echo "Recommendations:"
    echo "1. Use Qwen2.5-32B (62GB, fits on your GPUs)"
    echo "2. Use GPT-OSS-20B (13GB, fits on your GPUs)"
    echo "3. Use DeepSeek API instead of local inference"
fi
echo "=========================================="

exit $EXIT_CODE
