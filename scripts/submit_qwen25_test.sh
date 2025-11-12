#!/bin/bash
#SBATCH --job-name=qwen25-test
#SBATCH --partition=debug
#SBATCH --nodelist=node3
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:geforce:1
#SBATCH --mem=100G
#SBATCH --time=00:30:00
#SBATCH --output=/scratch/connectome/connectome1/ko-centaur/logs/qwen25_test_%j.out
#SBATCH --error=/scratch/connectome/connectome1/ko-centaur/logs/qwen25_test_%j.err

echo "=========================================="
echo "Qwen2.5-32B QLoRA Validation Test"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURM_NODELIST"
echo "GPUs: $CUDA_VISIBLE_DEVICES"
echo "Time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "=========================================="
echo ""

# Activate conda environment
source /scratch/connectome/connectome1/miniconda3/etc/profile.d/conda.sh
conda activate ko-centaur

# Python test script
python << 'ENDPYTHON'
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig

print("=" * 60)
print("Qwen2.5-32B 4-bit Loading Test")
print("=" * 60)

model_path = "/scratch/connectome/connectome1/ko-centaur/models/qwen2.5-32b-instruct"

print(f"\n[1/3] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
print(f"✅ Tokenizer loaded")

print(f"\n[2/3] Configuring 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)
print(f"✅ Quantization config created")

print(f"\n[3/3] Loading model in 4-bit (this may take 5-10 minutes)...")
print(f"Expected memory: ~18GB")
model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
)

print(f"\n✅ SUCCESS!")
print(f"Model loaded in 4-bit quantization")
print(f"Memory footprint: {model.get_memory_footprint() / 1024**3:.2f} GB")
print(f"\nGPU Memory Usage:")
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**3:.2f} GB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**3:.2f} GB")

print("\n" + "=" * 60)
print("✅ Validation test PASSED")
print("Model is ready for full training")
print("=" * 60)
ENDPYTHON

EXIT_CODE=$?

echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "✅ Validation test passed - proceed with full training"
else
    echo "❌ Validation test failed - DO NOT start full training"
fi

exit $EXIT_CODE
