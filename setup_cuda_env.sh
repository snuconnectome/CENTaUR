#\!/bin/bash
# CUDA 13.0 Environment Configuration for ARM64

export CUDA_HOME=/usr/local/cuda-13.0
export PATH=/usr/local/cuda-13.0/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-13.0/lib64:$LD_LIBRARY_PATH

# Verify CUDA setup
echo "CUDA Environment Configured:"
echo "  CUDA_HOME: $CUDA_HOME"
echo "  nvcc version:"
nvcc --version 2>/dev/null || echo "  nvcc not found in PATH"
