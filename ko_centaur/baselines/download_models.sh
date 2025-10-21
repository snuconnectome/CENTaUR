#!/bin/bash
#
# Automated baseline model download script
# Usage: bash download_models.sh [model_names...]
#

set -e  # Exit on error

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=================================================="
echo "Ko-CENTaUR Baseline Model Download"
echo "=================================================="
echo ""

# Check if Python script exists
if [ ! -f "$SCRIPT_DIR/download_models.py" ]; then
    echo "Error: download_models.py not found!"
    exit 1
fi

# Run Python download script
cd "$PROJECT_ROOT"

if [ $# -eq 0 ]; then
    echo "Downloading all baseline models..."
    python -m baselines.download_models
else
    echo "Downloading specific models: $@"
    python -m baselines.download_models --models "$@"
fi

echo ""
echo "=================================================="
echo "Download script completed"
echo "=================================================="
