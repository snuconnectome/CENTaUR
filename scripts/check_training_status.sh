#!/bin/bash
# Ko-CENTaUR Training Status Checker
# Usage: Run this script on the server to check training progress

echo "=========================================="
echo "Ko-CENTaUR Training Status Check"
echo "=========================================="
echo ""

# Server paths from SETUP_GUIDE.md
WORK_DIR="/scratch/connectome/connectome1/ko-centaur"
LOG_DIR="$WORK_DIR/logs"
MODEL_DIR="$WORK_DIR/models"

echo "📁 Checking directories..."
echo "Work directory: $WORK_DIR"
echo ""

# Check if work directory exists
if [ ! -d "$WORK_DIR" ]; then
    echo "❌ Work directory does not exist: $WORK_DIR"
    exit 1
fi

echo "✅ Work directory exists"
echo ""

# 1. Check for running training processes
echo "🔍 Checking for active training processes..."
echo "=========================================="
ps aux | grep -E "train_exaone_qlora|train_psych101" | grep -v grep
if [ $? -eq 0 ]; then
    echo "✅ Training process is RUNNING"
else
    echo "⏸️  No active training process found"
fi
echo ""

# 2. Check GPU usage
echo "🎮 GPU Status..."
echo "=========================================="
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader,nounits 2>/dev/null
if [ $? -eq 0 ]; then
    echo ""
    echo "GPU Summary:"
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv
else
    echo "⚠️  nvidia-smi not available or no GPU usage"
fi
echo ""

# 3. Check for recent log files
echo "📝 Recent log files (last 7 days)..."
echo "=========================================="
if [ -d "$LOG_DIR" ]; then
    find "$LOG_DIR" -name "*.log" -mtime -7 -exec ls -lh {} \; 2>/dev/null | tail -10
    echo ""

    # Show latest log file content (last 50 lines)
    LATEST_LOG=$(find "$LOG_DIR" -name "*.log" -type f -printf '%T@ %p\n' 2>/dev/null | sort -rn | head -1 | cut -d' ' -f2-)

    if [ -n "$LATEST_LOG" ]; then
        echo "📄 Latest log file: $LATEST_LOG"
        echo "=========================================="
        echo "Last 50 lines:"
        tail -50 "$LATEST_LOG"
    fi
else
    echo "⚠️  Log directory not found: $LOG_DIR"
    echo "Creating log directory..."
    mkdir -p "$LOG_DIR"
fi
echo ""

# 4. Check for trained models
echo "🤖 Trained models..."
echo "=========================================="
if [ -d "$MODEL_DIR" ]; then
    ls -lhR "$MODEL_DIR" | head -50

    # Count model checkpoints
    MODEL_COUNT=$(find "$MODEL_DIR" -name "adapter_model.bin" -o -name "pytorch_model.bin" 2>/dev/null | wc -l)
    echo ""
    echo "Total model checkpoints found: $MODEL_COUNT"
else
    echo "⚠️  Model directory not found: $MODEL_DIR"
fi
echo ""

# 5. Check for screen sessions
echo "🖥️  Screen sessions..."
echo "=========================================="
screen -ls 2>/dev/null
if [ $? -ne 0 ]; then
    echo "No screen sessions found"
fi
echo ""

# 6. Check recent data files
echo "📊 Data files..."
echo "=========================================="
if [ -d "$WORK_DIR/data" ]; then
    ls -lh "$WORK_DIR/data"/*.jsonl 2>/dev/null | tail -5
else
    echo "⚠️  Data directory not found"
fi
echo ""

# 7. Disk usage
echo "💾 Disk usage..."
echo "=========================================="
du -sh "$WORK_DIR"/* 2>/dev/null | sort -hr | head -10
echo ""

echo "=========================================="
echo "Status check complete!"
echo "=========================================="
echo ""
echo "Quick actions:"
echo "  - View live training: tail -f $LOG_DIR/train_*.log"
echo "  - Attach to screen: screen -r ko-centaur"
echo "  - Monitor GPU: watch -n 5 nvidia-smi"
echo ""
