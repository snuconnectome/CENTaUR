#!/bin/bash
# Ko-CENTaUR Training Monitor - Server version (runs on server, no SSH needed)
# Check every 30 minutes

JOB_ID=62407
LOG_FILE="/scratch/connectome/connectome1/ko-centaur/logs/slurm-${JOB_ID}.out"
ERR_FILE="/scratch/connectome/connectome1/ko-centaur/logs/slurm-${JOB_ID}.err"

echo "=========================================="
echo "Ko-CENTaUR Training Monitor"
echo "Job ID: $JOB_ID"
echo "Started: $(date)"
echo "=========================================="
echo ""

while true; do
    TIMESTAMP=$(date "+%Y-%m-%d %H:%M:%S")
    echo "[$TIMESTAMP] Checking training progress..."

    # Check if job is still running
    JOB_STATUS=$(squeue -j $JOB_ID -h -o %T 2>/dev/null)

    if [ -z "$JOB_STATUS" ]; then
        echo "❌ Job $JOB_ID is no longer in queue"
        echo ""
        echo "Final output:"
        tail -50 $LOG_FILE
        echo ""
        echo "Errors (if any):"
        tail -20 $ERR_FILE
        break
    fi

    echo "   Job Status: $JOB_STATUS"

    # Get training progress from stderr (tqdm output)
    echo "   Training Progress:"
    tail -5 $ERR_FILE | grep -E '%|it/s' | tail -1 | sed 's/^/   /'

    # Get last few lines from stdout for other info
    echo ""
    echo "   Recent log:"
    tail -20 $LOG_FILE | sed 's/^/   /'

    # Extract step info from stderr
    CURRENT_STEP=$(tail -1 $ERR_FILE | grep -oP '\d+(?=/22536)' | tail -1)
    if [ -n "$CURRENT_STEP" ]; then
        REMAINING=$((22536 - CURRENT_STEP))
        PROGRESS_PCT=$(echo "scale=2; $CURRENT_STEP * 100 / 22536" | bc)
        echo ""
        echo "   Current Step: $CURRENT_STEP / 22536 (${PROGRESS_PCT}%)"
        echo "   Remaining Steps: $REMAINING"
    fi

    # Check GPU usage
    GPU_INFO=$(squeue -j $JOB_ID -o '%b %R' -h 2>/dev/null)
    if [ -n "$GPU_INFO" ]; then
        echo "   GPU allocation: $GPU_INFO"
    fi

    echo ""
    echo "Next check in 30 minutes..."
    echo "=========================================="
    echo ""

    # Wait 30 minutes (1800 seconds)
    sleep 1800
done

echo ""
echo "Monitoring completed at $(date)"
