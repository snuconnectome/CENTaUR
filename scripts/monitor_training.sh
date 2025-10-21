#!/bin/bash
# Ko-CENTaUR Training Monitor - Check every 30 minutes

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
    JOB_STATUS=$(ssh server "squeue -j $JOB_ID -h -o %T" 2>/dev/null)

    if [ -z "$JOB_STATUS" ]; then
        echo "❌ Job $JOB_ID is no longer in queue"
        echo ""
        echo "Final output:"
        ssh server "tail -50 $LOG_FILE"
        echo ""
        echo "Errors (if any):"
        ssh server "tail -20 $ERR_FILE"
        break
    fi

    echo "   Job Status: $JOB_STATUS"

    # Get last 30 lines of log
    echo "   Recent log:"
    ssh server "tail -30 $LOG_FILE" | sed 's/^/   /'

    # Check for training progress indicators
    PROGRESS=$(ssh server "grep -E 'step|epoch|loss' $LOG_FILE 2>/dev/null | tail -3")
    if [ -n "$PROGRESS" ]; then
        echo ""
        echo "   Training progress:"
        echo "$PROGRESS" | sed 's/^/   /'
    fi

    # Check GPU usage
    GPU_INFO=$(ssh server "squeue -j $JOB_ID -o '%b %R' -h" 2>/dev/null)
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
