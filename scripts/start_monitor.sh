#!/bin/bash
# Start training monitor in background on server

MONITOR_SCRIPT="/scratch/connectome/connectome1/ko-centaur/scripts/monitor_training.sh"
LOG_FILE="/scratch/connectome/connectome1/ko-centaur/logs/monitor_62407.log"

echo "Starting Ko-CENTaUR training monitor..."
echo "Monitor log: $LOG_FILE"
echo ""

# Copy monitor script to server
scp /Users/jiookcha/Documents/git/CENTaUR/scripts/monitor_training.sh server:/scratch/connectome/connectome1/ko-centaur/scripts/

# Start monitor in background on server
ssh server "cd /scratch/connectome/connectome1/ko-centaur && nohup bash scripts/monitor_training.sh > $LOG_FILE 2>&1 &"

echo "✅ Monitor started in background on server"
echo ""
echo "Check monitor status:"
echo "  ssh server \"tail -50 $LOG_FILE\""
echo ""
echo "Stop monitor:"
echo "  ssh server \"pkill -f monitor_training.sh\""
