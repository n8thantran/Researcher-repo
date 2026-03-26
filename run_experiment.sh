#!/bin/bash
# Run a single experiment
# Usage: ./run_experiment.sh model_name dataset_name

MODEL=$1
DATASET=$2
EPOCHS=${3:-250}
LOG_DIR=/workspace/logs
RESULTS_DIR=/workspace/results

mkdir -p $LOG_DIR $RESULTS_DIR

EXP_NAME="${MODEL}_${DATASET}"
LOG_FILE="${LOG_DIR}/${EXP_NAME}.log"
RESULTS_FILE="${RESULTS_DIR}/${EXP_NAME}/results.json"

# Skip if already completed
if [ -f "$RESULTS_FILE" ]; then
    echo "SKIPPING $EXP_NAME (already completed)"
    exit 0
fi

echo "Starting $EXP_NAME..."
python /workspace/train.py --model $MODEL --dataset $DATASET --epochs $EPOCHS --batch_size 64 --num_workers 4 > "$LOG_FILE" 2>&1
echo "Finished $EXP_NAME"
