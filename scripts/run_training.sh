#!/bin/bash
# Runs train.py with full output logging.
# Usage: bash scripts/run_training.sh [train.py args...]
LOG=output/train_$(date +%s).log
echo "Logging to $LOG"
python -m src.train "$@" 2>&1 | tee "$LOG"
EXIT_CODE=${PIPESTATUS[0]}
echo "Training finished — exit code $EXIT_CODE"
exit $EXIT_CODE
