#!/usr/bin/env bash
# Dùng: bash run_schedule.sh 0   (seed; chạy thêm seed 1 nếu còn thời gian)
set -u
SEED="${1:-0}"
python schedule_train.py --hf wikimedia/wikipedia --config 20231101.vi --field text --mb 8 --experts 32 --L 128 --seed "$SEED" 2>&1 | tee "log_schedule_s${SEED}.txt"
