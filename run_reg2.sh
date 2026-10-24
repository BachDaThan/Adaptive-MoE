#!/usr/bin/env bash
# VÒNG 2: elastic có dropout, dropout mạnh hơn (0.2), k=4 có dropout, 3 lịch k theo layer có dropout.
# ~40 phút/seed. Chạy một hoặc nhiều seed:   bash run_reg2.sh 0     hoặc     bash run_reg2.sh 0 1
# Vòng 1 (đã chạy) là run_reg.sh. Cuối cùng gộp log của cả hai vòng.
set -u
SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then SEEDS=(0); fi
for SEED in "${SEEDS[@]}"; do
  python -u regcheck.py --hf wikimedia/wikipedia --config 20231101.vi --field text --mb 8 --experts 32 --L 128 --seed "$SEED" \
    --only elastic+drop,fixed-k2+drop2,fixed-k4+drop,sched-1124+drop,sched-4211+drop,sched-1122+drop 2>&1 | tee "log_reg2_s${SEED}.txt"
done
python summarize_reg.py "log_reg*_s*.txt"
