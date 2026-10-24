#!/usr/bin/env bash
# Dùng: bash run_reg.sh 0      (một seed ~40 phút trên T4: 6 cấu hình, 7 dòng kết quả)
#       bash run_reg.sh 1      (chạy thêm seed khác nếu còn thời gian)
#       python summarize_reg.py "log_reg_s*.txt"
set -u
SEED="${1:-0}"
python regcheck.py --hf wikimedia/wikipedia --config 20231101.vi --field text --mb 8 --experts 32 --L 128 --seed "$SEED" 2>&1 | tee "log_reg_s${SEED}.txt"
python summarize_reg.py "log_reg_s${SEED}.txt"
