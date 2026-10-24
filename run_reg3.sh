#!/usr/bin/env bash
# VÒNG 3. Hai phần, chạy riêng để không vượt giới hạn phiên Colab:
#   bash run_reg3.sh A 0 1   # dropout mạnh hơn cho cả elastic và fixed (4 cấu hình, ~27 phút/seed)
#   bash run_reg3.sh B 0 1   # nhóm dropout chạy 6000 bước thay vì 3000 (4 cấu hình, ~55 phút/seed)
# Tham số đầu là phần (A hoặc B), các tham số sau là seed (mặc định 0). Phần cần log của vòng 1 và 2 để gộp bảng so sánh.
set -u
STAGE="${1:-A}"; shift || true
SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then SEEDS=(0); fi
COMMON="--hf wikimedia/wikipedia --config 20231101.vi --field text --mb 8 --experts 32 --L 128"
for SEED in "${SEEDS[@]}"; do
  if [ "$STAGE" = "A" ]; then
    python -u regcheck.py $COMMON --seed "$SEED" \
      --only elastic+drop2,elastic+drop3,fixed-k2+drop3,fixed-k1+drop2 2>&1 | tee "log_reg3A_s${SEED}.txt"
  elif [ "$STAGE" = "B" ]; then
    python -u regcheck.py $COMMON --seed "$SEED" --steps 6000 --tag=-6k \
      --only elastic+drop,elastic+drop2,fixed-k2+drop,fixed-k2+drop2 2>&1 | tee "log_reg3B_s${SEED}.txt"
  else
    echo "Phần phải là A hoặc B"; exit 1
  fi
done
python summarize_reg.py "log_reg*_s*.txt"
