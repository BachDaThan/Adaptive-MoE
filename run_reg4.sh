#!/usr/bin/env bash
# VÒNG 4. Chạy từng phần riêng cho khỏi hết giờ Colab:   bash run_reg4.sh <phần> <seed> [seed ...]
# Thứ tự khuyến nghị: B, C, D, A.
#   B  seed mới (2 3): elastic+drop2 và fixed-k2+drop2 ở 6000 bước          ~27 phút/seed
#   C  k=1 cố định ở 6000 bước (+drop, +drop2), chạy cho seed 0 1 2 3          ~27 phút/seed
#   D  dropout 0.15 (fixed-k1, fixed-k2, elastic), seed 0 1                    ~20 phút/seed
#   A  seed mới (2 3) ở 3000 bước: fixed-k1/k2 +drop, +drop2, elastic+drop, elastic+drop2   ~40 phút/seed
# Bảng cuối gộp mọi log_reg*_s*.txt trong thư mục, nên để log các vòng trước cùng chỗ.
set -u
STAGE="${1:-B}"; shift || true
SEEDS=("$@")
if [ ${#SEEDS[@]} -eq 0 ]; then SEEDS=(2); fi
COMMON="--hf wikimedia/wikipedia --config 20231101.vi --field text --mb 8 --experts 32 --L 128"
for SEED in "${SEEDS[@]}"; do
  case "$STAGE" in
    B) python -u regcheck.py $COMMON --seed "$SEED" --steps 6000 --tag=-6k \
         --only elastic+drop2,fixed-k2+drop2 2>&1 | tee "log_reg4B_s${SEED}.txt" ;;
    C) python -u regcheck.py $COMMON --seed "$SEED" --steps 6000 --tag=-6k \
         --only fixed-k1+drop,fixed-k1+drop2 2>&1 | tee "log_reg4C_s${SEED}.txt" ;;
    D) python -u regcheck.py $COMMON --seed "$SEED" \
         --only fixed-k1+drop15,fixed-k2+drop15,elastic+drop15 2>&1 | tee "log_reg4D_s${SEED}.txt" ;;
    A) python -u regcheck.py $COMMON --seed "$SEED" \
         --only fixed-k1+drop,fixed-k1+drop2,fixed-k2+drop,fixed-k2+drop2,elastic+drop,elastic+drop2 2>&1 | tee "log_reg4A_s${SEED}.txt" ;;
    *) echo "Phần phải là A, B, C hoặc D"; exit 1 ;;
  esac
done
python summarize_reg.py "log_reg*_s*.txt"
