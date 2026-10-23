#!/usr/bin/env bash
# Self-test trước; chỉ khi OK mới chạy thí nghiệm (cửa sổ + lưới k x W) trên 8MB Wikipedia tiếng Việt.
set -u
python elastic_ctx.py --selftest 2>&1 | tee log_ctx_test.txt
grep -q "TẤT CẢ OK" log_ctx_test.txt || { echo "Self-test lỗi, dừng."; exit 1; }
HF="--hf wikimedia/wikipedia --config 20231101.vi --field text --mb 8"
python elastic_ctx.py --train $HF 2>&1 | tee log_ctx_train.txt
python elastic_ctx.py --grid --top_p 0.999 $HF 2>&1 | tee log_ctx_grid.txt
