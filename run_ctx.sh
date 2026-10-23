#!/usr/bin/env bash
# Dùng: bash run_ctx.sh duong_dan_file_text_tieng_viet.txt
# Self-test trước; chỉ khi OK mới train 2 model (đủ ngữ cảnh vs cửa sổ ngẫu nhiên) và in bảng.
set -u
python elastic_ctx.py --selftest 2>&1 | tee log_ctx_test.txt
grep -q "TẤT CẢ OK" log_ctx_test.txt || { echo "Self-test lỗi, dừng."; exit 1; }
python elastic_ctx.py --train --text "${1:-}" 2>&1 | tee log_ctx_train.txt
