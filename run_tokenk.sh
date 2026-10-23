#!/usr/bin/env bash
# 3 lần chạy một lượt (~35-40 phút trên T4): seed 1, seed 2 (train ngẫu nhiên) và seed 0 train bằng luật 'rare'.
# Seed 0 train ngẫu nhiên bạn đã có. Cần adaptive_moe.py 0.3.1, elastic_ctx.py, headroom.py cạnh file này.
set -u
HF="--hf wikimedia/wikipedia --config 20231101.vi --field text --mb 8 --experts 32 --L 128"
python tokenk.py $HF --seed 1 2>&1 | tee log_tokenk_s1.txt
python tokenk.py $HF --seed 2 2>&1 | tee log_tokenk_s2.txt
python tokenk.py $HF --seed 0 --train_mode rare 2>&1 | tee log_tokenk_rare.txt
