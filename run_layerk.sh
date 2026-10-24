#!/usr/bin/env bash
# 2 seed, mỗi lần ~15 phút trên T4. Cần adaptive_moe.py, elastic_ctx.py, headroom.py cạnh file này.
set -u
HF="--hf wikimedia/wikipedia --config 20231101.vi --field text --mb 8 --experts 32 --L 128"
python layerk.py $HF --seed 0 2>&1 | tee log_layerk_s0.txt
python layerk.py $HF --seed 1 2>&1 | tee log_layerk_s1.txt
