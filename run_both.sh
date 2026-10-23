#!/usr/bin/env bash
# Chạy 2 việc một lượt trên Colab: (1) test KV cache, (2) thí nghiệm 32 expert elastic 1..16.
set -u
python elastic_kv.py --selftest 2>&1 | tee log_kv.txt

# lm_experiment.py: tự dò tên cờ số expert
if python lm_experiment.py --help 2>&1 | grep -q -- "--n_experts"; then EXP="--n_experts 32"; else EXP="--experts 32"; fi
COMMON="$EXP --d_ff 128 --d_ff_shared 512"

python lm_experiment.py --mode random --uniform_k --k_lo 1 --k_hi 16 --eval_ks 1,2,4,8,16 $COMMON 2>&1 | tee log_elastic32.txt
python lm_experiment.py --mode fixed --k 8 $COMMON 2>&1 | tee log_fixed8.txt
python lm_experiment.py --mode fixed --k 4 $COMMON 2>&1 | tee log_fixed4.txt
