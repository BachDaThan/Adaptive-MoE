#!/bin/sh
# Kiểm tra đoán trước token: đầu ra PHẢI giống hệt khi tắt (--spec 0) và bật (--spec 4/8), với mọi nhiệt độ/seed.
# Dùng: ./test_spec.sh ten_model.yui   (biên dịch: g++ -O2 -std=c++17 -pthread yui.cpp -o yui)
M=${1:?cần file .yui}; BAD=0
for temp in 0 0.8 1.2; do for seed in 1 7 42; do
  for sp in 0 4 8; do YUI_DEBUG_IDS=1 ./yui --model "$M" --prompt "xin chào Việt" --temp $temp --seed $seed --max_new 300 --spec $sp 2>ids_$sp.txt >/dev/null; done
  if cmp -s ids_0.txt ids_4.txt && cmp -s ids_0.txt ids_8.txt; then echo "temp=$temp seed=$seed: GIỐNG HỆT"; else echo "temp=$temp seed=$seed: KHÁC!"; BAD=1; fi
done; done; rm -f ids_*.txt; exit $BAD
