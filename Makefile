# make            -> build ./yui (tối ưu cho CPU của chính máy này: -march=native)
# make portable   -> build một file chạy được trên MỌI CPU cùng kiến trúc (yui tự chọn AVX2 / NEON+dotprod lúc chạy) - dùng để phát hành
# Termux/Android: pkg install clang make ; make CXX=clang++
#   nếu báo lỗi về -march=native thì thử:  make CXX=clang++ CXXFLAGS="-O3 -mcpu=native"   hoặc   make CXXFLAGS="-O3"
# Số luồng: tự chọn (ARM: tối đa 4, x86: tối đa 6). Chỉnh:  ./yui --threads N   hoặc   YUI_THREADS=N ;  tìm số tốt nhất: ./yui --model m.yui --bench_threads
# Không cần OpenMP (nhóm luồng riêng trong yui.cpp), nên không cần -fopenmp.
CXX ?= g++
CXXFLAGS ?= -O3 -march=native
yui: yui.cpp
	$(CXX) $(CXXFLAGS) -std=c++17 -fopenmp-simd -pthread yui.cpp -o yui
portable: yui.cpp
	$(CXX) -O3 -std=c++17 -fopenmp-simd -pthread yui.cpp -o yui
test: yui
	$(CXX) -O2 -std=c++17 -pthread test_yui_kernels.cpp -o test_yui_kernels && ./test_yui_kernels
clean:
	rm -f yui test_yui_kernels libkb_core.so

# ---- kho tri thức C++17 (native/kb_core.cpp; đặc tả docs/APP_KB.md) ----
# make kb       -> libkb_core.so (chỉ xuất các hàm kb_*, không đụng tên của libyui/engine)
# make test-kb  -> pytest so lõi C++ với bộ vàng sinh từ Python (tests/golden/kb_golden.json)
# make kb-tables-> sinh lại native/kb_unicode.inc và native/kb_words.inc từ Python (khi đổi yui_rag.py/yui_graph.py)
# make kb-golden-> sinh lại bộ vàng (TZ=UTC, PYTHONHASHSEED=0)
# -ffp-contract=off: không cho trình biên dịch gộp nhân-cộng (FMA), giữ kết quả số thực giống Python (quan trọng trên ARM)
KBFLAGS ?= -O2 -std=c++17 -ffp-contract=off -fPIC -fvisibility=hidden
kb: libkb_core.so
libkb_core.so: native/kb_core.cpp native/kb_core.h native/kb_unicode.inc native/kb_words.inc
	$(CXX) $(KBFLAGS) -shared native/kb_core.cpp -o libkb_core.so
test-kb:
	python3 -m pytest -q tests/test_kb_core.py
kb-tables:
	python3 tools/gen_kb_unicode.py > native/kb_unicode.inc
	python3 tools/gen_kb_words.py > native/kb_words.inc
kb-golden:
	TZ=UTC PYTHONHASHSEED=0 python3 tests/golden/gen_golden.py
.PHONY: kb test-kb kb-tables kb-golden
