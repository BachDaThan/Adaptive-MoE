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
	rm -f yui test_yui_kernels
