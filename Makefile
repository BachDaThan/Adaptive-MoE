# make            -> build ./yui
# Termux/Android: pkg install clang make ; make CXX=clang++
#   nếu báo lỗi về -march=native thì thử:  make CXX=clang++ CXXFLAGS="-O3 -mcpu=native"   hoặc   make CXXFLAGS="-O3"
# Đa luồng cho lớp head (tuỳ chọn):  make CXXFLAGS="-O3 -march=native -fopenmp"
CXX ?= g++
CXXFLAGS ?= -O3 -march=native
yui: yui.cpp
	$(CXX) $(CXXFLAGS) -std=c++17 -fopenmp-simd -pthread yui.cpp -o yui
clean:
	rm -f yui
