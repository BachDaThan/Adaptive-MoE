// Kiểm tra từng kernel SIMD (AVX2 / NEON / NEON+dotprod) cho kết quả khớp kernel thường, mọi độ dài cột (kể cả không chia hết 32).
//   make test
#define YUI_LIB
#include "yui.cpp"
int main() {
    std::mt19937 rng(1); int bad = 0;
    for (int C : {1, 7, 31, 32, 33, 64, 100, 256, 512, 1000}) {
        const int R = 37; vector<int8_t> W((size_t)R * C); vector<float> x(C), sc(R), y0(R), y1(R), y2(R);
        for (auto& v : W) v = (int8_t)((int)(rng() % 255) - 127);
        for (auto& v : x) v = ((int)(rng() % 2001) - 1000) / 500.f * ((rng() % 9 == 0) ? 20.f : 1.f);
        for (auto& v : sc) v = 0.01f + (rng() % 100) / 1000.f;
        vector<int8_t> q((C + 31) / 32 * 32); vector<float> xs((C + 31) / 32); quant_act(x.data(), C, q.data(), xs.data());
        rows_q8_scalar(W.data(), C, q.data(), xs.data(), sc.data(), y0.data(), 0, R);
        auto cmp = [&](const char* nm, vector<float>& y) {
            double m = 0; for (int r = 0; r < R; r++) m = std::max(m, (double)fabsf(y[r] - y0[r]) / (fabsf(y0[r]) + 1e-3));
            if (m > 1e-4) { printf("C=%d %s lệch %.3g\n", C, nm, m); bad++; }
        };
#ifdef YUI_X86
        if (isa() == ISA_AVX2) { rows_q8_avx2(W.data(), C, q.data(), xs.data(), sc.data(), y1.data(), 0, R); cmp("avx2", y1); }
#endif
#ifdef YUI_ARM64
        rows_q8_neon(W.data(), C, q.data(), xs.data(), sc.data(), y1.data(), 0, R); cmp("neon", y1);
        if (isa() == ISA_DOT) { rows_q8_dot(W.data(), C, q.data(), xs.data(), sc.data(), y2.data(), 0, R); cmp("dotprod", y2); }
#endif
    }
    // nhóm luồng: tổng 1..N tính song song phải đúng
    pool().resize(4); std::vector<int> out(1000, 0);
    for (int rep = 0; rep < 200; rep++) { par_for(1000, [&](int i) { out[i] += i; }); }
    for (int i = 0; i < 1000; i++) if (out[i] != 200 * i) { printf("pool sai tại %d\n", i); bad++; break; }
    printf(bad ? "THẤT BẠI\n" : "OK: kernel + nhóm luồng (isa=%s)\n", isa_name()); return bad;
}
