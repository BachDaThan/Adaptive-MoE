#define YUI_LIB
#include "yui.cpp"
int main() {
    std::mt19937 rng(1);
    for (int cfg = 0; cfg < 4; cfg++) {
        const int R = 37, C = (cfg % 2) ? 256 : 512, grp = (cfg < 2) ? 64 : 32, gb = grp / 32, ng = C / grp, nb = C / 32;
        std::vector<int8_t> w((size_t)R * C), xq(C); std::vector<float> xs(nb), S((size_t)R * ng), x(C);
        for (auto& v : w) v = (int8_t)((int)(rng() % 3) - 1);
        for (auto& v : xq) v = (int8_t)((int)(rng() % 255) - 127);
        for (auto& v : xs) v = (rng() % 1000) / 5000.f + 0.01f;
        for (auto& v : S) v = (rng() % 1000) / 3000.f + 0.01f;
        for (int i = 0; i < C; i++) x[i] = xq[i] * xs[i / 32];
        std::vector<uint8_t> T((size_t)R * nb * 8, 0);
        for (int r = 0; r < R; r++) for (int b = 0; b < nb; b++) for (int j = 0; j < 8; j++) for (int s = 0; s < 4; s++) {
            int q = w[(size_t)r * C + b * 32 + j + 8 * s]; int code = q == -1 ? 2 : q; T[((size_t)r * nb + b) * 8 + j] |= (uint8_t)(code << (2 * s)); }
        std::vector<float> ref(R), y1(R), y2(R), y3(R);
        for (int r = 0; r < R; r++) { double a = 0; for (int g = 0; g < ng; g++) { double gs = 0; for (int b = g * gb; b < (g + 1) * gb; b++) { long si = 0; for (int i = 0; i < 32; i++) si += (long)w[(size_t)r * C + b * 32 + i] * xq[b * 32 + i]; gs += si * (double)xs[b]; } a += gs * S[(size_t)r * ng + g]; } ref[r] = (float)a; }
        rows_t2_scalar(T.data(), S.data(), nb, gb, ng, xq.data(), xs.data(), y1.data(), 0, R);
        rows_t2x_scalar(T.data(), S.data(), nb, gb, ng, x.data(), y3.data(), 0, R);
        float e1 = 0, e2 = 0, e3 = 0, mx = 0;
#ifdef YUI_X86
        rows_t2_avx2(T.data(), S.data(), nb, gb, ng, xq.data(), xs.data(), y2.data(), 0, R);
#endif
        for (int r = 0; r < R; r++) { e1 = std::max(e1, fabsf(y1[r] - ref[r])); e2 = std::max(e2, fabsf(y2[r] - ref[r])); e3 = std::max(e3, fabsf(y3[r] - ref[r])); mx = std::max(mx, fabsf(ref[r])); }
        printf("cfg C=%d grp=%d: max|ref|=%.2f err scalar %.2e avx2 %.2e exact %.2e %s\n", C, grp, mx, e1, e2, e3, (e1 < 1e-2 * 1 && e2 < 1e-2 && e3 < 5e-2) ? "OK" : "FAIL");
    }
}
