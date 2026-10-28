// yui.cpp - bộ chạy (runtime) C++ cho Adaptive-MoE: KHÔNG cần PyTorch/Python. Một file, chỉ dùng thư viện chuẩn + POSIX.
//
// Build:   g++ -O3 -march=native -fopenmp-simd -std=c++17 -pthread yui.cpp -o yui         (Linux / Termux: pkg install clang)
//          thêm -fopenmp nếu muốn đa luồng cho phép nhân ma trận lớp head
// Dùng:    ./yui --model model.yui                          chat trong terminal (cần model đã SFT; xem export_yui.py)
//          ./yui --model model.yui --prompt "Hà Nội là"     viết tiếp văn bản thuần
//          ./yui --model model.yui --serve --port 11434 --api_key MAT_KHAU      server giống Ollama/OpenAI
//          ./yui --model model.yui --bench                  đo tốc độ token/giây
// Kiểm tra: ./yui --model model.yui --encode "xin chào"    in id token (so với tokenizers của Hugging Face)
//           ./yui --model model.yui --ids 5,17,9 --k 2 --logits out.bin   ghi logits float32 từng vị trí (so với PyTorch)
//
// Định dạng .yui: v1 (float32 hoặc int8 theo hàng) và v2 (mỗi ma trận một kiểu: float32 / int8 / q4 = 4-bit theo khối 32, xem export_yui.py --quant q4).
// Phép tính khớp adaptive_moe.py + elastic_ctx.py: LayerNorm -> attention RoPE (KV cache) -> MoE top-p (shared expert luôn bật,
// expert SwiGLU, trọng số = prob chuẩn hoá * strength). Chat: "User: ...\nYui: ...<eos>" như sft_format.py.
#ifdef _WIN32
#define _CRT_SECURE_NO_WARNINGS
#define NOMINMAX
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <malloc.h>
#endif
#include <algorithm>
#include <array>
#include <cctype>
#include <ctime>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <mutex>
#include <random>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>
#include <memory>
#include <stdexcept>
#include <type_traits>
#if defined(__x86_64__) || defined(_M_X64) || defined(__i386__) || defined(_M_IX86)
#define YUI_X86 1
#include <immintrin.h>
#if defined(_MSC_VER)
#include <intrin.h>
#endif
#elif defined(__aarch64__) || defined(_M_ARM64)
#define YUI_ARM64 1
#include <arm_neon.h>
#if defined(__linux__) || defined(__ANDROID__)
#include <sys/auxv.h>
#endif
#endif
#ifndef _WIN32
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
// thuộc tính "target": cho phép biên dịch kernel AVX2 / dotprod trong file không dùng -mavx2 (chọn lúc chạy, xem detect_isa)
#if defined(_MSC_VER)
#define YUI_TGT_AVX2
#define YUI_TGT_DOT
#else
#define YUI_TGT_AVX2 __attribute__((target("avx2,fma")))
#if defined(__clang__)
#define YUI_TGT_DOT __attribute__((target("dotprod")))
#else
#define YUI_TGT_DOT __attribute__((target("arch=armv8.2-a+dotprod")))
#endif
#endif
#ifndef YUI_LIB   // server HTTP chỉ có ở bản dòng lệnh (POSIX); bản thư viện cho app (-DYUI_LIB) không cần
#include <arpa/inet.h>
#include <netinet/in.h>
#include <signal.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

using std::string;
using std::vector;

struct YuiError : std::runtime_error { using std::runtime_error::runtime_error; };
[[noreturn]] static void die(const string& m) { throw YuiError(m); }

// ============================ phần cứng: chọn SIMD lúc chạy + nhóm luồng ============================
// Một file chạy được trên mọi CPU: lúc khởi động đo CPU (AVX2+FMA trên x86, NEON / NEON+dotprod trên ARM64) rồi chọn kernel nhanh nhất.
// YUI_ISA=scalar ép chạy kernel thường (để so kết quả); YUI_EXACT=1 tính bằng float như PyTorch (chậm hơn, dùng khi kiểm tra);
// YUI_THREADS=N chọn số luồng; YUI_MMAP=0/1 tắt/bật nạp model bằng mmap.
enum Isa { ISA_SCALAR = 0, ISA_AVX2 = 1, ISA_NEON = 2, ISA_DOT = 3 };
static int detect_isa() {
    if (const char* e = getenv("YUI_ISA")) if (string(e) == "scalar") return ISA_SCALAR;
#if defined(YUI_X86)
  #if defined(_MSC_VER)
    int r[4]; __cpuid(r, 0);
    if (r[0] >= 7) {
        __cpuidex(r, 1, 0);
        const bool osxsave = (r[2] >> 27) & 1, avx = (r[2] >> 28) & 1, fma = (r[2] >> 12) & 1;
        if (osxsave && avx && fma && (_xgetbv(0) & 6) == 6) { __cpuidex(r, 7, 0); if ((r[1] >> 5) & 1) return ISA_AVX2; }
    }
    return ISA_SCALAR;
  #else
    __builtin_cpu_init();
    return (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma")) ? ISA_AVX2 : ISA_SCALAR;
  #endif
#elif defined(YUI_ARM64)
  #if defined(__linux__) || defined(__ANDROID__)
    return (getauxval(AT_HWCAP) & (1UL << 20)) ? ISA_DOT : ISA_NEON;   // HWCAP_ASIMDDP
  #elif defined(__ARM_FEATURE_DOTPROD)
    return ISA_DOT;
  #else
    return ISA_NEON;
  #endif
#else
    return ISA_SCALAR;
#endif
}
static int isa() { static const int v = detect_isa(); return v; }
static const char* isa_name() { static const char* n[] = {"scalar", "avx2+fma", "neon", "neon+dotprod"}; return n[isa()]; }
static bool g_exact = [] { const char* e = getenv("YUI_EXACT"); return e && *e == '1'; }();

// ---- tích vô hướng int8 x int8: kích hoạt (activation) được lượng tử theo khối 32 phần tử, mỗi khối một hệ số (như Q8_0 của llama.cpp);
//      trọng số vẫn đúng như trong file .yui (int8, một hệ số mỗi hàng) nên KHÔNG cần xuất lại model.
static inline float tail_q8(const int8_t* w, const int8_t* xq, const float* xs, int n) {   // phần lẻ cuối hàng (n % 32)
    const int nb = n / 32, lo = nb * 32; if (lo == n) return 0.f;
    int s = 0; for (int i = lo; i < n; i++) s += (int)w[i] * (int)xq[i];
    return (float)s * xs[nb];
}
static void rows_q8_scalar(const int8_t* W, int C, const int8_t* xq, const float* xs, const float* sc, float* y, int r0, int r1) {
    const int nb = (C + 31) / 32;
    for (int r = r0; r < r1; r++) {
        const int8_t* w = W + (size_t)r * C; float acc = 0.f;
        for (int b = 0; b < nb; b++) {
            const int lo = b * 32, hi = std::min(C, lo + 32); int s = 0;
            for (int i = lo; i < hi; i++) s += (int)w[i] * (int)xq[i];
            acc += (float)s * xs[b];
        }
        y[r] = acc * sc[r];
    }
}
static void rows_f32_scalar(const float* W, int C, const float* x, float* y, int r0, int r1) {
    for (int r = r0; r < r1; r++) {
        const float* w = W + (size_t)r * C; float acc = 0.f;
        #pragma omp simd reduction(+ : acc)
        for (int c = 0; c < C; c++) acc += w[c] * x[c];
        y[r] = acc;
    }
}
static void rows_q8x_scalar(const int8_t* W, int C, const float* x, const float* sc, float* y, int r0, int r1) {   // YUI_EXACT: int8 x float như bản cũ
    for (int r = r0; r < r1; r++) {
        const int8_t* w = W + (size_t)r * C; float acc = 0.f;
        #pragma omp simd reduction(+ : acc)
        for (int c = 0; c < C; c++) acc += (float)w[c] * x[c];
        y[r] = acc * sc[r];
    }
}
#ifdef YUI_X86
YUI_TGT_AVX2 static inline float hsum256(__m256 v) {
    __m128 lo = _mm256_castps256_ps128(v), hi = _mm256_extractf128_ps(v, 1);
    lo = _mm_add_ps(lo, hi); lo = _mm_add_ps(lo, _mm_movehl_ps(lo, lo)); lo = _mm_add_ss(lo, _mm_shuffle_ps(lo, lo, 1));
    return _mm_cvtss_f32(lo);
}
YUI_TGT_AVX2 static inline __m256 q8blk_avx2(const int8_t* w, const int8_t* x, __m256i one) {
    const __m256i vw = _mm256_loadu_si256((const __m256i*)w), vx = _mm256_loadu_si256((const __m256i*)x);
    // |w| (không dấu) x (x mang dấu của w): maddubs cần toán hạng đầu không dấu. |w| <= 127 nên cặp tổng <= 32258, không tràn int16.
    const __m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(vw, vw), _mm256_sign_epi8(vx, vw));
    return _mm256_cvtepi32_ps(_mm256_madd_epi16(p, one));
}
YUI_TGT_AVX2 static void rows_q8_avx2(const int8_t* W, int C, const int8_t* xq, const float* xs, const float* sc, float* y, int r0, int r1) {
    const int nb = C / 32; const __m256i one = _mm256_set1_epi16(1);
    for (int r = r0; r < r1; r++) {
        const int8_t* w = W + (size_t)r * C; __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps(); int b = 0;
        for (; b + 1 < nb; b += 2) {
            a0 = _mm256_fmadd_ps(q8blk_avx2(w + b * 32, xq + b * 32, one), _mm256_set1_ps(xs[b]), a0);
            a1 = _mm256_fmadd_ps(q8blk_avx2(w + b * 32 + 32, xq + b * 32 + 32, one), _mm256_set1_ps(xs[b + 1]), a1);
        }
        for (; b < nb; b++) a0 = _mm256_fmadd_ps(q8blk_avx2(w + b * 32, xq + b * 32, one), _mm256_set1_ps(xs[b]), a0);
        y[r] = (hsum256(_mm256_add_ps(a0, a1)) + tail_q8(w, xq, xs, C)) * sc[r];
    }
}
YUI_TGT_AVX2 static void rows_f32_avx2(const float* W, int C, const float* x, float* y, int r0, int r1) {
    const int n16 = C / 16 * 16;
    for (int r = r0; r < r1; r++) {
        const float* w = W + (size_t)r * C; __m256 a0 = _mm256_setzero_ps(), a1 = _mm256_setzero_ps();
        for (int c = 0; c < n16; c += 16) {
            a0 = _mm256_fmadd_ps(_mm256_loadu_ps(w + c), _mm256_loadu_ps(x + c), a0);
            a1 = _mm256_fmadd_ps(_mm256_loadu_ps(w + c + 8), _mm256_loadu_ps(x + c + 8), a1);
        }
        float a = hsum256(_mm256_add_ps(a0, a1)); for (int c = n16; c < C; c++) a += w[c] * x[c];
        y[r] = a;
    }
}
#endif
#ifdef YUI_ARM64
static void rows_q8_neon(const int8_t* W, int C, const int8_t* xq, const float* xs, const float* sc, float* y, int r0, int r1) {
    const int nb = C / 32;
    for (int r = r0; r < r1; r++) {
        const int8_t* w = W + (size_t)r * C; float32x4_t acc = vdupq_n_f32(0.f);
        for (int b = 0; b < nb; b++) {
            const int8x16_t w0 = vld1q_s8(w + b * 32), w1 = vld1q_s8(w + b * 32 + 16), x0 = vld1q_s8(xq + b * 32), x1 = vld1q_s8(xq + b * 32 + 16);
            int16x8_t p0 = vmull_s8(vget_low_s8(w0), vget_low_s8(x0)); p0 = vmlal_s8(p0, vget_high_s8(w0), vget_high_s8(x0));
            int16x8_t p1 = vmull_s8(vget_low_s8(w1), vget_low_s8(x1)); p1 = vmlal_s8(p1, vget_high_s8(w1), vget_high_s8(x1));
            acc = vmlaq_n_f32(acc, vcvtq_f32_s32(vpadalq_s16(vpaddlq_s16(p0), p1)), xs[b]);
        }
        y[r] = (vaddvq_f32(acc) + tail_q8(w, xq, xs, C)) * sc[r];
    }
}
YUI_TGT_DOT static void rows_q8_dot(const int8_t* W, int C, const int8_t* xq, const float* xs, const float* sc, float* y, int r0, int r1) {
    const int nb = C / 32;
    for (int r = r0; r < r1; r++) {
        const int8_t* w = W + (size_t)r * C; float32x4_t acc = vdupq_n_f32(0.f);
        for (int b = 0; b < nb; b++) {
            int32x4_t s = vdupq_n_s32(0);
            s = vdotq_s32(s, vld1q_s8(w + b * 32), vld1q_s8(xq + b * 32));
            s = vdotq_s32(s, vld1q_s8(w + b * 32 + 16), vld1q_s8(xq + b * 32 + 16));
            acc = vmlaq_n_f32(acc, vcvtq_f32_s32(s), xs[b]);
        }
        y[r] = (vaddvq_f32(acc) + tail_q8(w, xq, xs, C)) * sc[r];
    }
}
static void rows_f32_neon(const float* W, int C, const float* x, float* y, int r0, int r1) {
    const int n8 = C / 8 * 8;
    for (int r = r0; r < r1; r++) {
        const float* w = W + (size_t)r * C; float32x4_t a0 = vdupq_n_f32(0.f), a1 = vdupq_n_f32(0.f);
        for (int c = 0; c < n8; c += 8) { a0 = vfmaq_f32(a0, vld1q_f32(w + c), vld1q_f32(x + c)); a1 = vfmaq_f32(a1, vld1q_f32(w + c + 4), vld1q_f32(x + c + 4)); }
        float a = vaddvq_f32(vaddq_f32(a0, a1)); for (int c = n8; c < C; c++) a += w[c] * x[c];
        y[r] = a;
    }
}
#endif

// ---- q4: trọng số 4-bit theo khối 32 (byte j của khối chứa trọng số j ở nửa thấp, j+16 ở nửa cao; giá trị = (nibble - 8) * hệ số khối) ----
static void rows_q4_scalar(const uint8_t* Q, const float* S, int nb, const int8_t* xq, const float* xs, float* y, int r0, int r1) {
    for (int r = r0; r < r1; r++) {
        const uint8_t* q = Q + (size_t)r * nb * 16; const float* s = S + (size_t)r * nb; float acc = 0.f;
        for (int b = 0; b < nb; b++) {
            const uint8_t* p = q + b * 16; const int8_t* x = xq + b * 32; int sum = 0;
            for (int j = 0; j < 16; j++) sum += ((int)(p[j] & 15) - 8) * (int)x[j] + ((int)(p[j] >> 4) - 8) * (int)x[16 + j];
            acc += (float)sum * (s[b] * xs[b]);
        }
        y[r] = acc;
    }
}
static void rows_q4x_scalar(const uint8_t* Q, const float* S, int nb, int cols, const float* x, float* y, int r0, int r1) {   // YUI_EXACT: q4 x float
    for (int r = r0; r < r1; r++) {
        const uint8_t* q = Q + (size_t)r * nb * 16; const float* s = S + (size_t)r * nb; float acc = 0.f;
        for (int b = 0; b < nb; b++) {
            const uint8_t* p = q + b * 16; float blk = 0.f;
            for (int j = 0; j < 16; j++) {
                const int i0 = b * 32 + j, i1 = i0 + 16;
                if (i0 < cols) blk += (float)((int)(p[j] & 15) - 8) * x[i0];
                if (i1 < cols) blk += (float)((int)(p[j] >> 4) - 8) * x[i1];
            }
            acc += blk * s[b];
        }
        y[r] = acc;
    }
}
#ifdef YUI_X86
YUI_TGT_AVX2 static void rows_q4_avx2(const uint8_t* Q, const float* S, int nb, const int8_t* xq, const float* xs, float* y, int r0, int r1) {
    const __m128i m4 = _mm_set1_epi8(0x0F); const __m256i eight = _mm256_set1_epi8(8), one = _mm256_set1_epi16(1);
    for (int r = r0; r < r1; r++) {
        const uint8_t* q = Q + (size_t)r * nb * 16; const float* s = S + (size_t)r * nb; __m256 acc = _mm256_setzero_ps();
        for (int b = 0; b < nb; b++) {
            const __m128i raw = _mm_loadu_si128((const __m128i*)(q + b * 16));
            const __m128i lo = _mm_and_si128(raw, m4), hi = _mm_and_si128(_mm_srli_epi16(raw, 4), m4);
            const __m256i w = _mm256_sub_epi8(_mm256_inserti128_si256(_mm256_castsi128_si256(lo), hi, 1), eight);   // 32 trọng số int8 trong [-8, 7]
            const __m256i vx = _mm256_loadu_si256((const __m256i*)(xq + b * 32));
            const __m256i p = _mm256_maddubs_epi16(_mm256_sign_epi8(w, w), _mm256_sign_epi8(vx, w));
            acc = _mm256_fmadd_ps(_mm256_cvtepi32_ps(_mm256_madd_epi16(p, one)), _mm256_set1_ps(s[b] * xs[b]), acc);
        }
        y[r] = hsum256(acc);
    }
}
#endif
#ifdef YUI_ARM64
static void rows_q4_neon(const uint8_t* Q, const float* S, int nb, const int8_t* xq, const float* xs, float* y, int r0, int r1) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F); const int8x16_t eight = vdupq_n_s8(8);
    for (int r = r0; r < r1; r++) {
        const uint8_t* q = Q + (size_t)r * nb * 16; const float* s = S + (size_t)r * nb; float32x4_t acc = vdupq_n_f32(0.f);
        for (int b = 0; b < nb; b++) {
            const uint8x16_t raw = vld1q_u8(q + b * 16);
            const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, m4)), eight), hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);
            const int8x16_t x0 = vld1q_s8(xq + b * 32), x1 = vld1q_s8(xq + b * 32 + 16);
            int16x8_t p0 = vmull_s8(vget_low_s8(lo), vget_low_s8(x0)); p0 = vmlal_s8(p0, vget_high_s8(lo), vget_high_s8(x0));
            int16x8_t p1 = vmull_s8(vget_low_s8(hi), vget_low_s8(x1)); p1 = vmlal_s8(p1, vget_high_s8(hi), vget_high_s8(x1));
            acc = vmlaq_n_f32(acc, vcvtq_f32_s32(vpadalq_s16(vpaddlq_s16(p0), p1)), s[b] * xs[b]);
        }
        y[r] = vaddvq_f32(acc);
    }
}
YUI_TGT_DOT static void rows_q4_dot(const uint8_t* Q, const float* S, int nb, const int8_t* xq, const float* xs, float* y, int r0, int r1) {
    const uint8x16_t m4 = vdupq_n_u8(0x0F); const int8x16_t eight = vdupq_n_s8(8);
    for (int r = r0; r < r1; r++) {
        const uint8_t* q = Q + (size_t)r * nb * 16; const float* s = S + (size_t)r * nb; float32x4_t acc = vdupq_n_f32(0.f);
        for (int b = 0; b < nb; b++) {
            const uint8x16_t raw = vld1q_u8(q + b * 16);
            const int8x16_t lo = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(raw, m4)), eight), hi = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(raw, 4)), eight);
            int32x4_t t = vdotq_s32(vdupq_n_s32(0), lo, vld1q_s8(xq + b * 32));
            t = vdotq_s32(t, hi, vld1q_s8(xq + b * 32 + 16));
            acc = vmlaq_n_f32(acc, vcvtq_f32_s32(t), s[b] * xs[b]);
        }
        y[r] = vaddvq_f32(acc);
    }
}
#endif

// ---- nhóm luồng (không cần OpenMP nên chạy giống nhau trên Android / Windows / Linux) ----
// Mỗi lần run() chia n tác vụ cho các luồng nền + luồng gọi; luồng nền quay vòng chờ rất ngắn rồi ngủ nên không tốn pin lúc rảnh.
static inline void cpu_relax() {
#if defined(YUI_X86)
    _mm_pause();
#elif defined(YUI_ARM64) && !defined(_MSC_VER)
    __asm__ __volatile__("yield");
#else
    std::this_thread::yield();
#endif
}
struct Pool {
    typedef void (*Fn)(void*, int);
    std::mutex run_mu, mu; std::condition_variable cv;
    std::atomic<uint64_t> ticket{0};   // 32 bit cao: thế hệ lần chạy; 32 bit thấp: số tác vụ đã có người nhận
    std::atomic<Fn> fn{nullptr}; std::atomic<void*> ctx{nullptr};
    std::atomic<int> ntask{0}, remaining{0}, sleepers{0}, alive{0};
    std::atomic<bool> stop{false};
    int nthreads = 1;   // tính cả luồng gọi

    void claim(uint32_t g) {
        uint64_t cur = ticket.load();
        while (true) {
            if ((uint32_t)(cur >> 32) != g) return;
            const uint32_t idx = (uint32_t)cur; if ((int)idx >= ntask.load()) return;
            if (ticket.compare_exchange_weak(cur, cur + 1)) {
                Fn f = fn.load(); void* c = ctx.load(); f(c, (int)idx);
                remaining.fetch_sub(1); cur = ticket.load();
            }
        }
    }
    void worker() {
        uint32_t seen = (uint32_t)(ticket.load() >> 32);
        while (true) {
            auto t0 = std::chrono::steady_clock::now(); int spins = 0; uint64_t t;
            while (true) {
                if (stop.load()) { alive.fetch_sub(1); return; }
                t = ticket.load();
                if ((uint32_t)(t >> 32) != seen) break;
                if (++spins < 256) { cpu_relax(); continue; }
                if (std::chrono::steady_clock::now() - t0 < std::chrono::microseconds(150)) { spins = 0; continue; }   // quay vòng ~150µs rồi mới ngủ
                std::unique_lock<std::mutex> lk(mu); sleepers.fetch_add(1);
                cv.wait_for(lk, std::chrono::milliseconds(20), [&] { return stop.load() || (uint32_t)(ticket.load() >> 32) != seen; });
                sleepers.fetch_sub(1); t0 = std::chrono::steady_clock::now(); spins = 0;
            }
            seen = (uint32_t)(t >> 32); claim(seen);
        }
    }
    void resize(int n) {
        n = std::max(1, std::min(n, 64)); std::lock_guard<std::mutex> lk(run_mu);
        stop.store(true); { std::lock_guard<std::mutex> l2(mu); cv.notify_all(); }
        while (alive.load() > 0) std::this_thread::sleep_for(std::chrono::microseconds(200));
        stop.store(false); nthreads = n;
        for (int i = 1; i < n; i++) { alive.fetch_add(1); std::thread([this] { worker(); }).detach(); }
    }
    void run(int n, Fn f, void* c) {
        if (n <= 0) return;
        if (nthreads <= 1 || n == 1 || !run_mu.try_lock()) { for (int i = 0; i < n; i++) f(c, i); return; }   // đang bận (nhiều phiên cùng lúc) -> chạy tại chỗ
        fn.store(f); ctx.store(c); ntask.store(n); remaining.store(n);
        const uint32_t g = (uint32_t)(ticket.load() >> 32) + 1;
        ticket.store((uint64_t)g << 32);
        if (sleepers.load() > 0) { std::lock_guard<std::mutex> lk(mu); cv.notify_all(); }
        claim(g);
        for (int spins = 0; remaining.load() > 0; spins++) { if (spins < 2000) cpu_relax(); else std::this_thread::yield(); }
        run_mu.unlock();
    }
};
static int default_threads() {
    if (const char* e = getenv("YUI_THREADS")) { int v = atoi(e); if (v > 0) return std::min(v, 64); }
    int hw = (int)std::thread::hardware_concurrency(); if (hw <= 0) hw = 1;
#if defined(YUI_ARM64)
    return std::min(hw, 4);   // điện thoại big.LITTLE: dùng cả lõi nhỏ thường chậm hơn; chỉnh bằng YUI_THREADS hoặc yui_set_threads
#else
    return std::min(hw, 6);
#endif
}
static Pool& pool() {   // cố ý không huỷ (tránh treo khi gỡ DLL trên Windows)
    static Pool* p = [] { Pool* q = new Pool(); q->resize(default_threads()); return q; }();
    return *p;
}
template <class F> static void par_for(int n, F&& f) {
    Pool& p = pool();
    if (n <= 1 || p.nthreads <= 1) { for (int i = 0; i < n; i++) f(i); return; }
    p.run(n, [](void* c, int i) { (*(typename std::remove_reference<F>::type*)c)(i); }, (void*)&f);
}

// ---- file model: mmap (Linux/Android: không nhân đôi RAM, mở gần như tức thì) hoặc đọc vào RAM ----
static void* aalloc_raw(size_t n) {
    n = (n + 63) / 64 * 64; void* p = nullptr;
#ifdef _WIN32
    p = _aligned_malloc(n, 64); if (!p) die("hết RAM");
#else
    if (posix_memalign(&p, 64, n)) die("hết RAM");
#endif
    return p;
}
static void afree(void* p) {
#ifdef _WIN32
    _aligned_free(p);
#else
    free(p);
#endif
}
static FILE* fopen_utf8(const string& p) {   // đường dẫn UTF-8 (Windows: tên người dùng có dấu tiếng Việt)
#ifdef _WIN32
    int n = MultiByteToWideChar(CP_UTF8, 0, p.c_str(), -1, nullptr, 0); if (n <= 0) return nullptr;
    std::wstring w(n, L'\0'); MultiByteToWideChar(CP_UTF8, 0, p.c_str(), -1, &w[0], n);
    return _wfopen(w.c_str(), L"rb");
#else
    return fopen(p.c_str(), "rb");
#endif
}
struct MappedFile {
    const uint8_t* data = nullptr; size_t size = 0; bool mapped = false;
    void* heap = nullptr;
#ifdef _WIN32
    HANDLE hf = INVALID_HANDLE_VALUE, hm = nullptr;
#endif
    MappedFile() = default;
    MappedFile(const MappedFile&) = delete;
    MappedFile& operator=(const MappedFile&) = delete;
    ~MappedFile() { close(); }
    void close() {
#ifdef _WIN32
        if (mapped && data) UnmapViewOfFile(data);
        if (hm) CloseHandle(hm);
        if (hf != INVALID_HANDLE_VALUE) CloseHandle(hf);
        hm = nullptr; hf = INVALID_HANDLE_VALUE;
#else
        if (mapped && data) munmap((void*)data, size);
#endif
        if (heap) afree(heap);
        heap = nullptr; data = nullptr; size = 0; mapped = false;
    }
    void open(const string& path) {
        close();
        bool want = true;
#ifdef _WIN32
        want = false;   // Windows: file đang mmap không thể đổi tên/xoá -> mặc định đọc vào RAM (đặt YUI_MMAP=1 để bật)
#endif
        if (const char* e = getenv("YUI_MMAP")) want = (*e == '1');
        if (want) {
#ifdef _WIN32
            int n = MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, nullptr, 0);
            if (n > 0) {
                std::wstring w(n, L'\0'); MultiByteToWideChar(CP_UTF8, 0, path.c_str(), -1, &w[0], n);
                hf = CreateFileW(w.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, nullptr);
                LARGE_INTEGER sz; sz.QuadPart = 0;
                if (hf != INVALID_HANDLE_VALUE && GetFileSizeEx(hf, &sz) && sz.QuadPart > 0) {
                    hm = CreateFileMappingW(hf, nullptr, PAGE_READONLY, 0, 0, nullptr);
                    void* p = hm ? MapViewOfFile(hm, FILE_MAP_READ, 0, 0, 0) : nullptr;
                    if (p) { data = (const uint8_t*)p; size = (size_t)sz.QuadPart; mapped = true; return; }
                }
                close();
            }
#else
            int fd = ::open(path.c_str(), O_RDONLY);
            if (fd >= 0) {
                struct stat st;
                if (fstat(fd, &st) == 0 && st.st_size > 0) {
                    int flags = MAP_PRIVATE;
#ifdef MAP_POPULATE
                    if (const char* e = getenv("YUI_PRELOAD")) if (*e == '1') flags |= MAP_POPULATE;   // nạp sẵn toàn bộ vào RAM (khởi động chậm hơn, token đầu nhanh hơn)
#endif
                    void* p = mmap(nullptr, (size_t)st.st_size, PROT_READ, flags, fd, 0);
                    if (p != MAP_FAILED) {
#ifdef MADV_WILLNEED
                        madvise(p, (size_t)st.st_size, MADV_WILLNEED);
#endif
                        ::close(fd); data = (const uint8_t*)p; size = (size_t)st.st_size; mapped = true; return;
                    }
                }
                ::close(fd);
            }
#endif
        }
        std::unique_ptr<FILE, int (*)(FILE*)> fp(fopen_utf8(path), &fclose); if (!fp) die("không mở được " + path);
        if (fseek(fp.get(), 0, SEEK_END) != 0) die("không đọc được " + path);
        long long n = (long long)ftell(fp.get()); if (n <= 0) die("file rỗng: " + path);
        fseek(fp.get(), 0, SEEK_SET);
        heap = aalloc_raw((size_t)n);
        if (fread(heap, 1, (size_t)n, fp.get()) != (size_t)n) die("file .yui bị cụt");
        data = (const uint8_t*)heap; size = (size_t)n;
    }
};
struct Reader {
    const uint8_t* p; const uint8_t* end; vector<void*>* mem; int ver = 1;
    void need(size_t n) const { if ((size_t)(end - p) < n) die("file .yui bị cụt"); }
    void* alloc(size_t n) { void* q = aalloc_raw(n); mem->push_back(q); return q; }
    template <class T> T get() { need(sizeof(T)); T v; memcpy(&v, p, sizeof v); p += sizeof v; return v; }
    void bytes(void* dst, size_t n) { need(n); if (n) memcpy(dst, p, n); p += n; }
    const uint8_t* skip(size_t n) { need(n); const uint8_t* r = p; p += n; return r; }
    string str() { uint16_t n = get<uint16_t>(); string s(n, '\0'); bytes(&s[0], n); return s; }
};

struct Mat {
    int rows = 0, cols = 0, type = 0; bool q = false;   // type: 0 float32, 1 int8 (hệ số mỗi hàng), 2 q4 (4-bit, hệ số mỗi khối 32)
    float* f = nullptr; const int8_t* i8 = nullptr; float* sc = nullptr;   // i8 trỏ thẳng vào file mmap (không sao chép)
    const uint8_t* q4 = nullptr; float* bs = nullptr; int nb = 0;          // q4: nb khối mỗi hàng, mỗi khối 16 byte (2 trọng số/byte) + 1 hệ số float
};
static float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t)(h & 0x8000) << 16, e = (h >> 10) & 0x1F, m = h & 0x3FF, f;
    if (e == 0) {
        if (m == 0) f = sign;
        else { e = 1; while (!(m & 0x400)) { m <<= 1; e--; } f = sign | ((e + 127 - 15) << 23) | ((m & 0x3FF) << 13); }   // số dưới chuẩn
    } else if (e == 31) f = sign | 0x7F800000u | (m << 13);
    else f = sign | ((e + 127 - 15) << 23) | (m << 13);
    float r; memcpy(&r, &f, 4); return r;
}
// v1: kiểu theo cờ toàn file (v1_quant). v2: mỗi ma trận đứng sau 1 byte kiểu (typed=false cho ma trận luôn float như router).
static Mat load_mat(Reader& r, int rows, int cols, bool v1_quant, bool typed = true) {
    Mat m; m.rows = rows; m.cols = cols;
    int t = v1_quant ? 1 : 0;
    if (typed && r.ver >= 2) t = r.get<uint8_t>();
    m.type = t; m.q = (t == 1);
    const size_t n = (size_t)rows * cols;
    if (t == 1) {
        m.sc = (float*)r.alloc(rows * sizeof(float)); r.bytes(m.sc, rows * sizeof(float));
        m.i8 = (const int8_t*)r.skip(n);
    } else if (t == 2) {
        m.nb = (cols + 31) / 32; const size_t nblk = (size_t)rows * m.nb;
        m.bs = (float*)r.alloc(nblk * sizeof(float));
        const uint8_t* h = r.skip(nblk * 2);   // hệ số fp16 -> float một lần lúc nạp; phần 4-bit dùng thẳng từ mmap
        for (size_t i = 0; i < nblk; i++) { uint16_t hv; memcpy(&hv, h + 2 * i, 2); m.bs[i] = f16_to_f32(hv); }
        m.q4 = r.skip(nblk * 16);
    } else if (t == 0) {
        m.f = (float*)r.alloc(n * sizeof(float)); r.bytes(m.f, n * sizeof(float));
    } else die("kiểu ma trận không hỗ trợ trong file .yui");
    return m;
}
static float* load_vec(Reader& r, size_t n) { float* p = (float*)r.alloc(n * sizeof(float)); r.bytes(p, n * sizeof(float)); return p; }

// kích hoạt int8 theo khối 32: q có (n+31)/32*32 byte, s có (n+31)/32 float
static void quant_act(const float* x, int n, int8_t* q, float* s) {
    const int nb = (n + 31) / 32;
    for (int b = 0; b < nb; b++) {
        const int lo = b * 32, hi = std::min(n, lo + 32); float amax = 0.f;
        for (int i = lo; i < hi; i++) amax = std::max(amax, fabsf(x[i]));
        const float d = amax / 127.f, id = d > 0.f ? 1.f / d : 0.f; s[b] = d;
        for (int i = lo; i < hi; i++) q[i] = (int8_t)lrintf(x[i] * id);
        for (int i = hi; i < lo + 32; i++) q[i] = 0;
    }
}

static void rows_q(const Mat& m, const int8_t* xq, const float* xs, float* y, int r0, int r1) {
    switch (isa()) {
#ifdef YUI_X86
    case ISA_AVX2: rows_q8_avx2(m.i8, m.cols, xq, xs, m.sc, y, r0, r1); return;
#endif
#ifdef YUI_ARM64
    case ISA_DOT: rows_q8_dot(m.i8, m.cols, xq, xs, m.sc, y, r0, r1); return;
    case ISA_NEON: rows_q8_neon(m.i8, m.cols, xq, xs, m.sc, y, r0, r1); return;
#endif
    default: rows_q8_scalar(m.i8, m.cols, xq, xs, m.sc, y, r0, r1);
    }
}
static void rows_q4(const Mat& m, const int8_t* xq, const float* xs, float* y, int r0, int r1) {
    switch (isa()) {
#ifdef YUI_X86
    case ISA_AVX2: rows_q4_avx2(m.q4, m.bs, m.nb, xq, xs, y, r0, r1); return;
#endif
#ifdef YUI_ARM64
    case ISA_DOT: rows_q4_dot(m.q4, m.bs, m.nb, xq, xs, y, r0, r1); return;
    case ISA_NEON: rows_q4_neon(m.q4, m.bs, m.nb, xq, xs, y, r0, r1); return;
#endif
    default: rows_q4_scalar(m.q4, m.bs, m.nb, xq, xs, y, r0, r1);
    }
}
static void rows_f(const Mat& m, const float* x, float* y, int r0, int r1) {
    switch (isa()) {
#ifdef YUI_X86
    case ISA_AVX2: rows_f32_avx2(m.f, m.cols, x, y, r0, r1); return;
#endif
#ifdef YUI_ARM64
    case ISA_DOT: case ISA_NEON: rows_f32_neon(m.f, m.cols, x, y, r0, r1); return;
#endif
    default: rows_f32_scalar(m.f, m.cols, x, y, r0, r1);
    }
}
struct Act { int8_t* q; float* s; };   // kích hoạt đã lượng tử (chỉ dùng khi model int8 và không YUI_EXACT)
static inline bool use_qact(bool model_q) { return model_q && !g_exact; }

// y = W x. par=true: chia hàng cho nhóm luồng nếu ma trận đủ lớn (>= ~0,5 triệu phép nhân)
static void mv(const Mat& m, const float* x, const Act& a, float* y, bool par) {
    const int R = m.rows;
    auto body = [&](int r0, int r1) {
        if (m.type == 0) rows_f(m, x, y, r0, r1);
        else if (m.type == 1) { if (g_exact) rows_q8x_scalar(m.i8, m.cols, x, m.sc, y, r0, r1); else rows_q(m, a.q, a.s, y, r0, r1); }
        else { if (g_exact) rows_q4x_scalar(m.q4, m.bs, m.nb, m.cols, x, y, r0, r1); else rows_q4(m, a.q, a.s, y, r0, r1); }
    };
    const Pool& p = pool();
    if (!par || p.nthreads <= 1 || (size_t)R * m.cols < (1u << 19)) { body(0, R); return; }
    const int ch = std::max(16, (R / (p.nthreads * 4) + 3) / 4 * 4), nt = (R + ch - 1) / ch;
    par_for(nt, [&](int t) { body(t * ch, std::min(R, (t + 1) * ch)); });
}
static void layernorm(const float* x, const float* w, const float* b, float* y, int n) {
    float mean = 0.f; for (int i = 0; i < n; i++) mean += x[i]; mean /= n;
    float var = 0.f; for (int i = 0; i < n; i++) { float d = x[i] - mean; var += d * d; } var /= n;
    float inv = 1.f / sqrtf(var + 1e-5f);
    for (int i = 0; i < n; i++) y[i] = (x[i] - mean) * inv * w[i] + b[i];
}

// ============================ tokenizer (BPE byte-level, giống HF tokenizers) ============================
struct Tok {
    vector<string> vocab; std::unordered_map<string, int> v2i;
    std::unordered_map<uint64_t, std::pair<int, int>> merges;   // (a<<32|b) -> (rank, id gộp)
    int eos = -1;
    vector<std::array<uint32_t, 3>> ranges;                      // [start,end,cls] cls 1=chữ(\p{L}) 2=số(\p{N})
    string b2u[256]; std::unordered_map<uint32_t, int> u2b;

    static void put_utf8(string& s, uint32_t cp) {
        if (cp < 0x80) s += (char)cp;
        else if (cp < 0x800) { s += (char)(0xC0 | (cp >> 6)); s += (char)(0x80 | (cp & 63)); }
        else if (cp < 0x10000) { s += (char)(0xE0 | (cp >> 12)); s += (char)(0x80 | ((cp >> 6) & 63)); s += (char)(0x80 | (cp & 63)); }
        else { s += (char)(0xF0 | (cp >> 18)); s += (char)(0x80 | ((cp >> 12) & 63)); s += (char)(0x80 | ((cp >> 6) & 63)); s += (char)(0x80 | (cp & 63)); }
    }
    // Đọc 1 code point; trả số byte dùng. Chuỗi sai -> U+FFFD (1 byte). incomplete=true nếu bị cụt ở cuối.
    static int get_utf8(const string& s, size_t i, uint32_t& cp, bool* incomplete = nullptr) {
        unsigned char c = s[i]; size_t n = s.size();
        int len = c < 0x80 ? 1 : (c >> 5) == 6 ? 2 : (c >> 4) == 14 ? 3 : (c >> 3) == 30 ? 4 : 0;
        if (!len) { cp = 0xFFFD; return 1; }
        if (len == 1) { cp = c; return 1; }
        if (i + len > n) {
            bool ok = true; for (size_t k = i + 1; k < n; k++) if (((unsigned char)s[k] >> 6) != 2) ok = false;
            if (ok && incomplete) *incomplete = true;
            cp = 0xFFFD; return (int)(n - i);
        }
        uint32_t v = len == 2 ? (c & 31) : len == 3 ? (c & 15) : (c & 7);
        for (int k = 1; k < len; k++) { unsigned char d = s[i + k]; if ((d >> 6) != 2) { cp = 0xFFFD; return 1; } v = (v << 6) | (d & 63); }
        if ((len == 2 && v < 0x80) || (len == 3 && v < 0x800) || (len == 4 && (v < 0x10000 || v > 0x10FFFF)) || (v >= 0xD800 && v <= 0xDFFF)) { cp = 0xFFFD; return 1; }
        cp = v; return len;
    }
    void init_bytes() {
        vector<int> bs, cs;
        for (int b = '!'; b <= '~'; b++) bs.push_back(b);
        for (int b = 0xA1; b <= 0xAC; b++) bs.push_back(b);
        for (int b = 0xAE; b <= 0xFF; b++) bs.push_back(b);
        cs = bs; int n = 0;
        for (int b = 0; b < 256; b++) if (std::find(bs.begin(), bs.end(), b) == bs.end()) { bs.push_back(b); cs.push_back(256 + n); n++; }
        for (size_t i = 0; i < bs.size(); i++) { string u; put_utf8(u, cs[i]); b2u[bs[i]] = u; u2b[cs[i]] = bs[i]; }
    }
    int cls(uint32_t cp) const {
        size_t lo = 0, hi = ranges.size();
        while (lo < hi) { size_t mid = (lo + hi) / 2; if (ranges[mid][1] < cp) lo = mid + 1; else hi = mid; }
        return (lo < ranges.size() && ranges[lo][0] <= cp) ? (int)ranges[lo][2] : 0;
    }
    static bool is_space(uint32_t c) {
        return (c >= 9 && c <= 13) || c == 32 || c == 0x85 || c == 0xA0 || c == 0x1680 || (c >= 0x2000 && c <= 0x200A) ||
               c == 0x2028 || c == 0x2029 || c == 0x202F || c == 0x205F || c == 0x3000;
    }
    // regex GPT-2: 's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
    void pretok(const string& s, vector<std::pair<size_t, size_t>>& out) const {  // trả các đoạn [byte_start, byte_end)
        vector<uint32_t> cp; vector<size_t> off;
        for (size_t i = 0; i < s.size();) { uint32_t c; int l = get_utf8(s, i, c); off.push_back(i); cp.push_back(c); i += l; }
        off.push_back(s.size());
        const size_t n = cp.size(); size_t i = 0;
        auto emit = [&](size_t a, size_t b) { out.push_back({off[a], off[b]}); };
        auto L = [&](size_t k) { return k < n && !is_space(cp[k]) && cls(cp[k]) == 1; };
        auto N = [&](size_t k) { return k < n && !is_space(cp[k]) && cls(cp[k]) == 2; };
        auto O = [&](size_t k) { return k < n && !is_space(cp[k]) && cls(cp[k]) == 0; };
        while (i < n) {
            if (cp[i] == '\'' && i + 1 < n) {
                uint32_t a = cp[i + 1], b = i + 2 < n ? cp[i + 2] : 0;
                if (a == 's' || a == 't' || a == 'm' || a == 'd') { emit(i, i + 2); i += 2; continue; }
                if ((a == 'r' && b == 'e') || (a == 'v' && b == 'e') || (a == 'l' && b == 'l')) { emit(i, i + 3); i += 3; continue; }
            }
            size_t st = (cp[i] == ' ') ? i + 1 : i, e;
            if (L(st)) { e = st; while (L(e)) e++; emit(i, e); i = e; continue; }
            if (N(st)) { e = st; while (N(e)) e++; emit(i, e); i = e; continue; }
            if (O(st)) { e = st; while (O(e)) e++; emit(i, e); i = e; continue; }
            if (is_space(cp[i])) {
                e = i; while (e < n && is_space(cp[e])) e++;
                if (e == n) { emit(i, e); i = e; }
                else if (e - i > 1) { emit(i, e - 1); i = e - 1; }
                else { emit(i, e); i = e; }
                continue;
            }
            emit(i, i + 1); i++;   // không xảy ra
        }
    }
    void bpe(const string& piece, vector<int>& out) const {
        vector<int> sy;
        for (unsigned char b : piece) { auto it = v2i.find(b2u[b]); if (it != v2i.end()) sy.push_back(it->second); }
        while (sy.size() > 1) {
            int best = INT32_MAX, bi = -1, bm = -1;
            for (size_t i = 0; i + 1 < sy.size(); i++) {
                auto it = merges.find(((uint64_t)(uint32_t)sy[i] << 32) | (uint32_t)sy[i + 1]);
                if (it != merges.end() && it->second.first < best) { best = it->second.first; bi = (int)i; bm = it->second.second; }
            }
            if (bi < 0) break;
            sy[bi] = bm; sy.erase(sy.begin() + bi + 1);
        }
        out.insert(out.end(), sy.begin(), sy.end());
    }
    vector<int> encode(const string& text) const {
        vector<int> ids; vector<std::pair<size_t, size_t>> pieces; pretok(text, pieces);
        for (auto& p : pieces) bpe(text.substr(p.first, p.second - p.first), ids);
        return ids;
    }
    string raw_bytes(const vector<int>& ids) const {   // id -> byte thô (bỏ qua <eos>)
        string out;
        for (int id : ids) {
            if (id < 0 || id >= (int)vocab.size() || id == eos) continue;
            const string& t = vocab[id];
            for (size_t i = 0; i < t.size();) { uint32_t c; int l = get_utf8(t, i, c); auto it = u2b.find(c); if (it != u2b.end()) out += (char)it->second; i += l; }
        }
        return out;
    }
};

// Bytes -> UTF-8 hợp lệ (byte hỏng thành U+FFFD). incomplete=true nếu đuôi là ký tự dở dang (chưa đủ byte, bị giữ lại).
static string to_text(const string& b, bool& incomplete) {
    string out; incomplete = false;
    for (size_t i = 0; i < b.size();) {
        uint32_t cp; bool inc = false; int l = Tok::get_utf8(b, i, cp, &inc);
        if (inc) { incomplete = true; break; }
        Tok::put_utf8(out, cp); i += l;
    }
    return out;
}

// ============================ model ============================
static const char* quant_name(uint32_t q) { return q == 0 ? "float32" : (q == 1 ? "int8" : "q4"); }
struct Cfg { uint32_t quant, V, D, layers, heads, E, F, nshared, L, min_k, max_k; float top_p, min_strength; int hd; };
struct Expert { Mat g, u, d; };
struct Layer { float *n1w, *n1b, *n2w, *n2b; Mat qkv, o, gate; float* sw; float sb; vector<Expert> ex, sh; };

struct ExScratch { vector<float> tg, tu, ty, out, qs; vector<int8_t> q; };   // bộ nhớ riêng của MỖI expert để nhiều expert chạy song song
struct Session {   // KV cache + bộ nhớ tạm của MỘT cuộc sinh (model chỉ đọc nên nhiều Session chạy song song được)
    vector<float> K, V; int n = 0;
    vector<float> x, xn, qkv, att, y, moe, sc, xs; vector<int8_t> xq; vector<ExScratch> ex;
    explicit Session(const Cfg& c)
        : K((size_t)c.layers * c.L * c.D), V((size_t)c.layers * c.L * c.D), x(c.D), xn(c.D), qkv(3 * c.D), att(c.D),
          y(c.D), moe(c.D), sc(c.L) {
        const size_t pad = ((size_t)std::max(c.D, c.F) + 31) / 32 * 32;
        xq.assign(pad, 0); xs.assign(pad / 32 + 1, 0.f);
        const size_t nt = std::min<size_t>(std::max(c.max_k, c.min_k), c.E) + c.nshared;
        ex.resize(nt);
        for (auto& e : ex) { e.tg.resize(c.F); e.tu.resize(c.F); e.ty.resize(c.D); e.out.resize(c.D); e.q.assign(pad, 0); e.qs.assign(pad / 32 + 1, 0.f); }
    }
};

struct Model {
    Cfg c; Mat emb, head; float *nfw, *nfb; vector<Layer> ly; Tok tok; vector<float> cosT, sinT; vector<void*> mem; MappedFile file;
    Model() = default;
    Model(const Model&) = delete;
    Model& operator=(const Model&) = delete;
    ~Model() { for (void* p : mem) afree(p); }

    void load(const string& path) {
        file.open(path);
        Reader r{file.data, file.data + file.size, &mem};
        char mg[4]; r.bytes(mg, 4); if (memcmp(mg, "YUI1", 4)) die("không phải file .yui (sai magic)");
        uint32_t ver = r.get<uint32_t>(); if (ver != 1 && ver != 2) die("phiên bản .yui không hỗ trợ (cần yui.cpp mới hơn?)"); r.ver = (int)ver;
        c.quant = r.get<uint32_t>(); if (c.quant > 2) die("kiểu lượng tử không hỗ trợ"); c.V = r.get<uint32_t>(); c.D = r.get<uint32_t>(); c.layers = r.get<uint32_t>();
        c.heads = r.get<uint32_t>(); c.E = r.get<uint32_t>(); c.F = r.get<uint32_t>(); c.nshared = r.get<uint32_t>();
        c.L = r.get<uint32_t>(); c.min_k = r.get<uint32_t>(); c.max_k = r.get<uint32_t>();
        c.top_p = r.get<float>(); c.min_strength = r.get<float>();
        if (c.D % c.heads || c.E > 256 || c.D % (c.heads * 2)) die("cấu hình không hợp lệ");
        c.hd = c.D / c.heads;
        uint32_t nv = r.get<uint32_t>(); tok.vocab.resize(nv);
        for (uint32_t i = 0; i < nv; i++) { tok.vocab[i] = r.str(); tok.v2i[tok.vocab[i]] = (int)i; }
        uint32_t nm = r.get<uint32_t>();
        for (uint32_t i = 0; i < nm; i++) {
            string a = r.str(), b = r.str();
            auto ia = tok.v2i.find(a), ib = tok.v2i.find(b), im = tok.v2i.find(a + b);
            if (ia == tok.v2i.end() || ib == tok.v2i.end() || im == tok.v2i.end()) continue;
            tok.merges[((uint64_t)(uint32_t)ia->second << 32) | (uint32_t)ib->second] = {(int)i, im->second};
        }
        tok.eos = r.get<int32_t>();
        uint32_t nr = r.get<uint32_t>(); tok.ranges.resize(nr);
        for (uint32_t i = 0; i < nr; i++) { tok.ranges[i][0] = r.get<uint32_t>(); tok.ranges[i][1] = r.get<uint32_t>(); tok.ranges[i][2] = r.get<uint32_t>(); }
        tok.init_bytes();
        const bool q = c.quant == 1; const int D = c.D, F = c.F;
        emb = load_mat(r, c.V, D, q);
        ly.resize(c.layers);
        for (auto& l : ly) {
            l.n1w = load_vec(r, D); l.n1b = load_vec(r, D);
            l.qkv = load_mat(r, 3 * D, D, q); l.o = load_mat(r, D, D, q);
            l.n2w = load_vec(r, D); l.n2b = load_vec(r, D);
            l.gate = load_mat(r, c.E, D, false, false);
            l.sw = load_vec(r, D); l.sb = r.get<float>();
            l.ex.resize(c.E);
            for (auto& e : l.ex) { e.g = load_mat(r, F, D, q); e.u = load_mat(r, F, D, q); e.d = load_mat(r, D, F, q); }
            l.sh.resize(c.nshared);
            for (auto& e : l.sh) { e.g = load_mat(r, F, D, q); e.u = load_mat(r, F, D, q); e.d = load_mat(r, D, F, q); }
        }
        nfw = load_vec(r, D); nfb = load_vec(r, D);
        head = load_mat(r, c.V, D, q);
        const int half = c.hd / 2; cosT.resize((size_t)c.L * half); sinT.resize((size_t)c.L * half);
        for (int p = 0; p < (int)c.L; p++)
            for (int i = 0; i < half; i++) {
                float inv = 1.0f / powf(10000.0f, (float)i / (float)half), ang = (float)p * inv;
                cosT[(size_t)p * half + i] = cosf(ang); sinT[(size_t)p * half + i] = sinf(ang);
            }
    }

    void embed(int t, float* out) const {
        if (t < 0 || t >= (int)c.V) t = 0;
        if (emb.type == 0) memcpy(out, emb.f + (size_t)t * c.D, c.D * sizeof(float));
        else if (emb.type == 2) {
            const int D = (int)c.D;
            for (int b = 0; b < emb.nb; b++) {
                const float d = emb.bs[(size_t)t * emb.nb + b]; const uint8_t* p = emb.q4 + ((size_t)t * emb.nb + b) * 16;
                for (int j = 0; j < 16; j++) { const int i0 = b * 32 + j, i1 = i0 + 16; if (i0 < D) out[i0] = (float)((int)(p[j] & 15) - 8) * d; if (i1 < D) out[i1] = (float)((int)(p[j] >> 4) - 8) * d; }
            }
        }
        else { const int8_t* w = emb.i8 + (size_t)t * c.D; float s = emb.sc[t]; for (int i = 0; i < (int)c.D; i++) out[i] = (float)w[i] * s; }
    }
    // Một expert SwiGLU: out = w * down(silu(gate(x)) * up(x)). xa = x đã lượng tử (dùng chung cho gate/up của mọi expert).
    void run_expert(const Expert& e, const float* x, const Act& xa, ExScratch& sc, float w) const {
        mv(e.g, x, xa, sc.tg.data(), false); mv(e.u, x, xa, sc.tu.data(), false);
        for (int i = 0; i < (int)c.F; i++) { float z = sc.tg[i]; sc.tg[i] = z / (1.f + expf(-z)) * sc.tu[i]; }
        Act a2{sc.q.data(), sc.qs.data()};
        if (use_qact(c.quant >= 1)) quant_act(sc.tg.data(), (int)c.F, a2.q, a2.s);
        mv(e.d, sc.tg.data(), a2, sc.ty.data(), false);
        for (int i = 0; i < (int)c.D; i++) sc.out[i] = w * sc.ty[i];
    }
    // Nạp 1 token tại vị trí s.n (phải < L), cập nhật KV cache. logits != nullptr thì tính logits [V].
    void step(Session& s, int tok_id, int kcap, float* logits) const {
        const int D = c.D, H = c.heads, hd = c.hd, half = hd / 2, L = c.L, E = c.E;
        const int pos = s.n; const bool qa = use_qact(c.quant >= 1);
        if (pos >= L) die("Session.n >= L (lỗi nội bộ)");
        embed(tok_id, s.x.data());
        const float* cs = &cosT[(size_t)pos * half]; const float* sn = &sinT[(size_t)pos * half];
        const float scale = 1.f / sqrtf((float)hd);
        const Act xa{s.xq.data(), s.xs.data()};
        for (int li = 0; li < (int)c.layers; li++) {
            const Layer& l = ly[li];
            layernorm(s.x.data(), l.n1w, l.n1b, s.xn.data(), D);
            if (qa) quant_act(s.xn.data(), D, xa.q, xa.s);
            mv(l.qkv, s.xn.data(), xa, s.qkv.data(), true);
            float *q = s.qkv.data(), *k = q + D, *v = q + 2 * D;
            for (int h = 0; h < H; h++) for (int p = 0; p < 2; p++) {
                float* x = (p ? k : q) + h * hd;
                for (int i = 0; i < half; i++) { float x1 = x[i], x2 = x[i + half]; x[i] = x1 * cs[i] - x2 * sn[i]; x[i + half] = x1 * sn[i] + x2 * cs[i]; }
            }
            memcpy(&s.K[((size_t)li * L + pos) * D], k, D * sizeof(float));
            memcpy(&s.V[((size_t)li * L + pos) * D], v, D * sizeof(float));
            for (int h = 0; h < H; h++) {
                const float* qh = q + h * hd; float mx = -INFINITY;
                for (int t = 0; t <= pos; t++) {
                    const float* kt = &s.K[((size_t)li * L + t) * D + h * hd]; float d = 0.f;
                    #pragma omp simd reduction(+ : d)
                    for (int i = 0; i < hd; i++) d += qh[i] * kt[i];
                    s.sc[t] = d * scale; mx = std::max(mx, s.sc[t]);
                }
                float sum = 0.f; for (int t = 0; t <= pos; t++) { s.sc[t] = expf(s.sc[t] - mx); sum += s.sc[t]; }
                float* yh = s.att.data() + h * hd; for (int i = 0; i < hd; i++) yh[i] = 0.f;
                for (int t = 0; t <= pos; t++) {
                    float w = s.sc[t] / sum; const float* vt = &s.V[((size_t)li * L + t) * D + h * hd];
                    #pragma omp simd
                    for (int i = 0; i < hd; i++) yh[i] += w * vt[i];
                }
            }
            if (qa) quant_act(s.att.data(), D, xa.q, xa.s);
            mv(l.o, s.att.data(), xa, s.y.data(), true);
            for (int i = 0; i < D; i++) s.x[i] += s.y[i];
            // ---- MoE ----
            layernorm(s.x.data(), l.n2w, l.n2b, s.xn.data(), D);
            float lg[256], pr[256]; int idx[256];
            for (int e = 0; e < E; e++) { const float* g = l.gate.f + (size_t)e * D; float a = 0.f; for (int i = 0; i < D; i++) a += g[i] * s.xn[i]; lg[e] = a; }
            float mx = lg[0]; for (int e = 1; e < E; e++) mx = std::max(mx, lg[e]);
            float sm = 0.f; for (int e = 0; e < E; e++) { pr[e] = expf(lg[e] - mx); sm += pr[e]; }
            for (int e = 0; e < E; e++) pr[e] /= sm;
            for (int e = 0; e < E; e++) idx[e] = e;
            std::stable_sort(idx, idx + E, [&](int a, int b) { return pr[a] > pr[b]; });
            int cap = (kcap > 0) ? std::min((int)c.max_k, kcap) : (int)c.max_k; cap = std::max(cap, (int)c.min_k);
            float run = 0.f, wsum = 0.f; float wsort[256];
            for (int j = 0; j < E; j++) {
                float p = pr[idx[j]]; run += p; float before = run - p;
                bool keep = ((before < c.top_p) || j < (int)c.min_k) && j < cap;
                wsort[j] = keep ? p : 0.f; wsum += wsort[j];
            }
            float st = 0.f; for (int i = 0; i < D; i++) st += l.sw[i] * s.xn[i]; st += l.sb;
            float strength = c.min_strength + (1.f - c.min_strength) * (1.f / (1.f + expf(-st)));
            if (qa) quant_act(s.xn.data(), D, xa.q, xa.s);   // xn (sau LayerNorm 2) lượng tử MỘT lần cho gate/up của mọi expert
            struct Task { const Expert* e; float w; } tasks[260]; int nt = 0;
            for (auto& sh : l.sh) tasks[nt++] = {&sh, 1.f};
            float den = std::max(wsum, 1e-9f);
            for (int j = 0; j < E; j++) if (wsort[j] > 0.f) tasks[nt++] = {&l.ex[idx[j]], (wsort[j] / den) * strength};
            if (nt > (int)s.ex.size()) die("quá nhiều expert được chọn (lỗi nội bộ)");
            auto work = [&](int t) { run_expert(*tasks[t].e, s.xn.data(), xa, s.ex[t], tasks[t].w); };
            if (nt >= 2 && (size_t)c.F * D >= (1u << 15)) par_for(nt, work); else for (int t = 0; t < nt; t++) work(t);   // các expert độc lập nên chạy song song
            for (int i = 0; i < D; i++) s.moe[i] = 0.f;
            for (int t = 0; t < nt; t++) for (int i = 0; i < D; i++) s.moe[i] += s.ex[t].out[i];   // cộng theo thứ tự cố định -> kết quả không phụ thuộc số luồng
            for (int i = 0; i < D; i++) s.x[i] += s.moe[i];
        }
        if (logits) {
            layernorm(s.x.data(), nfw, nfb, s.xn.data(), D);
            if (qa) quant_act(s.xn.data(), D, xa.q, xa.s);
            mv(head, s.xn.data(), xa, logits, true);
        }
        s.n++;
    }
};

// ============================ định dạng chat (khớp sft_format.py) ============================
static const char* USER_TAG = "User:";
static const char* BOT_TAG = "Yui:";
static bool ws_ascii(char ch) { return ch == ' ' || ch == '\t' || ch == '\n' || ch == '\r' || ch == '\f' || ch == '\v'; }

static string clean(const string& t) {   // gộp xuống dòng + khoảng trắng thừa thành 1 dấu cách, strip hai đầu
    string out; size_t i = 0, n = t.size();
    while (i < n) {
        if (ws_ascii(t[i])) {
            size_t j = i; bool nl = false; int sp = 0;
            while (j < n && ws_ascii(t[j])) { if (t[j] == '\n') nl = true; if (t[j] == ' ' || t[j] == '\t') sp++; j++; }
            if (nl || (j - i >= 2 && sp == (int)(j - i))) out += ' '; else out.append(t, i, j - i);
            i = j;
        } else out += t[i++];
    }
    size_t a = 0, b = out.size();
    while (a < b && ws_ascii(out[a])) a++;
    while (b > a && ws_ascii(out[b - 1])) b--;
    return out.substr(a, b - a);
}
struct Msg { string role, content; };
static string lower(string s) { for (auto& ch : s) ch = (char)tolower((unsigned char)ch); return s; }

static bool build_prompt(const Model& m, const vector<Msg>& all, vector<int>& ids, string& err) {
    vector<Msg> msgs; for (auto& x : all) if (lower(x.role) != "system") msgs.push_back(x);
    int last = -1; for (int i = 0; i < (int)msgs.size(); i++) { string r = lower(msgs[i].role); if (r == "user" || r == "human") last = i; }
    if (last < 0 || clean(msgs[last].content).empty()) { err = "cần ít nhất một tin nhắn role=user có nội dung"; return false; }
    vector<std::pair<string, string>> pairs; string cur; bool has = false;   // to_pairs
    for (int i = 0; i < last; i++) {
        string r = lower(msgs[i].role), t = clean(msgs[i].content); if (t.empty()) continue;
        if (r == "user" || r == "human") { cur = has ? cur + " " + t : t; has = true; }
        else if ((r == "assistant" || r == "bot" || r == "gpt" || r == "model") && has) { pairs.push_back({cur, t}); has = false; }
    }
    const int budget = std::max(16, (int)m.c.L - 32);
    struct Seg { vector<int> p, r; };
    vector<Seg> segs;
    for (auto& pr : pairs) {
        Seg s; s.p = m.tok.encode(string(USER_TAG) + " " + pr.first + "\n" + BOT_TAG);
        s.r = m.tok.encode(" " + pr.second); s.r.push_back(m.tok.eos); segs.push_back(s);
    }
    vector<int> fin = m.tok.encode(string(USER_TAG) + " " + clean(msgs[last].content) + "\n" + BOT_TAG);
    auto total = [&]() { size_t t = fin.size(); for (auto& s : segs) t += s.p.size() + s.r.size(); return t; };
    size_t drop = 0;
    while (drop < segs.size() && total() > (size_t)budget) { segs.erase(segs.begin()); }
    (void)drop;
    ids.clear();
    for (auto& s : segs) { ids.insert(ids.end(), s.p.begin(), s.p.end()); ids.insert(ids.end(), s.r.begin(), s.r.end()); }
    ids.insert(ids.end(), fin.begin(), fin.end());
    if ((int)ids.size() > budget) ids.erase(ids.begin(), ids.end() - budget);
    return true;
}

// ============================ sinh token ============================
struct GenOpts { int max_new = 120; float temp = 0.8f; int top_k = 40; float top_p = 0.95f; float rep = 1.1f; int k = 2; uint64_t seed = 0; };

static int sample(vector<float>& lg, const vector<int>& gen, const GenOpts& o, std::mt19937_64& rng) {
    const int V = (int)lg.size();
    if (o.rep != 1.f && !gen.empty()) {
        vector<int> rec(gen.end() - std::min<size_t>(48, gen.size()), gen.end());
        std::sort(rec.begin(), rec.end()); rec.erase(std::unique(rec.begin(), rec.end()), rec.end());
        for (int t : rec) lg[t] = lg[t] > 0 ? lg[t] / o.rep : lg[t] * o.rep;
    }
    if (o.temp <= 0.f) return (int)(std::max_element(lg.begin(), lg.end()) - lg.begin());
    for (auto& v : lg) v /= o.temp;
    // Chỉ giữ ứng viên top-k (cùng kết quả với việc che phần còn lại bằng -inf) nên top-p chỉ phải sắp xếp ~40 phần tử thay vì cả từ vựng.
    vector<int> cand;
    if (o.top_k > 0 && o.top_k < V) {
        vector<float> tmp(lg); std::nth_element(tmp.begin(), tmp.begin() + (o.top_k - 1), tmp.end(), std::greater<float>());
        const float kth = tmp[o.top_k - 1]; for (int i = 0; i < V; i++) if (!(lg[i] < kth)) cand.push_back(i);
    } else { cand.resize(V); for (int i = 0; i < V; i++) cand[i] = i; }
    const int C = (int)cand.size();
    float mx = -INFINITY; for (int j = 0; j < C; j++) mx = std::max(mx, lg[cand[j]]);
    float sum = 0.f; vector<float> p(C); for (int j = 0; j < C; j++) { p[j] = expf(lg[cand[j]] - mx); sum += p[j]; }
    for (auto& v : p) v /= sum;
    if (o.top_p < 1.f) {
        vector<int> id(C); for (int j = 0; j < C; j++) id[j] = j;
        std::sort(id.begin(), id.end(), [&](int a, int b) { return p[a] > p[b]; });
        float cs = 0.f, tot = 0.f; vector<char> drop(C, 0);
        for (int j = 0; j < C; j++) { float pi = p[id[j]]; if (cs > o.top_p) drop[id[j]] = 1; cs += pi; }
        for (int j = 0; j < C; j++) { if (drop[j]) p[j] = 0.f; tot += p[j]; }
        for (auto& v : p) v /= tot;
    }
    std::discrete_distribution<int> d(p.begin(), p.end());
    return cand[d(rng)];
}

// Chuyển id -> văn bản từng đoạn; ở chế độ chat bỏ khoảng trắng đầu, dừng khi model tự viết "User:".
struct TextStream {
    const Model& m; bool chat; vector<int> gen; string shown; bool done = false;
    TextStream(const Model& mm, bool c) : m(mm), chat(c) {}
    string view(bool& stop, bool& incomplete) {
        stop = false; string t = to_text(m.tok.raw_bytes(gen), incomplete);
        if (chat) {
            size_t a = 0; while (a < t.size() && ws_ascii(t[a])) a++; t = t.substr(a);
            size_t cut = t.find(USER_TAG);
            if (cut != string::npos) { t = t.substr(0, cut); while (!t.empty() && ws_ascii(t.back())) t.pop_back(); stop = true; }
        }
        return t;
    }
    string feed(int id) {
        gen.push_back(id); bool stop, inc; string t = view(stop, inc);
        if (!stop) {
            if (inc) return "";
            if (chat) { const string tag = USER_TAG; for (int n = (int)tag.size() - 1; n >= 1; n--) if (t.size() >= (size_t)n && t.compare(t.size() - n, n, tag, 0, n) == 0) { t.resize(t.size() - n); break; } }
        }
        string d; if (t.size() > shown.size() && t.compare(0, shown.size(), shown) == 0) { d = t.substr(shown.size()); shown = t; }
        if (stop) done = true;
        return d;
    }
    string finish() {
        bool stop, inc; string t = view(stop, inc);
        if (t.size() > shown.size() && t.compare(0, shown.size(), shown) == 0) { string d = t.substr(shown.size()); shown = t; return d; }
        return "";
    }
};

// Bộ sinh từng bước (KV cache + dịch ngữ cảnh khi cache đầy). next() trả đoạn văn bản mới cho tới khi hết; dùng chung cho CLI, server và API C.
struct Gen {
    const Model& m; GenOpts o; bool chat; Session s; vector<float> lg; std::mt19937_64 rng; vector<int> hist, gen;
    TextStream ts; int n = 0; bool finished = false, flushed = false; int L, keep;
    Gen(const Model& mm, vector<int> ids, const GenOpts& op, bool c)
        : m(mm), o(op), chat(c), s(mm.c), lg(mm.c.V), rng(op.seed ? op.seed : std::random_device{}()), hist(std::move(ids)), ts(mm, c),
          L((int)mm.c.L), keep(std::max(16, (int)mm.c.L / 2)) {
        if (hist.empty()) hist.push_back(m.tok.eos >= 0 ? m.tok.eos : 0);
        prefill(hist.size() > (size_t)(L - 8) ? hist.size() - (L - 8) : 0);
    }
    void prefill(size_t from) { s.n = 0; for (size_t i = from; i < hist.size(); i++) m.step(s, hist[i], o.k, i + 1 == hist.size() ? lg.data() : nullptr); }
    bool next(string& delta) {
        delta.clear();
        while (true) {
            if (finished) {
                if (flushed) return false;
                flushed = true; delta = ts.finish(); return !delta.empty();
            }
            if (n >= o.max_new) { finished = true; continue; }
            vector<float> work(lg);
            int nxt = sample(work, gen, o, rng);
            if (nxt == m.tok.eos) { finished = true; continue; }
            gen.push_back(nxt); hist.push_back(nxt); n++;
            if (getenv("YUI_DEBUG_IDS")) fprintf(stderr, "%d ", nxt);
            string d = ts.feed(nxt);
            if (ts.done) { finished = true; flushed = true; delta = d; return !delta.empty(); }
            if (s.n >= L) prefill(hist.size() > (size_t)keep ? hist.size() - keep : 0);
            else m.step(s, nxt, o.k, lg.data());
            if (!d.empty()) { delta = d; return true; }
        }
    }
};
// on_text trả false = dừng (khách ngắt kết nối). Trả số token đã sinh.
static int generate(const Model& m, vector<int> hist, const GenOpts& o, bool chat, const std::function<bool(const string&)>& on_text) {
    Gen g(m, std::move(hist), o, chat); string d;
    while (g.next(d)) if (!on_text(d)) break;
    return g.n;
}

// ============================ JSON tối giản ============================
struct J {
    enum T { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL; bool b = false; double n = 0; string s; vector<J> a; vector<std::pair<string, J>> o;
    const J* get(const string& k) const { for (auto& kv : o) if (kv.first == k) return &kv.second; return nullptr; }
};
struct JP {
    const string& s; size_t i = 0; int depth = 0;
    explicit JP(const string& x) : s(x) {}
    void ws() { while (i < s.size() && (s[i] == ' ' || s[i] == '\t' || s[i] == '\n' || s[i] == '\r')) i++; }
    bool lit(const char* w) { size_t l = strlen(w); if (s.compare(i, l, w) == 0) { i += l; return true; } return false; }
    bool str(string& out) {
        if (i >= s.size() || s[i] != '"') return false;
        i++;
        while (i < s.size()) {
            char c = s[i++];
            if (c == '"') return true;
            if (c != '\\') { out += c; continue; }
            if (i >= s.size()) return false;
            char e = s[i++];
            switch (e) {
                case 'n': out += '\n'; break; case 't': out += '\t'; break; case 'r': out += '\r'; break; case 'b': out += '\b'; break;
                case 'f': out += '\f'; break; case '/': out += '/'; break; case '\\': out += '\\'; break; case '"': out += '"'; break;
                case 'u': {
                    auto hex = [&](uint32_t& v) { if (i + 4 > s.size()) return false; v = 0; for (int k = 0; k < 4; k++) { char h = s[i + k]; v <<= 4;
                        if (h >= '0' && h <= '9') v |= h - '0'; else if (h >= 'a' && h <= 'f') v |= h - 'a' + 10; else if (h >= 'A' && h <= 'F') v |= h - 'A' + 10; else return false; } i += 4; return true; };
                    uint32_t v; if (!hex(v)) return false;
                    if (v >= 0xD800 && v < 0xDC00 && i + 1 < s.size() && s[i] == '\\' && s[i + 1] == 'u') { i += 2; uint32_t w; if (!hex(w)) return false; v = 0x10000 + ((v - 0xD800) << 10) + (w - 0xDC00); }
                    Tok::put_utf8(out, v); break; }
                default: return false;
            }
        }
        return false;
    }
    bool val(J& v) {
        ws(); if (i >= s.size() || ++depth > 64) return false;
        bool ok = false; char c = s[i];
        if (c == '{') {
            v.t = J::OBJ; i++; ws();
            if (i < s.size() && s[i] == '}') { i++; ok = true; }
            else while (true) {
                ws(); string k; if (!str(k)) break; ws(); if (i >= s.size() || s[i++] != ':') break;
                J x; if (!val(x)) break; v.o.push_back({k, std::move(x)}); ws();
                if (i < s.size() && s[i] == ',') { i++; continue; }
                if (i < s.size() && s[i] == '}') { i++; ok = true; } break;
            }
        } else if (c == '[') {
            v.t = J::ARR; i++; ws();
            if (i < s.size() && s[i] == ']') { i++; ok = true; }
            else while (true) {
                J x; if (!val(x)) break; v.a.push_back(std::move(x)); ws();
                if (i < s.size() && s[i] == ',') { i++; continue; }
                if (i < s.size() && s[i] == ']') { i++; ok = true; } break;
            }
        } else if (c == '"') { v.t = J::STR; ok = str(v.s); }
        else if (lit("true")) { v.t = J::BOOL; v.b = true; ok = true; }
        else if (lit("false")) { v.t = J::BOOL; v.b = false; ok = true; }
        else if (lit("null")) { v.t = J::NUL; ok = true; }
        else { char* e; v.n = strtod(s.c_str() + i, &e); if (e != s.c_str() + i) { v.t = J::NUM; i = e - s.c_str(); ok = true; } }
        depth--; return ok;
    }
};
static string jesc(const string& s) {
    string o = "\"";
    for (unsigned char c : s) {
        switch (c) { case '"': o += "\\\""; break; case '\\': o += "\\\\"; break; case '\n': o += "\\n"; break; case '\r': o += "\\r"; break; case '\t': o += "\\t"; break;
            default: if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; } else o += (char)c; }
    }
    return o + "\"";
}
static string iso_now() { time_t t = time(nullptr); char b[32]; strftime(b, sizeof b, "%Y-%m-%dT%H:%M:%SZ", gmtime(&t)); return b; }

static void parse_messages(const J& j, vector<Msg>& msgs) {   // j: mảng [{role,content}] hoặc object có "messages"
    const J* ms = j.t == J::ARR ? &j : j.get("messages");
    if (ms && ms->t == J::ARR) for (auto& x : ms->a) {
        if (x.t != J::OBJ) continue;
        Msg mm; auto* r = x.get("role"); auto* c = x.get("content");
        if (r && r->t == J::STR) mm.role = r->s;
        if (c && c->t == J::STR) mm.content = c->s;
        msgs.push_back(mm);
    }
}
static GenOpts parse_opts(const J& body, int def_k) {
    GenOpts o; o.k = def_k; const J* op = body.get("options");
    auto num = [&](std::initializer_list<const char*> names, double def) {
        for (auto n : names) { if (op && op->t == J::OBJ) { auto* v = op->get(n); if (v && v->t == J::NUM) return v->n; } auto* v = body.get(n); if (v && v->t == J::NUM) return v->n; }
        return def; };
    int mx = (int)num({"num_predict", "max_tokens"}, 120); o.max_new = mx <= 0 ? 256 : std::min(mx, 4096);
    o.temp = (float)num({"temperature"}, 0.8); o.top_k = (int)num({"top_k"}, 40); o.top_p = (float)num({"top_p"}, 0.95);
    o.rep = (float)num({"repeat_penalty"}, 1.1); int k = (int)num({"k"}, def_k); o.k = (k == 1 || k == 2) ? k : def_k;
    o.seed = (uint64_t)num({"seed"}, 0); return o;
}

// ============================ API C (cho app Flutter/FFI, kiểu poll, không callback) ============================
#ifdef _WIN32
#define YUI_API extern "C" __declspec(dllexport)
#else
#define YUI_API extern "C" __attribute__((visibility("default")))
#endif
struct YuiGenH { Gen g; string pending; size_t off = 0; YuiGenH(const Model& m, vector<int> ids, const GenOpts& o, bool chat) : g(m, std::move(ids), o, chat) {} };
static void set_err(char* err, int len, const string& m) { if (err && len > 0) snprintf(err, (size_t)len, "%s", m.c_str()); }
static GenOpts opts_from_json(const char* js, int def_k) {
    if (js && *js) { string t = js; J j; JP jp(t); if (jp.val(j) && j.t == J::OBJ) return parse_opts(j, def_k); }
    GenOpts o; o.k = def_k; return o;
}
YUI_API const char* yui_version() { return "yui-engine 1.2"; }
YUI_API void yui_set_threads(int n) { pool().resize(n > 0 ? n : default_threads()); }   // n <= 0: tự chọn
YUI_API int yui_get_threads() { return pool().nthreads; }
YUI_API int yui_cpu_features(char* out, int len) {   // vd "neon+dotprod, 4 luồng"; trả số byte, <0 nếu lỗi
    if (!out || len <= 0) return -1;
    string s = string(isa_name()) + (g_exact ? ", exact-float" : "") + ", " + std::to_string(pool().nthreads) + " thread";
    if ((int)s.size() + 1 > len) return -2;
    memcpy(out, s.c_str(), s.size() + 1); return (int)s.size();
}
YUI_API void* yui_load(const char* path, char* err, int errlen) {
    Model* m = new Model();
    try { m->load(path ? path : ""); } catch (const std::exception& e) { set_err(err, errlen, e.what()); delete m; return nullptr; }
    return m;
}
YUI_API void yui_free(void* h) { delete (Model*)h; }
YUI_API int yui_info(void* h, char* out, int len) {   // JSON thông tin model; trả số byte, <0 nếu lỗi
    Model* m = (Model*)h; if (!m || !out) return -1;
    string s = "{\"d\":" + std::to_string(m->c.D) + ",\"layers\":" + std::to_string(m->c.layers) + ",\"experts\":" + std::to_string(m->c.E) +
               ",\"heads\":" + std::to_string(m->c.heads) + ",\"ctx\":" + std::to_string(m->c.L) + ",\"vocab\":" + std::to_string(m->c.V) +
               ",\"quant\":\"" + quant_name(m->c.quant) + "\"}";
    if ((int)s.size() + 1 > len) return -2;
    memcpy(out, s.c_str(), s.size() + 1); return (int)s.size();
}
YUI_API void* yui_chat_begin(void* h, const char* messages_json, const char* opts_json, int def_k, char* err, int errlen) {
    try {
        Model* m = (Model*)h; if (!m) throw YuiError("model chưa nạp");
        string t = messages_json ? messages_json : ""; J j; JP jp(t); if (!jp.val(j)) throw YuiError("messages_json không phải JSON hợp lệ");
        vector<Msg> msgs; parse_messages(j, msgs); vector<int> ids; string e;
        if (!build_prompt(*m, msgs, ids, e)) throw YuiError(e);
        return new YuiGenH(*m, ids, opts_from_json(opts_json, def_k == 1 ? 1 : 2), true);
    } catch (const std::exception& e) { set_err(err, errlen, e.what()); return nullptr; }
}
YUI_API void* yui_complete_begin(void* h, const char* prompt, const char* opts_json, int def_k, char* err, int errlen) {
    try {
        Model* m = (Model*)h; if (!m) throw YuiError("model chưa nạp");
        return new YuiGenH(*m, m->tok.encode(prompt ? prompt : ""), opts_from_json(opts_json, def_k == 1 ? 1 : 2), false);
    } catch (const std::exception& e) { set_err(err, errlen, e.what()); return nullptr; }
}
// >0: số byte UTF-8 đã ghi vào buf; 0: đã xong; <0: lỗi (-2: buf quá nhỏ)
YUI_API int yui_gen_next(void* g, char* buf, int buflen) {
    YuiGenH* h = (YuiGenH*)g; if (!h || !buf || buflen < 8) return -1;
    try {
        if (h->off >= h->pending.size()) { h->pending.clear(); h->off = 0; string d; if (!h->g.next(d)) return 0; h->pending = d; }
        size_t left = h->pending.size() - h->off, n = std::min((size_t)buflen, left);
        while (n > 0 && n < left && (((unsigned char)h->pending[h->off + n]) & 0xC0) == 0x80) n--;   // không cắt giữa ký tự UTF-8
        if (n == 0) return -2;
        memcpy(buf, h->pending.data() + h->off, n); h->off += n; return (int)n;
    } catch (...) { return -1; }
}
YUI_API int yui_gen_tokens(void* g) { return g ? ((YuiGenH*)g)->g.n : 0; }
YUI_API void yui_gen_end(void* g) { delete (YuiGenH*)g; }

#ifndef YUI_LIB
// ============================ server HTTP (Ollama + OpenAI) ============================
static const char* PAGE = R"HTML(<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Yui</title><style>body{font:16px system-ui;margin:0;display:flex;flex-direction:column;height:100dvh}
#log{flex:1;overflow:auto;padding:12px}.u,.a{margin:6px 0;padding:8px 12px;border-radius:12px;max-width:85%;white-space:pre-wrap}
.u{background:#dbeafe;margin-left:auto}.a{background:#f3f4f6}form{display:flex;gap:6px;padding:8px;border-top:1px solid #ddd}
input{flex:1;font:inherit;padding:8px}button{font:inherit;padding:8px 14px}</style>
<div id=log></div><form id=f><input id=m placeholder="Nhắn cho Yui..." autocomplete=off><button>Gửi</button></form>
<script>
const log=document.getElementById('log'),msgs=[];
function add(c,t){const d=document.createElement('div');d.className=c;d.textContent=t;log.appendChild(d);log.scrollTop=1e9;return d}
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const i=document.getElementById('m'),t=i.value.trim();if(!t)return;i.value='';
msgs.push({role:'user',content:t});add('u',t);const el=add('a','…');let out='';
const key=localStorage.getItem('key')||'';const h={'Content-Type':'application/json'};if(key)h.Authorization='Bearer '+key;
const r=await fetch('/api/chat',{method:'POST',headers:h,body:JSON.stringify({messages:msgs.slice(-8),stream:true})});
if(r.status==401){const k=prompt('API key:');if(k)localStorage.setItem('key',k);el.textContent='(nhập key rồi gửi lại)';msgs.pop();return}
if(!r.ok){el.textContent='Lỗi: '+(await r.text());msgs.pop();return}
const rd=r.body.getReader(),dec=new TextDecoder();let buf='';
for(;;){const{done,value}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});const ls=buf.split('\n');buf=ls.pop();
for(const l of ls){if(!l.trim())continue;const j=JSON.parse(l);if(j.message){out+=j.message.content;el.textContent=out||'…'}}}
msgs.push({role:'assistant',content:out})};
</script>)HTML";

struct Server {
    const Model& m; string api_key; int def_k; int parallel; std::mutex mu; std::condition_variable cv; int active = 0;
    const char* NAME = "yui-adaptive-moe";

    static bool send_all(int fd, const string& s) {
        size_t o = 0; while (o < s.size()) { ssize_t k = send(fd, s.data() + o, s.size() - o, MSG_NOSIGNAL); if (k <= 0) return false; o += k; } return true;
    }
    static string status_text(int c) { return c == 200 ? "OK" : c == 204 ? "No Content" : c == 400 ? "Bad Request" : c == 401 ? "Unauthorized" : c == 404 ? "Not Found" : "Error"; }
    static string cors() { return "Access-Control-Allow-Origin: *\r\nAccess-Control-Allow-Headers: Content-Type, Authorization\r\nAccess-Control-Allow-Methods: GET, POST, OPTIONS\r\n"; }
    static void reply(int fd, int code, const string& ctype, const string& body) {
        string h = "HTTP/1.0 " + std::to_string(code) + " " + status_text(code) + "\r\nContent-Type: " + ctype + "\r\nContent-Length: " + std::to_string(body.size()) + "\r\n" + cors() + "Connection: close\r\n\r\n";
        send_all(fd, h + body);
    }
    static void jerr(int fd, int code, const string& msg) { reply(fd, code, "application/json; charset=utf-8", "{\"error\":" + jesc(msg) + "}"); }
    static bool start_stream(int fd, const char* ctype) {
        return send_all(fd, string("HTTP/1.0 200 OK\r\nContent-Type: ") + ctype + "\r\nCache-Control: no-cache\r\n" + cors() + "Connection: close\r\n\r\n");
    }

    GenOpts opts(const J& body) { return parse_opts(body, def_k); }
    void acquire() { std::unique_lock<std::mutex> lk(mu); cv.wait(lk, [&] { return active < parallel; }); active++; }
    void release() { { std::lock_guard<std::mutex> lk(mu); active--; } cv.notify_one(); }

    void handle(int fd) {
        string req; char buf[4096]; size_t hend = string::npos;
        while (hend == string::npos) {
            ssize_t k = recv(fd, buf, sizeof buf, 0); if (k <= 0) { close(fd); return; }
            req.append(buf, k); hend = req.find("\r\n\r\n"); if (req.size() > (1 << 20) && hend == string::npos) { close(fd); return; }
        }
        string head = req.substr(0, hend), body = req.substr(hend + 4);
        size_t eol = head.find("\r\n"); string line = head.substr(0, eol);
        size_t sp1 = line.find(' '), sp2 = line.find(' ', sp1 + 1);
        if (sp1 == string::npos || sp2 == string::npos) { close(fd); return; }
        string method = line.substr(0, sp1), path = line.substr(sp1 + 1, sp2 - sp1 - 1); size_t qm = path.find('?'); if (qm != string::npos) path.resize(qm);
        size_t clen = 0; string auth;
        for (size_t p = eol; p != string::npos && p < head.size();) {
            size_t e = head.find("\r\n", p + 2); string h = head.substr(p + 2, (e == string::npos ? head.size() : e) - p - 2); p = e;
            size_t c = h.find(':'); if (c == string::npos) continue; string k = lower(h.substr(0, c)), v = h.substr(c + 1); while (!v.empty() && v[0] == ' ') v.erase(0, 1);
            if (k == "content-length") clen = (size_t)atol(v.c_str()); else if (k == "authorization") auth = v;
        }
        if (clen > (1u << 20)) { jerr(fd, 400, "body quá lớn"); close(fd); return; }
        while (body.size() < clen) { ssize_t k = recv(fd, buf, sizeof buf, 0); if (k <= 0) break; body.append(buf, k); }
        route(fd, method, path, body, auth); close(fd);
    }

    void route(int fd, const string& method, const string& path, const string& body, const string& auth) {
        if (method == "OPTIONS") { send_all(fd, "HTTP/1.0 204 No Content\r\n" + cors() + "Connection: close\r\n\r\n"); return; }
        if (method == "GET" && path == "/") { reply(fd, 200, "text/html; charset=utf-8", PAGE); return; }
        if (method == "GET" && path == "/health") { reply(fd, 200, "application/json", string("{\"status\":\"ok\",\"model\":\"") + NAME + "\"}"); return; }
        if (!api_key.empty() && auth != "Bearer " + api_key) { jerr(fd, 401, "unauthorized"); return; }
        if (method == "GET" && (path == "/api/tags" || path == "/v1/models")) {
            reply(fd, 200, "application/json", string("{\"models\":[{\"name\":\"") + NAME + "\",\"model\":\"" + NAME + "\"}],\"object\":\"list\",\"data\":[{\"id\":\"" + NAME + "\",\"object\":\"model\"}]}"); return; }
        if (method != "POST") { jerr(fd, 404, "not found"); return; }
        J j; JP jp(body); if (body.empty() || !jp.val(j) || j.t != J::OBJ) { jerr(fd, 400, "yêu cầu không hợp lệ: body phải là JSON object"); return; }
        GenOpts o = opts(j); const J* st = j.get("stream"); 
        if (path == "/api/chat" || path == "/v1/chat/completions") {
            bool openai = path[1] == 'v'; bool stream = st && st->t == J::BOOL ? st->b : !openai;
            vector<Msg> msgs; const J* ms = j.get("messages"); if (ms && ms->t == J::ARR) parse_messages(*ms, msgs);
            vector<int> ids; string err; if (!build_prompt(m, msgs, ids, err)) { jerr(fd, 400, err); return; }
            run(fd, ids, o, true, stream, openai); return;
        }
        if (path == "/api/generate") {
            const J* p = j.get("prompt"); bool stream = st && st->t == J::BOOL ? st->b : true;
            run(fd, m.tok.encode(p && p->t == J::STR ? p->s : ""), o, false, stream, false); return;
        }
        jerr(fd, 404, "not found");
    }

    void run(int fd, const vector<int>& ids, const GenOpts& o, bool chat, bool stream, bool openai) {
        acquire(); auto t0 = std::chrono::steady_clock::now(); int n = 0; string full; bool alive = true;
        string cid = "chatcmpl-" + std::to_string(time(nullptr)); long created = (long)time(nullptr);
        auto chunk = [&](const string& delta_json, const char* fin) {
            return "data: {\"id\":" + jesc(cid) + ",\"object\":\"chat.completion.chunk\",\"created\":" + std::to_string(created) + ",\"model\":\"" + NAME + "\",\"choices\":[{\"index\":0,\"delta\":" + delta_json + ",\"finish_reason\":" + fin + "}]}\n\n"; };
        auto piece = [&](const string& d, bool done) {
            string r = string("{\"model\":\"") + NAME + "\",\"created_at\":\"" + iso_now() + "\",";
            r += chat ? "\"message\":{\"role\":\"assistant\",\"content\":" + jesc(d) + "}" : "\"response\":" + jesc(d);
            return r + ",\"done\":" + (done ? "true" : "false"); };
        if (stream) alive = start_stream(fd, openai ? "text/event-stream" : "application/x-ndjson");
        bool first = true;
        if (alive) n = generate(m, ids, o, chat, [&](const string& d) {
            if (!stream) { full += d; return true; }
            bool ok = openai ? send_all(fd, chunk(string("{") + (first ? "\"role\":\"assistant\"," : "") + "\"content\":" + jesc(d) + "}", "null")) : send_all(fd, piece(d, false) + "}\n");
            first = false; return ok; });
        double ns = std::chrono::duration<double, std::nano>(std::chrono::steady_clock::now() - t0).count();
        if (stream && alive) {
            if (openai) send_all(fd, chunk("{}", "\"stop\"") + "data: [DONE]\n\n");
            else send_all(fd, piece("", true) + ",\"done_reason\":\"stop\",\"eval_count\":" + std::to_string(n) + ",\"total_duration\":" + std::to_string((long long)ns) + "}\n");
        } else if (!stream) {
            if (openai) reply(fd, 200, "application/json; charset=utf-8", "{\"id\":" + jesc(cid) + ",\"object\":\"chat.completion\",\"created\":" + std::to_string(created) + ",\"model\":\"" + NAME + "\",\"choices\":[{\"index\":0,\"message\":{\"role\":\"assistant\",\"content\":" + jesc(full) + "},\"finish_reason\":\"stop\"}]}");
            else reply(fd, 200, "application/json; charset=utf-8", piece(full, true) + ",\"done_reason\":\"stop\",\"eval_count\":" + std::to_string(n) + ",\"total_duration\":" + std::to_string((long long)ns) + "}");
        }
        release();
    }

    void serve(const string& host, int port) {
        signal(SIGPIPE, SIG_IGN);
        int s = socket(AF_INET, SOCK_STREAM, 0); int one = 1; setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &one, sizeof one);
        sockaddr_in a{}; a.sin_family = AF_INET; a.sin_port = htons(port);
        if (inet_pton(AF_INET, host.c_str(), &a.sin_addr) != 1) die("host không hợp lệ: " + host);
        if (bind(s, (sockaddr*)&a, sizeof a) < 0) die("không bind được cổng " + std::to_string(port));
        listen(s, 64);
        fprintf(stderr, "[yui] http://%s:%d  (trang chat ở '/'; /api/chat, /api/generate, /v1/chat/completions) %s\n", host.c_str(), port,
                api_key.empty() ? "[KHÔNG có API key - đừng mở ra Internet]" : "[có API key]");
        while (true) { int fd = accept(s, nullptr, nullptr); if (fd < 0) continue; std::thread([this, fd] { try { handle(fd); } catch (...) { close(fd); } }).detach(); }
    }
};

// ============================ main ============================
static int cli_main(int argc, char** argv) {
    string model_path, prompt, enc, ids_s, logits_path, host = "127.0.0.1", api_key; int port = 11434, parallel = 2;
    int threads = 0; bool serve = false, bench = false, bench_threads = false, have_enc = false; GenOpts o; int max_turns = 4;
    if (const char* e = getenv("ENGINE_API_KEY")) api_key = e;
    for (int i = 1; i < argc; i++) {
        string a = argv[i]; auto nx = [&]() -> string { if (i + 1 >= argc) die("thiếu giá trị cho " + a); return argv[++i]; };
        if (a == "--model") model_path = nx(); else if (a == "--prompt") prompt = nx(); else if (a == "--encode") { enc = nx(); have_enc = true; }
        else if (a == "--ids") ids_s = nx(); else if (a == "--logits") logits_path = nx(); else if (a == "--serve") serve = true;
        else if (a == "--bench") bench = true; else if (a == "--host") host = nx(); else if (a == "--port") port = atoi(nx().c_str());
        else if (a == "--api_key") api_key = nx(); else if (a == "--parallel") parallel = std::max(1, atoi(nx().c_str()));
        else if (a == "--k") o.k = atoi(nx().c_str()); else if (a == "--max_new") o.max_new = atoi(nx().c_str());
        else if (a == "--temp") o.temp = (float)atof(nx().c_str()); else if (a == "--top_k") o.top_k = atoi(nx().c_str());
        else if (a == "--top_p") o.top_p = (float)atof(nx().c_str()); else if (a == "--rep") o.rep = (float)atof(nx().c_str());
        else if (a == "--seed") o.seed = (uint64_t)atoll(nx().c_str()); else if (a == "--max_turns") max_turns = atoi(nx().c_str());
        else if (a == "--threads") threads = atoi(nx().c_str()); else if (a == "--exact") g_exact = true; else if (a == "--bench_threads") bench_threads = true;
        else die("tham số lạ: " + a);
    }
    if (model_path.empty()) die("cần --model file.yui");
    if (threads > 0) pool().resize(threads);
    Model m; m.load(model_path);
    fprintf(stderr, "[yui] %s | d=%u layers=%u experts=%u heads=%u | ngữ cảnh %u | %s | k=%d | CPU: %s, %d luồng%s%s\n", model_path.c_str(), m.c.D, m.c.layers, m.c.E, m.c.heads, m.c.L, quant_name(m.c.quant), o.k, isa_name(), pool().nthreads, g_exact ? ", exact" : "", m.file.mapped ? ", mmap" : "");

    if (have_enc) { for (int id : m.tok.encode(enc)) printf("%d ", id); printf("\n"); return 0; }
    if (!ids_s.empty()) {   // xuất logits từng vị trí để so với PyTorch
        vector<int> ids; size_t p = 0; while (p < ids_s.size()) { ids.push_back(atoi(ids_s.c_str() + p)); size_t q = ids_s.find(',', p); if (q == string::npos) break; p = q + 1; }
        Session s(m.c); vector<float> lg((size_t)m.c.V * ids.size());
        if (ids.size() > m.c.L) die("quá dài so với ngữ cảnh");
        for (size_t i = 0; i < ids.size(); i++) m.step(s, ids[i], o.k, &lg[i * m.c.V]);
        if (!logits_path.empty()) { FILE* f = fopen(logits_path.c_str(), "wb"); if (!f) die("không ghi được " + logits_path); fwrite(lg.data(), sizeof(float), lg.size(), f); fclose(f); }
        printf("ok %zu vị trí\n", ids.size()); return 0;
    }
    Server sv{m, api_key, o.k, parallel, {}, {}, 0};
    if (serve) { sv.serve(host, port); return 0; }
    if (bench_threads) {   // thử 1..N luồng, in tốc độ để chọn số luồng tốt nhất cho máy này
        vector<int> ids = m.tok.encode("Hà Nội là thủ đô của Việt Nam"); GenOpts b = o; b.max_new = 120; b.temp = 0; b.rep = 1.0f;
        int hw = std::max(1, (int)std::thread::hardware_concurrency()); double best = 0; int bt = 1;
        for (int t = 1; t <= std::min(hw, 16); t++) {
            pool().resize(t); generate(m, ids, b, false, [](const string&) { return true; });   // làm nóng
            auto t0 = std::chrono::steady_clock::now(); int n = generate(m, ids, b, false, [](const string&) { return true; });
            double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count(), tps = n / sec;
            printf("%2d luồng: %.1f token/giây\n", t, tps); if (tps > best) { best = tps; bt = t; }
        }
        printf("=> nên dùng --threads %d (hoặc YUI_THREADS=%d)\n", bt, bt); return 0;
    }
    if (bench) {
        vector<int> ids = m.tok.encode("Hà Nội là thủ đô của Việt Nam"); GenOpts b = o; b.max_new = 200; b.temp = 0; b.rep = 1.0f;
        auto t0 = std::chrono::steady_clock::now(); int n = generate(m, ids, b, false, [](const string&) { return true; });
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("%d token trong %.2fs = %.1f token/giây (prompt %zu token, k=%d)\n", n, sec, n / sec, ids.size(), o.k); return 0;
    }
    if (!prompt.empty()) { generate(m, m.tok.encode(prompt), o, false, [](const string& d) { fputs(d.c_str(), stdout); fflush(stdout); return true; }); printf("\n"); return 0; }
    printf("Chat với Yui. Dòng trống để thoát.\n");
    vector<Msg> hist;
    while (true) {
        printf("\nBạn: "); fflush(stdout); char line[4096]; if (!fgets(line, sizeof line, stdin)) break;
        string t = clean(line); if (t.empty()) break;
        hist.push_back({"user", t}); size_t from = hist.size() > (size_t)(2 * max_turns + 1) ? hist.size() - (2 * max_turns + 1) : 0;
        vector<Msg> win(hist.begin() + from, hist.end()); vector<int> ids; string err;
        if (!build_prompt(m, win, ids, err)) { printf("(%s)\n", err.c_str()); hist.pop_back(); continue; }
        printf("Yui: "); fflush(stdout); string reply; auto t0 = std::chrono::steady_clock::now();
        int n = generate(m, ids, o, true, [&](const string& d) { reply += d; fputs(d.c_str(), stdout); fflush(stdout); return true; });
        double sec = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        printf("\n  [%d token, %.1f/s]\n", n, n / std::max(sec, 1e-9));
        hist.push_back({"assistant", reply});
    }
    return 0;
}

int main(int argc, char** argv) {
    try { return cli_main(argc, argv); }
    catch (const std::exception& e) { fprintf(stderr, "[yui] LỖI: %s\n", e.what()); return 1; }
}
#endif   // YUI_LIB
