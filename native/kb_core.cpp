// kb_core.cpp - lõi kho tri thức C++17: tách từ tiếng Việt + BM25 (yui_rag.py) + Graphify (yui_graph.py). Xem kb_core.h và docs/APP_KB.md.
// Bám Python TỪNG BƯỚC (kể cả thứ tự phép cộng số thực và thứ tự chèn của dict) để bộ test vàng so khớp; chỗ nào cố ý khác đã ghi trong APP_KB.md §9.
// Build: g++ -O2 -std=c++17 -ffp-contract=off -fPIC -shared kb_core.cpp -o libkb_core.so   (-ffp-contract=off: không cho trình biên dịch gộp nhân-cộng, giữ kết quả số thực giống Python)
#include "kb_core.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <mutex>
#include <regex>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "kb_unicode.inc"
#include "kb_words.inc"

namespace {
using U = std::u32string;
using S = std::string;

// ============================ Unicode ============================
bool in_r(const uint32_t (*r)[2], size_t n, uint32_t cp) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t m = (lo + hi) / 2;
        if (cp < r[m][0]) hi = m;
        else if (cp > r[m][1]) lo = m + 1;
        else return true;
    }
    return false;
}
#define KB_IN(name, cp) in_r(name, name##_n, (cp))

bool is_alnum(uint32_t c) { return c < 0x80 ? ((c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z')) : KB_IN(kb_alnum, c); }
bool is_word_cp(uint32_t c) { return c == '_' || is_alnum(c); }                       // \w của re
bool is_nd(uint32_t c) { return c < 0x80 ? (c >= '0' && c <= '9') : KB_IN(kb_nd, c); } // \d của re
bool is_letter_cp(uint32_t c) { return is_alnum(c) && !is_nd(c); }                      // [^\W\d_]
bool is_space(uint32_t c) { return c < 0x80 ? (c == ' ' || (c >= 9 && c <= 13) || (c >= 0x1C && c <= 0x1F)) : KB_IN(kb_space, c); }
bool is_digit_cp(uint32_t c) { return c < 0x80 ? (c >= '0' && c <= '9') : KB_IN(kb_isdigit, c); }  // str.isdigit
bool is_upper_cp(uint32_t c) { return c < 0x80 ? (c >= 'A' && c <= 'Z') : KB_IN(kb_upper, c); }
bool is_lower_cp(uint32_t c) { return c < 0x80 ? (c >= 'a' && c <= 'z') : KB_IN(kb_lower, c); }
bool is_title_cp(uint32_t c) { return c >= 0x80 && KB_IN(kb_cased, c) && !is_upper_cp(c) && !is_lower_cp(c); }
// str.isupper() của cả chuỗi: có ký tự hoa và không có ký tự thường/title
bool str_isupper(const U& s) {
    bool cased = false;
    for (char32_t c : s) {
        if (is_lower_cp(c) || is_title_cp(c)) return false;
        if (!cased && is_upper_cp(c)) cased = true;
    }
    return cased;
}
bool is_ascii_u(const U& s) { for (char32_t c : s) if (c >= 0x80) return false; return true; }
bool is_ascii_s(const S& s) { for (unsigned char c : s) if (c >= 0x80) return false; return true; }

U to_u32(const S& s) {
    U o;
    o.reserve(s.size());
    size_t i = 0, n = s.size();
    while (i < n) {
        unsigned char c = (unsigned char)s[i];
        uint32_t cp = 0xFFFD;
        int len = 1;
        if (c < 0x80) cp = c;
        else if (c >= 0xC2 && c <= 0xDF && i + 1 < n && ((unsigned char)s[i + 1] & 0xC0) == 0x80) { cp = ((c & 0x1F) << 6) | (s[i + 1] & 0x3F); len = 2; }
        else if (c >= 0xE0 && c <= 0xEF && i + 2 < n && ((unsigned char)s[i + 1] & 0xC0) == 0x80 && ((unsigned char)s[i + 2] & 0xC0) == 0x80) {
            uint32_t v = ((c & 0x0F) << 12) | ((s[i + 1] & 0x3F) << 6) | (s[i + 2] & 0x3F);
            if (v >= 0x800 && !(v >= 0xD800 && v <= 0xDFFF)) { cp = v; len = 3; }
        } else if (c >= 0xF0 && c <= 0xF4 && i + 3 < n && ((unsigned char)s[i + 1] & 0xC0) == 0x80 && ((unsigned char)s[i + 2] & 0xC0) == 0x80 && ((unsigned char)s[i + 3] & 0xC0) == 0x80) {
            uint32_t v = ((c & 0x07) << 18) | ((s[i + 1] & 0x3F) << 12) | ((s[i + 2] & 0x3F) << 6) | (s[i + 3] & 0x3F);
            if (v >= 0x10000 && v <= 0x10FFFF) { cp = v; len = 4; }
        }
        o.push_back(cp);
        i += len;
    }
    return o;
}
void put_u8(S& o, uint32_t cp) {
    if (cp < 0x80) o.push_back((char)cp);
    else if (cp < 0x800) { o.push_back((char)(0xC0 | (cp >> 6))); o.push_back((char)(0x80 | (cp & 0x3F))); }
    else if (cp < 0x10000) { o.push_back((char)(0xE0 | (cp >> 12))); o.push_back((char)(0x80 | ((cp >> 6) & 0x3F))); o.push_back((char)(0x80 | (cp & 0x3F))); }
    else { o.push_back((char)(0xF0 | (cp >> 18))); o.push_back((char)(0x80 | ((cp >> 12) & 0x3F))); o.push_back((char)(0x80 | ((cp >> 6) & 0x3F))); o.push_back((char)(0x80 | (cp & 0x3F))); }
}
S to_u8(const U& u) { S o; o.reserve(u.size()); for (char32_t c : u) put_u8(o, c); return o; }
U lit(const char* s) { return to_u32(S(s)); }

bool map_find(const uint32_t (*ent)[3], size_t n, const uint32_t* pool, uint32_t cp, const uint32_t*& out, uint32_t& len) {
    size_t lo = 0, hi = n;
    while (lo < hi) {
        size_t m = (lo + hi) / 2;
        if (cp < ent[m][0]) hi = m;
        else if (cp > ent[m][0]) lo = m + 1;
        else { out = pool + ent[m][1]; len = ent[m][2]; return true; }
    }
    return false;
}
U lower_u(const U& s) {   // str.lower() (không xử lý ς cuối từ của tiếng Hy Lạp: ghi trong APP_KB.md)
    U o; o.reserve(s.size());
    for (char32_t c : s) {
        if (c < 0x80) { o.push_back((c >= 'A' && c <= 'Z') ? c + 32 : c); continue; }
        const uint32_t* p; uint32_t n;
        if (map_find(kb_lowmap_ent, kb_lowmap_n, kb_lowmap_pool, c, p, n)) { for (uint32_t i = 0; i < n; i++) o.push_back(p[i]); }
        else o.push_back(c);
    }
    return o;
}
U fold_u(const U& s) {   // bỏ dấu: đ->d, NFD rồi bỏ dấu kết hợp
    U o; o.reserve(s.size());
    for (char32_t c : s) {
        if (c < 0x80) { o.push_back(c); continue; }
        if (c == 0x111) { o.push_back('d'); continue; }
        if (c == 0x110) { o.push_back('D'); continue; }
        if (c >= 0xAC00 && c <= 0xD7A3) {   // Hangul: phân rã chuẩn thành jamo
            uint32_t si = c - 0xAC00, l = 0x1100 + si / 588, v = 0x1161 + (si % 588) / 28, t = 0x11A7 + si % 28;
            o.push_back(l); o.push_back(v); if (t != 0x11A7) o.push_back(t);
            continue;
        }
        const uint32_t* p; uint32_t n;
        if (map_find(kb_foldmap_ent, kb_foldmap_n, kb_foldmap_pool, c, p, n)) { for (uint32_t i = 0; i < n; i++) o.push_back(p[i]); }
        else o.push_back(c);
    }
    return o;
}
S fold8(const S& s) { return is_ascii_s(s) ? s : to_u8(fold_u(to_u32(s))); }

U strip_u(const U& s) {
    size_t a = 0, b = s.size();
    while (a < b && is_space(s[a])) a++;
    while (b > a && is_space(s[b - 1])) b--;
    return s.substr(a, b - a);
}
U rstrip_u(const U& s) { size_t b = s.size(); while (b > 0 && is_space(s[b - 1])) b--; return s.substr(0, b); }
U collapse_ws(const U& s) {   // " ".join(s.split())
    U o; bool pend = false;
    for (char32_t c : s) {
        if (is_space(c)) { pend = !o.empty(); continue; }
        if (pend) { o.push_back(' '); pend = false; }
        o.push_back(c);
    }
    return o;
}
S collapse8(const S& s) { return to_u8(collapse_ws(to_u32(s))); }
std::vector<U> split_ws(const U& s) {
    std::vector<U> o; U cur;
    for (char32_t c : s) {
        if (is_space(c)) { if (!cur.empty()) { o.push_back(cur); cur.clear(); } }
        else cur.push_back(c);
    }
    if (!cur.empty()) o.push_back(cur);
    return o;
}
bool starts_with(const S& s, const char* p) { size_t n = strlen(p); return s.size() >= n && s.compare(0, n, p) == 0; }

// ============================ danh sách từ ============================
struct Sets {
    std::unordered_set<S> stop, stop_acc, common_cap, self_forms, qwords, time_words;
    Sets() {
        for (size_t i = 0; i < KB_STOP_N; i++) stop.insert(KB_STOP[i]);
        for (size_t i = 0; i < KB_STOP_ACC_N; i++) stop_acc.insert(KB_STOP_ACC[i]);
        for (size_t i = 0; i < KB_COMMON_CAP_N; i++) common_cap.insert(KB_COMMON_CAP[i]);
        for (size_t i = 0; i < KB_SELF_FORMS_N; i++) self_forms.insert(KB_SELF_FORMS[i]);
        for (size_t i = 0; i < KB_QWORDS_N; i++) qwords.insert(KB_QWORDS[i]);
        for (size_t i = 0; i < KB_TIME_WORDS_N; i++) time_words.insert(KB_TIME_WORDS[i]);
    }
};
const Sets& W() { static const Sets s; return s; }

// ============================ tokens / split_text ============================
bool word_char(uint32_t c) { return (c >= '0' && c <= '9') || (c >= 'A' && c <= 'Z') || (c >= 'a' && c <= 'z') || (c >= 0xC0 && c <= 0x1EF9); }   // [0-9A-Za-zÀ-ỹ]
std::vector<std::pair<size_t, size_t>> word_runs(const U& t) {
    std::vector<std::pair<size_t, size_t>> r;
    size_t i = 0, n = t.size();
    while (i < n) {
        if (!word_char(t[i])) { i++; continue; }
        size_t j = i;
        while (j < n && word_char(t[j])) j++;
        r.emplace_back(i, j);
        i = j;
    }
    return r;
}

std::vector<S> tokens_u(const U& text, bool bigrams) {
    std::vector<S> out;
    S prev;   // rỗng = None
    const Sets& st = W();
    auto runs = word_runs(text);
    for (size_t i = 0; i < runs.size(); i++) {
        U orig = text.substr(runs[i].first, runs[i].second - runs[i].first);
        U w = lower_u(orig);
        if (w.size() > 40) { prev.clear(); continue; }
        U f = fold_u(w);
        bool name = i > 0 && is_upper_cp(orig[0]) && !str_isupper(orig);
        S f8 = to_u8(f), w8 = to_u8(w);
        bool same = (w == f);
        if ((st.stop.count(f8) && !name && (same || st.stop_acc.count(w8))) || (f.size() == 1 && !is_digit_cp(f[0]))) { prev.clear(); continue; }
        out.push_back(f8);
        if (!same) out.push_back(w8);
        if (bigrams && !prev.empty()) out.push_back(prev + "_" + f8);
        prev = f8;
    }
    return out;
}
std::vector<S> tokens8(const S& text, bool bigrams) { return tokens_u(to_u32(text), bigrams); }

// _SENT = (?<=[.!?…])\s+|\n+  (re.split)
std::vector<U> split_sent(const U& t) {
    std::vector<U> out;
    size_t last = 0, i = 0, n = t.size();
    auto punct = [](char32_t c) { return c == '.' || c == '!' || c == '?' || c == 0x2026; };
    while (i < n) {
        if (i > 0 && punct(t[i - 1]) && is_space(t[i])) {
            size_t j = i; while (j < n && is_space(t[j])) j++;
            out.push_back(t.substr(last, i - last)); last = j; i = j; continue;
        }
        if (t[i] == '\n') {
            size_t j = i; while (j < n && t[j] == '\n') j++;
            out.push_back(t.substr(last, i - last)); last = j; i = j; continue;
        }
        i++;
    }
    out.push_back(t.substr(last));
    return out;
}
std::vector<U> split_text_u(const U& text0, size_t maxlen) {
    U text = strip_u(text0);
    std::vector<U> parts;
    if (text.size() <= maxlen) { if (!text.empty()) parts.push_back(text); return parts; }
    U cur;
    for (U s : split_sent(text)) {
        if (s.empty()) continue;
        while (s.size() > maxlen) {
            if (!cur.empty()) { parts.push_back(cur); cur.clear(); }
            parts.push_back(s.substr(0, maxlen));
            s = s.substr(maxlen);
        }
        if (cur.size() + s.size() + 1 > maxlen && !cur.empty()) { parts.push_back(cur); cur = s; }
        else { U t = cur; t.push_back(' '); t += s; cur = strip_u(t); }   // (cur + " " + s).strip()
    }
    if (!cur.empty()) parts.push_back(cur);
    return parts;
}

// ============================ dict giữ thứ tự chèn (như dict của Python) ============================
template <class K, class V> struct ODict {
    std::vector<std::pair<K, V>> items;
    std::unordered_map<K, size_t> idx;
    bool has(const K& k) const { return idx.find(k) != idx.end(); }
    V* find(const K& k) { auto it = idx.find(k); return it == idx.end() ? nullptr : &items[it->second].second; }
    const V* find(const K& k) const { auto it = idx.find(k); return it == idx.end() ? nullptr : &items[it->second].second; }
    V& ref(const K& k) {   // d[k] (tạo giá trị mặc định nếu chưa có, đặt ở cuối)
        auto it = idx.find(k);
        if (it != idx.end()) return items[it->second].second;
        idx.emplace(k, items.size());
        items.emplace_back(k, V());
        return items.back().second;
    }
    void set(const K& k, const V& v) { ref(k) = v; }
    size_t size() const { return items.size(); }
    bool empty() const { return items.empty(); }
    void clear() { items.clear(); idx.clear(); }
};

double round_n(double x, int nd) {   // round(x, nd) của Python: làm tròn đúng theo giá trị nhị phân chính xác
    char buf[64];
    snprintf(buf, sizeof buf, "%.*f", nd, x);
    return strtod(buf, nullptr);
}

void civil_from_days(long long z, int& y, int& m, int& d) {
    z += 719468;
    long long era = (z >= 0 ? z : z - 146096) / 146097;
    unsigned doe = (unsigned)(z - era * 146097);
    unsigned yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
    long long yy = (long long)yoe + era * 400;
    unsigned doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    unsigned mp = (5 * doy + 2) / 153;
    d = (int)(doy - (153 * mp + 2) / 5 + 1);
    m = (int)(mp < 10 ? mp + 3 : mp - 9);
    y = (int)(yy + (m <= 2));
}
S ddmm(int64_t ts, int tz) {
    long long t = (long long)ts + tz;
    long long days = t >= 0 ? t / 86400 : -((-t + 86399) / 86400);
    int y, m, d;
    civil_from_days(days, y, m, d);
    char b[16];
    snprintf(b, sizeof b, "%02d/%02d", d, m);
    return b;
}

// ============================ RAG: chỉ mục + BM25 ============================
struct Chunk {
    bool live = true;
    S src, conv, role, text;
    int idx = 0, pos = 0;
    int64_t ts = 0;
};
struct Hit { int cid; double score; };

struct Rag {
    S bot = "Yui";
    int tz = 0;
    std::vector<Chunk> chunks;
    std::vector<std::vector<S>> terms;
    std::vector<int> dl;
    std::unordered_map<S, std::map<int, int>> inv;       // từ -> {đoạn -> tf}, theo mã đoạn tăng dần = thứ tự chèn
    std::unordered_map<S, ODict<S, int>> graph;          // đồ thị đồng xuất hiện giữa các từ khoá
    std::unordered_map<S, std::vector<int>> convs;
    std::unordered_map<S, std::vector<int>> files;
    std::vector<int> pending;
    long long N = 0, totlen = 0;

    int add_chunk(const S& src, const S& conv, const S& role, const S& text, int64_t ts) {
        int cid = (int)chunks.size();
        std::vector<S> toks = tokens8(text, true);
        ODict<S, int> tf;
        for (const S& t : toks) tf.ref(t)++;
        auto& lst = convs[conv];
        Chunk c;
        c.src = src; c.conv = conv; c.role = role; c.text = text; c.ts = ts;
        c.pos = (int)lst.size(); c.idx = c.pos;
        chunks.push_back(std::move(c));
        lst.push_back(cid);
        std::vector<S> tl;
        tl.reserve(tf.size());
        for (auto& kv : tf.items) tl.push_back(kv.first);
        terms.push_back(std::move(tl));
        dl.push_back((int)toks.size());
        totlen += (long long)toks.size();
        N++;
        for (auto& kv : tf.items) inv[kv.first][cid] = kv.second;
        files[src].push_back(cid);
        pending.push_back(cid);
        return cid;
    }

    int drop_source(const S& src) {
        auto it = files.find(src);
        if (it == files.end()) return 0;
        std::vector<int> cids = it->second;
        files.erase(it);
        int n = 0;
        for (int cid : cids) {
            Chunk& ch = chunks[cid];
            if (!ch.live) continue;
            for (const S& t : terms[cid]) {
                auto d = inv.find(t);
                if (d != inv.end()) { d->second.erase(cid); if (d->second.empty()) inv.erase(d); }
            }
            auto& lst = convs[ch.conv];
            lst.erase(std::remove(lst.begin(), lst.end(), cid), lst.end());
            totlen -= dl[cid];
            N--;
            ch.live = false;
            terms[cid].clear();
            n++;
        }
        return n;
    }

    static void top_neighbors(const ODict<S, int>& nb, size_t n, std::vector<std::pair<S, int>>& out) {   // sorted(items, key=-count)[:n], ổn định
        out = nb.items;
        std::stable_sort(out.begin(), out.end(), [](const std::pair<S, int>& a, const std::pair<S, int>& b) { return a.second > b.second; });
        if (out.size() > n) out.resize(n);
    }

    void graph_batch(const std::vector<int>& cids) {
        long long Nn = std::max<long long>(2, N);
        for (int cid : cids) {
            if (!chunks[cid].live) continue;
            std::vector<std::pair<S, double>> cand;
            for (const S& t : terms[cid]) {
                if (!is_ascii_s(t) || t.find('_') != S::npos || to_u32(t).size() < 3) continue;
                bool alld = true;
                for (unsigned char ch : t) if (!(ch >= '0' && ch <= '9')) { alld = false; break; }
                if (alld) continue;
                auto it = inv.find(t);
                size_t df = it == inv.end() ? 0 : it->second.size();
                cand.emplace_back(t, std::log((double)Nn / (double)(1 + df)));
            }
            std::stable_sort(cand.begin(), cand.end(), [](const std::pair<S, double>& a, const std::pair<S, double>& b) { return a.second > b.second; });   // sort(reverse=True) ổn định
            if (cand.size() > 8) cand.resize(8);
            for (size_t i = 0; i < cand.size(); i++)
                for (size_t j = i + 1; j < cand.size(); j++) {
                    const S &a = cand[i].first, &b = cand[j].first;
                    graph[a].ref(b)++;
                    graph[b].ref(a)++;
                }
        }
        for (auto& kv : graph) {
            if (kv.second.size() > 80) {
                std::vector<std::pair<S, int>> top;
                top_neighbors(kv.second, 40, top);
                ODict<S, int> nd;
                for (auto& p : top) nd.set(p.first, p.second);
                kv.second = std::move(nd);
            }
        }
    }
    void commit() { std::vector<int> p; p.swap(pending); graph_batch(p); }

    // ---- BM25 ----
    std::vector<Hit> search(const S& query, int k, const S* role) {
        std::vector<S> qt = tokens8(query, true);
        if (qt.empty() || N == 0) return {};
        ODict<S, double> w;
        for (const S& t : qt) w.set(t, t.find('_') != S::npos ? 1.6 : 1.0);
        for (const S& t : qt) {
            if (!is_ascii_s(t) || t.find('_') != S::npos) continue;
            auto g = graph.find(t);
            if (g == graph.end()) continue;
            std::vector<std::pair<S, int>> top;
            top_neighbors(g->second, 3, top);
            for (auto& p : top)
                if (p.second >= 2 && !w.has(p.first)) w.set(p.first, 0.35);
        }
        double avg = (double)totlen / (double)std::max<long long>(1, N);
        const double k1 = 1.5, b = 0.75;
        ODict<int, double> sc;
        for (auto& kv : w.items) {
            auto it = inv.find(kv.first);
            if (it == inv.end() || it->second.empty()) continue;
            double wq = kv.second;
            double idf = std::log(1 + ((double)(N - (long long)it->second.size()) + 0.5) / ((double)it->second.size() + 0.5));
            for (auto& pc : it->second) {
                int cid = pc.first, tf = pc.second;
                double add = wq * idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * dl[cid] / avg));
                sc.ref(cid) += add;
            }
        }
        if (sc.empty()) return {};
        int64_t lo = chunks[sc.items[0].first].ts, hi = lo;
        for (auto& kv : sc.items) { lo = std::min(lo, chunks[kv.first].ts); hi = std::max(hi, chunks[kv.first].ts); }
        std::vector<std::pair<int, double>> kept;
        for (auto& kv : sc.items) {
            const Chunk& ch = chunks[kv.first];
            if (role && !role->empty() && ch.role != *role) continue;
            double s = kv.second;
            if (hi > lo) s *= 1 + 0.15 * (double)(ch.ts - lo) / (double)(hi - lo);
            kept.emplace_back(kv.first, s);
        }
        std::stable_sort(kept.begin(), kept.end(), [](const std::pair<int, double>& a, const std::pair<int, double>& c) { return a.second > c.second; });
        std::vector<std::pair<int, double>> ranked = kept;
        if (k >= 0 && ranked.size() > (size_t)k) ranked.resize(k);
        ODict<int, double> fin;
        for (auto& p : ranked) fin.set(p.first, p.second);
        size_t nb = (size_t)std::max(1, k / 2);
        for (size_t r = 0; r < ranked.size() && r < nb; r++) {
            int cid = ranked[r].first;
            double s = ranked[r].second;
            const Chunk& ch = chunks[cid];
            const std::vector<int>& lst = convs[ch.conv];
            for (int d = -1; d <= 1; d += 2) {
                long long j = (long long)ch.pos + d;
                if (j >= 0 && j < (long long)lst.size() && !fin.has(lst[j]) && chunks[lst[j]].live && chunks[lst[j]].role != ch.role) fin.set(lst[j], s * 0.45);
            }
        }
        std::vector<std::pair<int, double>> top = fin.items;
        std::stable_sort(top.begin(), top.end(), [](const std::pair<int, double>& a, const std::pair<int, double>& c) { return a.second > c.second; });
        if (top.size() > (size_t)(k + 2)) top.resize(k + 2);
        std::stable_sort(top.begin(), top.end(), [&](const std::pair<int, double>& a, const std::pair<int, double>& c) {
            const Chunk &x = chunks[a.first], &y = chunks[c.first];
            int cmp = x.conv.compare(y.conv);
            if (cmp != 0) return cmp < 0;
            return x.idx < y.idx;
        });
        std::vector<Hit> out;
        for (auto& p : top) out.push_back({p.first, round_n(p.second, 3)});
        return out;
    }

    S who(const Chunk& ch) const { return ch.role == "user" ? S("bạn") : ch.role == "assistant" ? bot : ch.role == "kb" ? S("tài liệu") : S("ghi chú"); }
    S fmt(const Chunk& ch) const { return "[" + ddmm(ch.ts, tz) + "] " + who(ch) + ": " + collapse8(ch.text); }

    S recall(const S& query, int chars, int k) {
        std::vector<Hit> hits = search(query, k, nullptr);
        if (hits.empty()) return "không tìm thấy trong lịch sử";
        std::vector<Hit> best = hits;
        std::stable_sort(best.begin(), best.end(), [](const Hit& a, const Hit& b) { return a.score > b.score; });
        std::unordered_map<int, S> lines;
        int budget = chars;
        for (const Hit& h : best) {
            U line = to_u32(fmt(chunks[h.cid]));
            if (budget < 40) break;
            if ((int)line.size() > budget) { line = line.substr(0, budget - 1); line.push_back(0x2026); }
            budget -= (int)line.size() + 1;
            lines[h.cid] = to_u8(line);
        }
        S out;
        bool first = true;
        for (const Hit& h : hits) {
            auto it = lines.find(h.cid);
            if (it == lines.end()) continue;
            if (!first) out += "\n";
            out += it->second;
            first = false;
        }
        return out;
    }
};

// ============================ tiện ích cho Graphify ============================
double pysum(const std::vector<double>& v) {   // sum() của Python 3.12: cộng bù Neumaier
    if (v.empty()) return 0.0;
    double f = v[0], c = 0.0;
    for (size_t i = 1; i < v.size(); i++) {
        double x = v[i], t = f + x;
        if (std::fabs(f) >= std::fabs(x)) c += (f - t) + x;
        else c += (x - t) + f;
        f = t;
    }
    if (c != 0.0 && std::isfinite(c)) f += c;
    return f;
}

// Bộ khung ASCII 1 byte / 1 ký tự để dùng std::regex mà vị trí khớp = chỉ số ký tự: chữ/số Unicode -> '_' (là \w), khoảng trắng -> ' ', còn lại -> '\x01'.
S skeleton(const U& f) {
    S o; o.reserve(f.size());
    for (char32_t c : f) {
        if (c < 0x80) o.push_back(is_space(c) ? ' ' : (char)c);
        else if (is_space(c)) o.push_back(' ');
        else if (is_word_cp(c)) o.push_back('_');
        else o.push_back('\x01');
    }
    return o;
}
struct Rx {
    std::vector<std::regex> durable, intent;
    std::regex place, ban;
    Rx() {
        for (size_t i = 0; i < KB_DURABLE_N; i++) durable.emplace_back(KB_DURABLE[i].rx, std::regex::ECMAScript | std::regex::optimize);
        for (size_t i = 0; i < KB_INTENT_N; i++) intent.emplace_back(KB_INTENT[i].rx, std::regex::ECMAScript | std::regex::optimize);
        place = std::regex(R"(\b(o dau|dang o|song o|o)\b)", std::regex::ECMAScript | std::regex::optimize);
        ban = std::regex(R"(\bban\s+\w+)", std::regex::ECMAScript | std::regex::optimize);
    }
};
const Rx& RX() { static const Rx r; return r; }
bool rx_search(const std::regex& rx, const S& s, long long* pos = nullptr) {
    try {
        std::smatch m;
        if (std::regex_search(s, m, rx)) { if (pos) *pos = (long long)m.position(0); return true; }
    } catch (const std::regex_error&) {}
    return false;
}

std::vector<std::pair<size_t, size_t>> letter_runs(const U& t) {   // _LETTERS = [^\W\d_]+
    std::vector<std::pair<size_t, size_t>> r;
    size_t i = 0, n = t.size();
    while (i < n) {
        if (!is_letter_cp(t[i])) { i++; continue; }
        size_t j = i;
        while (j < n && is_letter_cp(t[j])) j++;
        r.emplace_back(i, j);
        i = j;
    }
    return r;
}
std::vector<std::pair<size_t, U>> cap_phrases(const U& text) {
    std::vector<std::pair<size_t, U>> out;
    std::vector<U> cur;
    size_t start = 0, last_end = 0;
    auto join = [](const std::vector<U>& v) { U o; for (size_t i = 0; i < v.size(); i++) { if (i) o.push_back(' '); o += v[i]; } return o; };
    for (auto& run : letter_runs(text)) {
        U w = text.substr(run.first, run.second - run.first);
        if (is_upper_cp(w[0]) && !str_isupper(w)) {
            if (!cur.empty() && strip_u(text.substr(last_end, run.first - last_end)).empty()) cur.push_back(w);
            else {
                if (!cur.empty()) out.emplace_back(start, join(cur));
                cur.assign(1, w);
                start = run.first;
            }
            last_end = run.second;
        } else {
            if (!cur.empty()) out.emplace_back(start, join(cur));
            cur.clear();
        }
    }
    if (!cur.empty()) out.emplace_back(start, join(cur));
    return out;
}

// _NUM = (\d[\d.,]*)\s*(triệu|trieu|tr|nghìn|nghin|ngàn|ngan|k|đồng|dong|vnđ|vnd|tuổi|tuoi|phút|phut|giờ|gio|ngày|ngay|tháng|thang|năm|nam|%|kg|km)?  (re.I)
// Giữ NGUYÊN thứ tự chọn của Python (rẽ nhánh theo thứ tự): "kg"/"km" luôn bị "k" nuốt trước, "tr" khớp đầu "trong"... (ghi ở APP_KB.md §9). \d chỉ nhận chữ số ASCII.
const char* const NUM_UNITS[] = {"triệu", "trieu", "tr", "nghìn", "nghin", "ngàn", "ngan", "k", "đồng", "dong", "vnđ", "vnd", "tuổi", "tuoi", "phút", "phut", "giờ", "gio", "ngày", "ngay", "tháng", "thang", "năm", "nam", "%", "kg", "km"};
struct NumMatch { size_t start, end; U g1; U unit; bool has_unit; };
std::vector<NumMatch> num_matches(const U& t) {
    static std::vector<U> units;
    if (units.empty()) for (const char* u : NUM_UNITS) units.push_back(lower_u(lit(u)));
    std::vector<NumMatch> out;
    size_t i = 0, n = t.size();
    while (i < n) {
        if (!(t[i] >= '0' && t[i] <= '9')) { i++; continue; }
        size_t j = i + 1;
        while (j < n && ((t[j] >= '0' && t[j] <= '9') || t[j] == '.' || t[j] == ',')) j++;
        NumMatch m;
        m.start = i; m.g1 = t.substr(i, j - i);
        size_t k = j;
        while (k < n && is_space(t[k])) k++;
        m.has_unit = false;
        size_t end = k;
        for (const U& u : units) {
            if (k + u.size() > n) continue;
            bool ok = true;
            for (size_t x = 0; x < u.size(); x++) {
                U one(1, t[k + x]);
                if (lower_u(one)[0] != u[x]) { ok = false; break; }
            }
            if (ok) { m.has_unit = true; m.unit = t.substr(k, u.size()); end = k + u.size(); break; }
        }
        m.end = end;
        out.push_back(m);
        i = std::max(end, i + 1);
    }
    return out;
}
bool num_value(const NumMatch& m, S& out) {   // str(int(float(g1...) * mult)) hoặc không có giá trị (Python sẽ ném lỗi/None)
    S g;
    for (char32_t c : m.g1) if (c != '.') g.push_back(c == ',' ? '.' : (char)c);
    // float() chỉ nhận: chữ số, tối đa một dấu '.', ít nhất một chữ số
    int dots = 0, digits = 0;
    for (char c : g) { if (c == '.') dots++; else digits++; }
    if (dots > 1 || digits == 0) return false;
    double v = strtod(g.c_str(), nullptr);
    double mul = 1;
    if (m.has_unit) {
        S u = to_u8(lower_u(m.unit));
        for (size_t i = 0; i < KB_MULT_N; i++) if (u == KB_MULT[i].unit) { mul = (double)KB_MULT[i].mul; break; }
    }
    double r = v * mul;
    if (!std::isfinite(r)) return false;
    char buf[400];
    snprintf(buf, sizeof buf, "%.0f", std::trunc(r));
    out = buf;
    return true;
}

// _DATE = \b(\d{1,2})\s*/\s*(\d{1,2})(?:\s*/\s*(\d{2,4}))?\b  (nhánh thử theo đúng thứ tự quay lui của re)
bool is_dg(const U& t, size_t i) { return i < t.size() && t[i] >= '0' && t[i] <= '9'; }
bool word_at(const U& t, long long i) { return i >= 0 && i < (long long)t.size() && is_word_cp(t[i]); }
std::vector<std::pair<int, int>> date_matches(const U& t) {
    std::vector<std::pair<int, int>> out;
    size_t n = t.size(), i = 0;
    while (i < n) {
        if (!is_dg(t, i) || word_at(t, (long long)i - 1)) { i++; continue; }
        bool found = false;
        size_t end = 0;
        int d1 = 0, d2 = 0;
        for (int l1 = (is_dg(t, i + 1) ? 2 : 1); l1 >= 1 && !found; l1--) {
            size_t p = i + l1;
            size_t q = p; while (q < n && is_space(t[q])) q++;
            if (q >= n || t[q] != '/') continue;
            q++;
            while (q < n && is_space(t[q])) q++;
            if (!is_dg(t, q)) continue;
            for (int l2 = (is_dg(t, q + 1) ? 2 : 1); l2 >= 1 && !found; l2--) {
                size_t p2 = q + l2;
                // nhánh có nhóm 3: \s*/\s*(\d{2,4})
                size_t r = p2; while (r < n && is_space(t[r])) r++;
                if (r < n && t[r] == '/') {
                    size_t r2 = r + 1; while (r2 < n && is_space(t[r2])) r2++;
                    int avail = 0; while (avail < 4 && is_dg(t, r2 + avail)) avail++;
                    for (int l3 = avail; l3 >= 2 && !found; l3--) {
                        size_t e = r2 + l3;
                        if (!word_at(t, (long long)e)) { found = true; end = e; }
                    }
                    if (found) { d1 = atoi(to_u8(t.substr(i, l1)).c_str()); d2 = atoi(to_u8(t.substr(q, l2)).c_str()); break; }
                }
                // nhánh không có nhóm 3
                if (!word_at(t, (long long)p2)) { found = true; end = p2; d1 = atoi(to_u8(t.substr(i, l1)).c_str()); d2 = atoi(to_u8(t.substr(q, l2)).c_str()); }
            }
        }
        if (found) { out.emplace_back(d1, d2); i = std::max(end, i + 1); }
        else i++;
    }
    return out;
}

std::vector<S> dedup(const std::vector<S>& v) {
    std::vector<S> o; std::unordered_set<S> seen;
    for (const S& x : v) if (seen.insert(x).second) o.push_back(x);
    return o;
}
std::vector<S> entities_u(const U& text) {
    std::vector<S> out;
    const Sets& st = W();
    for (auto& pw : cap_phrases(text)) {
        size_t pos = pw.first;
        const U& w = pw.second;
        bool first = pos == 0;
        if (!first) {
            U pre = rstrip_u(text.substr(0, pos));
            if (!pre.empty()) { char32_t c = pre.back(); first = (c == '.' || c == '!' || c == '?' || c == ':' || c == '\n'); }
        }
        if (st.common_cap.count(to_u8(w)) || (first && w.find(U(1, U' ')) == U::npos && text.size() > 90)) continue;
        U joined;
        for (const U& p : split_ws(w)) {
            if (st.common_cap.count(to_u8(p))) continue;
            if (!joined.empty()) joined.push_back(' ');
            joined += p;
        }
        if (!joined.empty()) out.push_back("p:" + to_u8(fold_u(lower_u(joined))));
    }
    for (auto& d : date_matches(text)) out.push_back("d:" + std::to_string(d.first) + "/" + std::to_string(d.second));
    for (auto& m : num_matches(text)) {
        S v;
        if (num_value(m, v) && (v.size() >= 2 || m.has_unit)) out.push_back("n:" + v);
    }
    return dedup(out);
}

bool has_self_u(const U& text) {
    const Sets& st = W();
    for (auto& r : word_runs(text)) {
        if (is_upper_cp(text[r.first])) continue;
        if (st.self_forms.count(to_u8(text.substr(r.first, r.second - r.first)))) return true;
    }
    return false;
}
bool placey_search(const U& t) {   // (?:ở|sang|về|tới|đến|chuyển)\s+(\w) và chữ cái đó viết hoa
    static std::vector<U> alts;
    if (alts.empty()) for (const char* a : {"ở", "sang", "về", "tới", "đến", "chuyển"}) alts.push_back(lit(a));
    size_t n = t.size(), i = 0;
    while (i < n) {
        bool matched = false;
        for (const U& a : alts) {
            if (i + a.size() > n || t.compare(i, a.size(), a) != 0) continue;
            size_t k = i + a.size(), s = k;
            while (k < n && is_space(t[k])) k++;
            if (k == s || k >= n || !is_word_cp(t[k])) continue;
            if (is_upper_cp(t[k])) return true;
            i = k + 1; matched = true; break;
        }
        if (!matched) i++;
    }
    return false;
}

S intent_of(const S& q) {
    U f = fold_u(lower_u(to_u32(q)));
    S sk = skeleton(f);
    for (size_t i = 0; i < KB_INTENT_N; i++) if (rx_search(RX().intent[i], sk)) return KB_INTENT[i].name;
    return "other";
}
bool fact_key_u(const S& text, S& key) {
    U f = fold_u(lower_u(to_u32(text)));
    S sk = skeleton(f);
    for (size_t i = 0; i < KB_DURABLE_N; i++) {
        long long pos;
        if (!rx_search(RX().durable[i], sk, &pos)) continue;
        std::set<S> subj;
        for (const S& t : tokens_u(f.substr(0, (size_t)pos), false))
            if (is_ascii_s(t) && !W().time_words.count(t)) subj.insert(t);
        S s;
        int n = 0;
        for (const S& t : subj) { if (n == 4) break; if (n) s += " "; s += t; n++; }
        key = S(KB_DURABLE[i].name) + "|" + (s.empty() ? S("user") : s);
        return true;
    }
    return false;
}

std::vector<U> split_after(const U& t, const char* punct) {   // re.split(r"(?<=[punct])\s+")
    std::vector<U> out;
    U p = lit(punct);
    size_t last = 0, i = 0, n = t.size();
    while (i < n) {
        if (i > 0 && p.find(t[i - 1]) != U::npos && is_space(t[i])) {
            size_t j = i; while (j < n && is_space(t[j])) j++;
            out.push_back(t.substr(last, i - last)); last = j; i = j; continue;
        }
        i++;
    }
    out.push_back(t.substr(last));
    return out;
}
std::vector<U> split_facts_u(const U& text0, size_t maxlen = 170) {
    U text = collapse_ws(text0);
    std::vector<U> out;
    if (text.size() <= maxlen) { if (text.size() >= 8) out.push_back(text); return out; }
    for (const U& piece : split_after(text, ".!?\xE2\x80\xA6;")) {
        U s = strip_u(piece);
        if (s.size() < 8) continue;
        while (s.size() > maxlen) {
            long long c1 = -1, c2 = -1;
            for (long long x = (long long)std::min(s.size(), maxlen) - 1; x >= 0; x--) if (s[x] == ',') { c1 = x; break; }
            for (long long x = (long long)std::min(s.size(), maxlen) - 1; x >= 0; x--) if (s[x] == ' ') { c2 = x; break; }
            long long cut = std::max(c1, c2);
            size_t at = cut > 40 ? (size_t)cut : maxlen;
            out.push_back(strip_u(s.substr(0, at)));
            U rest = s.substr(at);
            size_t a = 0, b = rest.size();
            while (a < b && (rest[a] == ' ' || rest[a] == ',')) a++;
            while (b > a && (rest[b - 1] == ' ' || rest[b - 1] == ',')) b--;
            s = rest.substr(a, b - a);
        }
        if (!s.empty()) out.push_back(s);
    }
    return out;
}

U focus_u(const U& text0, const std::set<S>& qset, size_t maxlen) {
    U text = collapse_ws(text0);
    if (text.size() <= maxlen) return text;
    std::vector<U> clauses;
    for (const U& c : split_after(text, ",;:")) { U s = strip_u(c); if (!s.empty()) clauses.push_back(s); }
    if (clauses.empty()) return text.substr(0, maxlen);
    size_t best = 0; int bs = -1;
    for (size_t i = 0; i < clauses.size(); i++) {
        int sc = 0;
        for (const S& t : tokens_u(clauses[i], false)) if (qset.count(t)) sc++;
        if (sc > bs) { bs = sc; best = i; }
    }
    U out = clauses[best];
    size_t i = best, j = best;
    while (true) {
        bool grew = false;
        if (i > 0 && clauses[i - 1].size() + out.size() + 1 <= maxlen) { i--; out = clauses[i] + U(1, U' ') + out; grew = true; }
        if (j + 1 < clauses.size() && out.size() + clauses[j + 1].size() + 1 <= maxlen) { j++; out = out + U(1, U' ') + clauses[j]; grew = true; }
        if (!grew) break;
    }
    if (out.size() <= maxlen) return out;
    U cut = rstrip_u(out.substr(0, maxlen - 1));
    cut.push_back(0x2026);
    return cut;
}

// ============================ Graphify ============================
struct Fact {
    int cid = 0;
    S text, who;
    int64_t ts = 0;
    std::vector<S> ents;
    S key;            // rỗng = None
    bool stale = false;
    S conv;
    int pos = 0;
};
struct QueryResult {
    S context, intent;
    std::vector<S> facts;
    std::vector<int> ids;
    double conf = 0.0;
};
const char* const NOT_FOUND = "không tìm thấy trong lịch sử";

struct Graphify {
    Rag& rag;
    std::vector<Fact> facts;
    std::unordered_map<int, std::vector<int>> by_cid;
    std::unordered_map<S, std::vector<int>> ent;
    std::unordered_map<S, ODict<S, int>> co;
    std::unordered_map<S, std::vector<int>> by_key;
    size_t mark = 0;

    explicit Graphify(Rag& r) : rag(r) {}
    void reset() { facts.clear(); by_cid.clear(); ent.clear(); co.clear(); by_key.clear(); mark = 0; }

    int add(int cid, const S& text, const S& who, int64_t ts, const std::vector<S>& ents, const S& key, const S& conv, int pos) {
        int fid = (int)facts.size();
        Fact f;
        f.cid = cid; f.text = text; f.who = who; f.ts = ts; f.ents = ents; f.key = key; f.conv = conv; f.pos = pos;
        facts.push_back(std::move(f));
        by_cid[cid].push_back(fid);
        for (const S& e : ents) ent[e].push_back(fid);
        for (size_t i = 0; i < ents.size(); i++)
            for (size_t j = i + 1; j < ents.size(); j++) {
                co[ents[i]].ref(ents[j])++;
                co[ents[j]].ref(ents[i])++;
            }
        return fid;
    }
    bool live(int fid) const { return rag.chunks[facts[fid].cid].live; }

    void supersede() {
        std::vector<S> order;
        std::unordered_map<S, std::vector<int>> groups;
        for (int fid = 0; fid < (int)facts.size(); fid++) {
            Fact& f = facts[fid];
            f.stale = false;
            if (!f.key.empty() && live(fid)) {
                if (!groups.count(f.key)) order.push_back(f.key);
                groups[f.key].push_back(fid);
            }
        }
        by_key.clear();
        for (const S& key : order) {
            auto& fids = groups[key];
            std::stable_sort(fids.begin(), fids.end(), [&](int a, int b) { return facts[a].ts != facts[b].ts ? facts[a].ts < facts[b].ts : a < b; });
            by_key[key] = fids;
            for (size_t i = 0; i + 1 < fids.size(); i++) facts[fids[i]].stale = true;
        }
    }
    void compact() {
        std::vector<Fact> keep;
        for (const Fact& f : facts) if (rag.chunks[f.cid].live) keep.push_back(f);
        size_t m = mark;
        reset();
        mark = m;
        for (const Fact& f : keep) add(f.cid, f.text, f.who, f.ts, f.ents, f.key, f.conv, f.pos);
        supersede();
    }
    bool is_echo(int cid, const Chunk& ch) {
        (void)cid;
        if (ch.role != "assistant") return false;
        auto it = rag.convs.find(ch.conv);
        if (it == rag.convs.end()) return false;
        long long j = (long long)ch.pos - 1;
        if (j < 0 || j >= (long long)it->second.size()) return false;
        const Chunk& prev = rag.chunks[it->second[j]];
        if (!prev.live || prev.role != "user") return false;
        std::set<S> a, b;
        for (const S& t : tokens8(ch.text, false)) a.insert(t);
        for (const S& t : tokens8(prev.text, false)) b.insert(t);
        if (b.empty()) return false;
        size_t inter = 0;
        for (const S& t : b) if (a.count(t)) inter++;
        return (double)inter / (double)b.size() >= 0.7;
    }

    int sync() {
        int n_new = 0;
        for (size_t cid = mark; cid < rag.chunks.size(); cid++) {
            const Chunk& ch = rag.chunks[cid];
            if (!ch.live || is_echo((int)cid, ch)) continue;
            S base = ch.text;
            if (ch.role == "note" || ch.role == "kb") {
                U u = to_u32(ch.text);
                size_t lim = std::min<size_t>(80, u.size());
                size_t at = U::npos;
                for (size_t i = 0; i + 1 < lim; i++) if (u[i] == ':' && u[i + 1] == ' ') { at = i; break; }
                if (at != U::npos) base = to_u8(u.substr(at + 2));
            }
            for (const U& su : split_facts_u(to_u32(base))) {
                S s = to_u8(su);
                std::vector<S> ents = entities_u(su);
                std::set<S> toks;
                for (const S& t : tokens_u(su, false)) {
                    if (!is_ascii_s(t) || t.size() < 3) continue;
                    bool alld = true;
                    for (unsigned char c : t) if (!(c >= '0' && c <= '9')) { alld = false; break; }
                    if (!alld) toks.insert(t);
                }
                std::vector<std::pair<size_t, S>> by_df;
                for (const S& t : toks) { auto it = rag.inv.find(t); by_df.emplace_back(it == rag.inv.end() ? 0 : it->second.size(), t); }
                std::sort(by_df.begin(), by_df.end());
                for (size_t i = 0; i < by_df.size() && i < 5; i++) ents.push_back("c:" + by_df[i].second);
                S key;
                if (ch.role == "user" || ch.role == "note" || ch.role == "kb") fact_key_u(s, key);
                add((int)cid, s, ch.role, ch.ts, dedup(ents), key, ch.conv, ch.pos);
                n_new++;
            }
        }
        mark = rag.chunks.size();
        if (n_new) supersede();
        if (facts.size() > 400) {
            size_t dead = 0;
            for (const Fact& f : facts) if (!rag.chunks[f.cid].live) dead++;
            if ((double)dead > (double)facts.size() * 0.35) compact();
        }
        for (auto& kv : co) {
            if (kv.second.size() > 60) {
                std::vector<std::pair<S, int>> top;
                Rag::top_neighbors(kv.second, 30, top);
                ODict<S, int> nd;
                for (auto& p : top) nd.set(p.first, p.second);
                kv.second = std::move(nd);
            }
        }
        return n_new;
    }

    ODict<S, double> ppr(const ODict<S, double>& seeds, int hops = 2, double damp = 0.5) {
        ODict<S, double> score, frontier = seeds;
        for (auto& kv : seeds.items) score.ref(kv.first) += kv.second;
        for (int h = 0; h < hops; h++) {
            ODict<S, double> nxt;
            for (auto& fe : frontier.items) {
                auto it = co.find(fe.first);
                if (it == co.end() || it->second.empty()) continue;
                long long tot = 0;
                for (auto& nb : it->second.items) tot += nb.second;
                for (auto& nb : it->second.items) nxt.ref(nb.first) += fe.second * damp * nb.second / (double)tot;
            }
            for (auto& kv : nxt.items) score.ref(kv.first) += kv.second;
            frontier = std::move(nxt);
        }
        return score;
    }

    static QueryResult nothing(const S& intent, double conf = 0.0) { QueryResult r; r.context = NOT_FOUND; r.intent = intent; r.conf = conf; return r; }

    QueryResult query(const S& q, int chars, int k, bool abstain) {
        sync();
        S it = intent_of(q);
        if (k <= 0) { k = 4; for (size_t i = 0; i < KB_K_BY_INTENT_N; i++) if (it == KB_K_BY_INTENT[i].name) k = KB_K_BY_INTENT[i].k; }
        U qlow = lower_u(to_u32(q));
        std::vector<S> qt;
        for (const S& t : tokens8(q, false)) if (is_ascii_s(t)) qt.push_back(t);
        std::set<S> qset;
        for (const S& t : qt) if (!W().qwords.count(t)) qset.insert(t);
        if (qset.empty()) qset.insert(qt.begin(), qt.end());
        S qsk = skeleton(fold_u(qlow));
        bool wants_place = rx_search(RX().place, qsk);
        bool q_self = has_self_u(to_u32(q));
        if (qset.empty() || facts.empty()) return nothing(it);
        long long N = std::max<long long>(2, rag.N);
        ODict<S, double> idf;
        for (const S& t : qset) { auto p = rag.inv.find(t); size_t df = p == rag.inv.end() ? 0 : p->second.size(); idf.set(t, std::log(1 + (double)N / (double)(1 + df))); }
        auto idf_vals = [&]() { std::vector<double> v; for (auto& kv : idf.items) v.push_back(kv.second); return v; };
        double idf_sum = pysum(idf_vals());
        if (idf_sum == 0.0) idf_sum = 1.0;
        double idf_max = 0; for (auto& kv : idf.items) idf_max = std::max(idf_max, kv.second);
        auto isum = [&](const std::set<S>& ts) { std::vector<double> v; for (const S& t : ts) v.push_back(*idf.find(t)); return pysum(v); };   // ts đã có thứ tự từ điển
        std::unordered_map<S, S> typed;
        for (auto& r : word_runs(qlow)) {
            U w = qlow.substr(r.first, r.second - r.first);
            U f = fold_u(w);
            if (f != w) typed[to_u8(f)] = to_u8(w);
        }
        auto covers = [&](const S& text) {
            std::set<S> ft;
            for (const S& t : tokens8(text, false)) ft.insert(t);
            std::set<S> out;
            for (const S& t : qset) {
                auto ty = typed.find(t);
                if (ty != typed.end() ? ft.count(ty->second) : ft.count(t)) out.insert(t);
            }
            return out;
        };
        std::vector<S> qents = entities_u(to_u32(q));
        for (const S& t : qset) if (ent.count("c:" + t)) qents.push_back("c:" + t);
        auto in_qents = [&](const S& e) { return std::find(qents.begin(), qents.end(), e) != qents.end(); };

        // ---- 1. ứng viên ----
        ODict<int, int> cand;
        ODict<int, double> bm_chunks;
        for (const Hit& h : rag.search(q, 12, nullptr)) bm_chunks.set(h.cid, h.score);
        double bm_max = 0; bool first_bm = true;
        for (auto& kv : bm_chunks.items) { if (first_bm || kv.second > bm_max) bm_max = kv.second; first_bm = false; }
        if (first_bm || bm_max == 0.0) bm_max = 1.0;
        for (auto& kv : bm_chunks.items) { auto b = by_cid.find(kv.first); if (b != by_cid.end()) for (int fid : b->second) cand.set(fid, 1); }
        for (const S& e : qents) { auto p = ent.find(e); if (p == ent.end()) continue; const auto& v = p->second; for (size_t i = v.size() > 40 ? v.size() - 40 : 0; i < v.size(); i++) cand.set(v[i], 1); }
        ODict<S, double> seeds;
        for (const S& e : qents) if (ent.count(e)) seeds.set(e, 1.0);
        std::vector<std::pair<int, double>> bm_sorted = bm_chunks.items;
        std::stable_sort(bm_sorted.begin(), bm_sorted.end(), [](const std::pair<int, double>& a, const std::pair<int, double>& b) { return a.second > b.second; });
        for (size_t i = 0; i < bm_sorted.size() && i < 3; i++) {
            auto b = by_cid.find(bm_sorted[i].first);
            if (b == by_cid.end()) continue;
            for (int fid : b->second)
                for (const S& e : facts[fid].ents) {
                    if (starts_with(e, "c:")) continue;
                    double* cur = seeds.find(e);
                    seeds.set(e, std::max(cur ? *cur : 0.0, 0.5));
                }
        }
        ODict<S, double> pr;
        if (!seeds.empty()) pr = ppr(seeds);
        {
            std::vector<std::pair<S, double>> top = pr.items;
            std::stable_sort(top.begin(), top.end(), [](const std::pair<S, double>& a, const std::pair<S, double>& b) { return a.second > b.second; });
            for (size_t i = 0; i < top.size() && i < 12; i++) {
                if (!(top[i].second > 0.08)) continue;
                auto p = ent.find(top[i].first);
                if (p == ent.end()) continue;
                const auto& v = p->second;
                for (size_t x = v.size() > 12 ? v.size() - 12 : 0; x < v.size(); x++) cand.set(v[x], 1);
            }
        }
        std::vector<int> candv;
        for (auto& kv : cand.items) if (live(kv.first)) candv.push_back(kv.first);

        std::unordered_map<int, std::set<S>> ftcache;
        auto ft_of = [&](int fid) -> const std::set<S>& {
            auto f = ftcache.find(fid);
            if (f == ftcache.end()) {
                std::set<S> s;
                for (const S& t : tokens8(facts[fid].text, false)) if (is_ascii_s(t)) s.insert(t);
                f = ftcache.emplace(fid, std::move(s)).first;
            }
            return f->second;
        };
        auto lex_of = [&](int fid) {
            const std::set<S>& ft = ft_of(fid);
            std::set<S> inter;
            for (const S& t : qset) if (ft.count(t)) inter.insert(t);
            return isum(inter) / idf_sum;
        };
        std::vector<S> rare;
        {
            std::vector<std::pair<double, S>> rr;
            for (const S& t : qset) if (rag.inv.count(t)) rr.emplace_back(-*idf.find(t), t);
            std::sort(rr.begin(), rr.end());
            for (size_t i = 0; i < rr.size() && i < 2; i++) rare.push_back(rr[i].second);
        }
        std::vector<int> stage1;
        for (int fid : candv) {
            const std::set<S>& ft = ft_of(fid);
            bool any = false;
            for (const S& t : rare) if (ft.count(t)) { any = true; break; }
            if (any) stage1.push_back(fid);   // (chú ý: bản Python viết `(gen) or cand` nên KHÔNG bao giờ lùi về cand; giữ nguyên)
        }
        {
            std::vector<std::pair<double, int>> keyed;
            for (int fid : stage1) {
                auto bm = bm_chunks.find(facts[fid].cid);
                keyed.emplace_back(-(lex_of(fid) + 0.5 * (bm ? *bm : 0.0) / bm_max), fid);
            }
            std::stable_sort(keyed.begin(), keyed.end(), [](const std::pair<double, int>& a, const std::pair<double, int>& b) { return a.first < b.first; });
            stage1.clear();
            for (size_t i = 0; i < keyed.size() && i < 6; i++) stage1.push_back(keyed[i].second);
        }
        ODict<S, std::set<S>> bridges;
        for (int fid : stage1) {
            double lx = lex_of(fid);
            if (lx <= 0.2) continue;
            const std::set<S>& ft = ft_of(fid);
            for (const S& e : facts[fid].ents)
                if (starts_with(e, "p:") && !in_qents(e)) { auto& s = bridges.ref(e); s.insert(ft.begin(), ft.end()); }
        }
        for (auto& be : bridges.items) {
            auto p = ent.find(be.first);
            if (p == ent.end()) continue;
            const auto& v = p->second;
            for (size_t x = v.size() > 15 ? v.size() - 15 : 0; x < v.size(); x++)
                if (live(v[x]) && std::find(candv.begin(), candv.end(), v[x]) == candv.end()) candv.push_back(v[x]);
        }
        if (candv.empty()) return nothing(it);

        // ---- 2. chấm điểm cấp sự kiện ----
        int64_t lo = facts[candv[0]].ts, hi = lo;
        for (int fid : candv) { lo = std::min(lo, facts[fid].ts); hi = std::max(hi, facts[fid].ts); }
        struct Raw { double v[6]; };
        std::vector<Raw> raws;
        for (int fid : candv) {
            const Fact& f = facts[fid];
            const std::set<S>& ft = ft_of(fid);
            double lex = isum(covers(f.text)) / idf_sum;
            double eo = 0.0;
            if (!qents.empty()) {
                std::vector<double> v;
                for (const S& e : f.ents) if (in_qents(e)) { auto p = ent.find(e); v.push_back(1.0 / std::sqrt(1 + (double)(p == ent.end() ? 0 : p->second.size()))); }
                eo = pysum(v) / (double)qents.size();
            }
            double gr;
            {
                std::vector<double> v;
                for (const S& e : f.ents) { const double* p = pr.find(e); v.push_back(p ? *p : 0.0); }
                gr = pysum(v) / std::sqrt(1 + (double)f.ents.size());
            }
            auto bmp = bm_chunks.find(f.cid);
            double bm = (bmp ? *bmp : 0.0) / bm_max;
            double bonus = 0.0;
            bool has_d = false, has_n = false, has_p = false;
            for (const S& e : f.ents) { if (starts_with(e, "d:")) has_d = true; if (starts_with(e, "n:")) has_n = true; if (starts_with(e, "p:")) has_p = true; }
            if (it == "when" && has_d) bonus += 0.3;
            if (it == "howmany" && has_n) bonus += 0.3;
            if ((it == "who" || it == "where") && has_p) bonus += 0.15;
            double rec = hi > lo ? (double)(f.ts - lo) / (double)(hi - lo) : 0.0;
            bonus += (it == "history" ? 0.2 * (1 - rec) : 0.1 * rec) + (f.who == "user" ? 0.08 : f.who == "assistant" ? -0.05 : 0.02);
            if (f.stale && it != "history") bonus -= 0.6;
            if (wants_place && (starts_with(f.key, "noi o") || placey_search(to_u32(f.text)))) bonus += 0.4;
            U ftext = to_u32(f.text);
            if (q_self) {
                bool selfy = has_self_u(ftext);
                bool others = false;
                for (const S& e : f.ents) if (starts_with(e, "p:") && !in_qents(e) && !bridges.has(e)) { others = true; break; }
                bonus += selfy ? 0.3 : 0.0;
                bonus -= (others && !selfy) ? 0.3 : 0.0;
                bool anyq = false;
                for (const S& e : qents) if (std::find(f.ents.begin(), f.ents.end(), e) != f.ents.end()) { anyq = true; break; }
                if (rx_search(RX().ban, skeleton(fold_u(lower_u(ftext)))) && !selfy && !anyq) bonus -= 0.15;
            }
            double br = 0.0;
            for (auto& be : bridges.items) {
                if (std::find(f.ents.begin(), f.ents.end(), be.first) == f.ents.end()) continue;
                std::set<S> rest;
                for (const S& t : qset) if (!be.second.count(t)) rest.insert(t);
                size_t inter = 0;
                for (const S& t : rest) if (ft.count(t)) inter++;
                br = std::max(br, (double)inter / (double)std::max<size_t>(1, rest.size()));
            }
            Raw r; r.v[0] = lex; r.v[1] = eo; r.v[2] = gr; r.v[3] = bm; r.v[4] = bonus; r.v[5] = br;
            raws.push_back(r);
        }
        double mx[4];
        for (int i = 0; i < 4; i++) { double m = raws[0].v[i]; for (auto& r : raws) m = std::max(m, r.v[i]); mx[i] = (m == 0.0) ? 1.0 : m; }
        std::unordered_map<int, double> final_;
        std::vector<std::pair<int, double>> fin_items;
        for (size_t i = 0; i < candv.size(); i++) {
            const double* r = raws[i].v;
            double val = 1.0 * r[0] / mx[0] + 0.8 * r[1] / mx[1] + 0.5 * r[2] / mx[2] + 0.6 * r[3] / mx[3] + r[4] + 1.2 * r[5];
            final_[candv[i]] = val;
            fin_items.emplace_back(candv[i], val);
        }

        // ---- 3. cắt tỉa: MMR + dừng khi đã phủ đủ ý ----
        std::vector<std::pair<int, double>> order = fin_items;
        std::stable_sort(order.begin(), order.end(), [](const std::pair<int, double>& a, const std::pair<int, double>& b) { return a.second > b.second; });
        double top = order[0].second;
        size_t lim40 = std::min<size_t>(40, order.size());
        std::vector<int> chosen;
        std::set<S> covered;
        while ((int)chosen.size() < k) {
            int best = -1; double bs = -1e9;
            for (size_t oi = 0; oi < lim40; oi++) {
                int fid = order[oi].first;
                if (std::find(chosen.begin(), chosen.end(), fid) != chosen.end()) continue;
                const std::set<S>& a = ft_of(fid);
                double sim = 0.0; bool firstsim = true;
                for (int c : chosen) {
                    const std::set<S>& b = ft_of(c);
                    size_t inter = 0;
                    for (const S& t : a) if (b.count(t)) inter++;
                    double sc = (double)inter / (double)std::max<size_t>(1, a.size() + b.size() - inter);
                    if (firstsim || sc > sim) sim = sc;
                    firstsim = false;
                }
                if (sim > 0.8) continue;
                double s = 0.75 * final_[fid] - 0.25 * sim * top;
                if (s > bs) { best = fid; bs = s; }
            }
            if (best < 0 || final_[best] < 0.32 * top) break;
            chosen.push_back(best);
            for (const S& t : covers(facts[best].text)) covered.insert(t);
            double cov = isum(covered) / idf_sum;
            if (cov >= 0.9 && (int)chosen.size() >= ((it == "list" || it == "history") ? 2 : 1) && it != "list") break;
        }
        if (it == "history") {
            std::vector<int> snapshot = chosen;
            for (int fid : snapshot) {
                auto g = by_key.find(facts[fid].key);
                if (facts[fid].key.empty() || g == by_key.end()) continue;
                int first_live = -1;
                for (int x : g->second) if (live(x)) { first_live = x; break; }
                if (first_live >= 0 && std::find(chosen.begin(), chosen.end(), first_live) == chosen.end() && (int)chosen.size() < k + 1) chosen.push_back(first_live);
            }
        }
        if (it != "history") {
            for (size_t i = 0; i < chosen.size(); i++) {
                const Fact& f = facts[chosen[i]];
                if (f.stale && !f.key.empty()) {
                    int newest = -1;
                    for (int x = 0; x < (int)facts.size(); x++) {
                        if (facts[x].key != f.key || !live(x)) continue;
                        if (newest < 0 || facts[x].ts > facts[newest].ts || (facts[x].ts == facts[newest].ts && x > newest)) newest = x;
                    }
                    if (newest >= 0) chosen[i] = newest;
                }
            }
            std::vector<int> uniq;
            for (int c : chosen) if (std::find(uniq.begin(), uniq.end(), c) == uniq.end()) uniq.push_back(c);
            chosen = uniq;
        }

        // ---- 4. độ tự tin ----
        std::set<S> cov_ents, got;
        for (int fid : chosen) {
            for (const S& e : facts[fid].ents) cov_ents.insert(e);
            for (const S& t : covers(facts[fid].text)) got.insert(t);
        }
        double cov = isum(got) / idf_sum;
        {
            double alt = 0;
            if (!qents.empty()) {
                std::set<S> qs(qents.begin(), qents.end());
                size_t inter = 0;
                for (const S& e : qs) if (cov_ents.count(e)) inter++;
                alt = 0.5 * (double)inter / (double)qents.size();
            }
            cov = std::max(cov, alt);
        }
        double conf = round_n(std::min(1.0, 0.7 * cov + 0.3 * std::min(1.0, top)), 2);
        std::vector<S> keyt;
        for (const S& t : qset) if (*idf.find(t) >= 0.75 * idf_max) keyt.push_back(t);
        if (abstain) {
            bool keyhit = false;
            for (const S& t : keyt) if (got.count(t)) { keyhit = true; break; }
            bool entshit = false;
            for (const S& e : qents) if (!starts_with(e, "c:") && cov_ents.count(e)) { entshit = true; break; }
            if (cov < 0.34 || (!keyt.empty() && !keyhit && !entshit)) return nothing(it, conf);
        }

        // ---- 5. kể lại gọn, theo thời gian ----
        std::stable_sort(chosen.begin(), chosen.end(), [&](int a, int b) {
            if (facts[a].ts != facts[b].ts) return facts[a].ts < facts[b].ts;
            int c = facts[a].conv.compare(facts[b].conv);
            if (c != 0) return c < 0;
            return facts[a].pos < facts[b].pos;
        });
        std::vector<U> lines;
        long long budget = chars;
        for (size_t j = 0; j < chosen.size(); j++) {
            const Fact& f = facts[chosen[j]];
            S who = f.who == "user" ? S("bạn") : f.who == "assistant" ? rag.bot : f.who == "kb" ? S("tài liệu") : S("ghi chú");
            S tag = f.stale ? " (cũ)" : "";
            U head = to_u32("[" + ddmm(f.ts, rag.tz) + "] " + who + tag + ": ");
            long long room = budget - (long long)head.size() - (long long)(chosen.size() - j - 1) * 30;
            if (room < 25) break;
            U body = focus_u(to_u32(f.text), qset, (size_t)std::min(room, budget - (long long)head.size()));
            lines.push_back(head + body);
            budget -= (long long)lines.back().size() + 1;
        }
        QueryResult r;
        if (lines.empty()) r.context = NOT_FOUND;
        else for (size_t i = 0; i < lines.size(); i++) { if (i) r.context += "\n"; r.context += to_u8(lines[i]); }
        for (int fid : chosen) { r.facts.push_back(facts[fid].text); r.ids.push_back(fid); }
        r.conf = conf;
        r.intent = it;
        return r;
    }

    S recall(const S& q, int chars, int k, bool fallback) {
        QueryResult r = query(q, chars, k, true);
        if (!r.facts.empty() || !fallback) return r.context;
        return rag.recall(q, std::min(chars, 160), 2);
    }
};

// ============================ JSON (chỉ để kiểm thử/gỡ lỗi) ============================
S jstr(const S& s) {
    S o = "\"";
    for (unsigned char c : s) {
        if (c == '"') o += "\\\"";
        else if (c == '\\') o += "\\\\";
        else if (c == '\n') o += "\\n";
        else if (c == '\r') o += "\\r";
        else if (c == '\t') o += "\\t";
        else if (c < 0x20) { char b[8]; snprintf(b, sizeof b, "\\u%04x", c); o += b; }
        else o.push_back((char)c);
    }
    return o + "\"";
}
S jnum(double d) { char b[64]; snprintf(b, sizeof b, "%.17g", d); return b; }
S jarr(const std::vector<S>& v) { S o = "["; for (size_t i = 0; i < v.size(); i++) { if (i) o += ","; o += jstr(v[i]); } return o + "]"; }
int put(const S& s, char* out, int cap) {
    int n = (int)s.size();
    if (out && cap > 0) { int m = std::min(n, cap - 1); memcpy(out, s.data(), (size_t)m); out[m] = 0; }
    return n;
}

// ============================ nạp tài liệu ============================
std::vector<U> split_paras(const U& t) {   // re.split(r"\n\s*\n", text)
    std::vector<U> out;
    size_t last = 0, i = 0, n = t.size();
    while (i < n) {
        if (t[i] == '\n') {
            size_t j = i + 1;
            while (j < n && is_space(t[j])) j++;
            long long k = -1;
            for (size_t x = j; x > i + 1; x--) if (t[x - 1] == '\n') { k = (long long)x - 1; break; }
            if (k >= 0) { out.push_back(t.substr(last, i - last)); last = (size_t)k + 1; i = last; continue; }
        }
        i++;
    }
    out.push_back(t.substr(last));
    return out;
}
S doc_title(const S& rel) {
    size_t sl = rel.rfind('/');
    S name = sl == S::npos ? rel : rel.substr(sl + 1);
    S low = to_u8(lower_u(to_u32(name)));
    for (const char* suf : {".md", ".txt", ".enc"})
        if (low.size() >= strlen(suf) && low.compare(low.size() - strlen(suf), strlen(suf), suf) == 0) { name = name.substr(0, name.size() - strlen(suf)); break; }
    return starts_with(rel, "kb/_tomtat/") ? "tóm tắt " + name : name;
}
}  // namespace

struct kb_t {
    std::mutex mu;
    Rag rag;
    Graphify g;
    kb_t() : g(rag) {}
};

#define KB_LOCK std::lock_guard<std::mutex> lk(kb->mu)

extern "C" {
KB_API int kb_abi(void) { return KB_ABI; }
KB_API kb_t* kb_new(const char* bot_name) {
    kb_t* kb = new kb_t();
    if (bot_name && *bot_name) kb->rag.bot = bot_name;
    return kb;
}
KB_API void kb_free(kb_t* kb) { delete kb; }
KB_API void kb_set_tz(kb_t* kb, int off) { if (!kb) return; KB_LOCK; kb->rag.tz = off; }
KB_API void kb_clear(kb_t* kb) {
    if (!kb) return;
    KB_LOCK;
    S bot = kb->rag.bot; int tz = kb->rag.tz;
    kb->g.reset();
    kb->rag = Rag();
    kb->rag.bot = bot; kb->rag.tz = tz;
}

KB_API int kb_add_chat_turn(kb_t* kb, const char* src, const char* conv, const char* role, const char* text, int64_t ts) {
    if (!kb || !src || !conv || !role || !text) return -1;
    KB_LOCK;
    int n = 0;
    for (const U& part : split_text_u(collapse_ws(to_u32(text)), 500)) { kb->rag.add_chunk(src, conv, role, to_u8(part), ts); n++; }
    return n;
}
KB_API int kb_add_doc(kb_t* kb, const char* kind, const char* src, const char* text, int64_t ts) {
    if (!kb || !kind || !src || !text) return -1;
    KB_LOCK;
    bool is_note = S(kind) == "note";
    S title = doc_title(src), role = is_note ? "note" : "kb";
    int i = 0;
    for (const U& para : split_paras(to_u32(text)))
        for (const U& part : split_text_u(para, 500)) {
            S t = to_u8(part);
            kb->rag.add_chunk(src, src, role, (i == 0 || !is_note) ? title + ": " + t : t, ts);
            i++;
        }
    return i;
}
KB_API int kb_remember(kb_t* kb, const char* src, const char* conv, const char* user, const char* assistant, int64_t ts) {
    if (!kb || !src || !conv || !user || !assistant) return 0;
    KB_LOCK;
    U u = collapse_ws(to_u32(user)), a = collapse_ws(to_u32(assistant));
    if (u.size() > 4000) u.resize(4000);
    if (a.size() > 4000) a.resize(4000);
    if (u.empty() || a.empty()) return 0;
    size_t before = kb->rag.pending.size();
    for (int r = 0; r < 2; r++)
        for (const U& part : split_text_u(r == 0 ? u : a, 500)) kb->rag.add_chunk(src, conv, r == 0 ? "user" : "assistant", to_u8(part), ts);
    std::vector<int> mine(kb->rag.pending.begin() + (long)before, kb->rag.pending.end());
    kb->rag.pending.resize(before);
    kb->rag.graph_batch(mine);
    return 1;
}
KB_API int kb_drop_source(kb_t* kb, const char* src) { if (!kb || !src) return 0; KB_LOCK; return kb->rag.drop_source(src); }
KB_API void kb_commit(kb_t* kb) { if (!kb) return; KB_LOCK; kb->rag.commit(); }

KB_API int kb_recall(kb_t* kb, const char* query, int chars, int k, int mode, char* out, int cap) {
    if (!kb || !query) return put("", out, cap);
    KB_LOCK;
    if (chars < 0) chars = 0;
    S r;
    if (mode == 0) r = kb->rag.recall(query, chars, k > 0 ? k : 4);
    else r = kb->g.recall(query, chars, k, mode == 2);
    return put(r, out, cap);
}
KB_API void kb_stats(kb_t* kb, kb_stats_t* o) {
    if (!kb || !o) return;
    KB_LOCK;
    o->chunks = (int)kb->rag.N; o->files = (int)kb->rag.files.size(); o->terms = (int)kb->rag.inv.size();
    int live = 0, stale = 0;
    for (size_t i = 0; i < kb->g.facts.size(); i++) if (kb->g.live((int)i)) { live++; if (kb->g.facts[i].stale) stale++; }
    o->facts = live; o->entities = (int)kb->g.ent.size(); o->superseded = stale;
}

// ---- gỡ lỗi / kiểm thử ----
KB_API int kb_dbg_fold(const char* text, char* out, int cap) { return put(text ? to_u8(fold_u(to_u32(text))) : "", out, cap); }
KB_API int kb_dbg_tokens(const char* text, int bigrams, char* out, int cap) { return put(jarr(tokens8(text ? text : "", bigrams != 0)), out, cap); }
KB_API int kb_dbg_split_text(const char* text, int maxlen, char* out, int cap) {
    std::vector<S> v;
    for (const U& p : split_text_u(to_u32(text ? text : ""), (size_t)(maxlen > 0 ? maxlen : 500))) v.push_back(to_u8(p));
    return put(jarr(v), out, cap);
}
KB_API int kb_dbg_entities(const char* text, char* out, int cap) { return put(jarr(entities_u(to_u32(text ? text : ""))), out, cap); }
KB_API int kb_dbg_fact_key(const char* text, char* out, int cap) { S k; return put(fact_key_u(text ? text : "", k) ? jstr(k) : S("null"), out, cap); }
KB_API int kb_dbg_intent(const char* text, char* out, int cap) { return put(intent_of(text ? text : ""), out, cap); }
KB_API int kb_dbg_has_self(const char* text) { return has_self_u(to_u32(text ? text : "")) ? 1 : 0; }
KB_API int kb_dbg_split_facts(const char* text, char* out, int cap) {
    std::vector<S> v;
    for (const U& p : split_facts_u(to_u32(text ? text : ""))) v.push_back(to_u8(p));
    return put(jarr(v), out, cap);
}
KB_API int kb_dbg_search_json(kb_t* kb, const char* query, int k, const char* role, char* out, int cap) {
    if (!kb) return put("[]", out, cap);
    KB_LOCK;
    S rl = role ? role : "";
    std::vector<Hit> hits = kb->rag.search(query ? query : "", k > 0 ? k : 4, rl.empty() ? nullptr : &rl);
    S o = "[";
    for (size_t i = 0; i < hits.size(); i++) {
        const Chunk& c = kb->rag.chunks[hits[i].cid];
        if (i) o += ",";
        o += "{\"cid\":" + std::to_string(hits[i].cid) + ",\"score\":" + jnum(hits[i].score) + ",\"src\":" + jstr(c.src) + ",\"conv\":" + jstr(c.conv) + ",\"i\":" + std::to_string(c.idx) +
             ",\"p\":" + std::to_string(c.pos) + ",\"r\":" + jstr(c.role) + ",\"t\":" + jstr(c.text) + ",\"ts\":" + std::to_string((long long)c.ts) + "}";
    }
    return put(o + "]", out, cap);
}
KB_API int kb_dbg_query_json(kb_t* kb, const char* query, int chars, int k, int abstain, char* out, int cap) {
    if (!kb) return put("{}", out, cap);
    KB_LOCK;
    QueryResult r = kb->g.query(query ? query : "", chars, k, abstain != 0);
    S ids = "[";
    for (size_t i = 0; i < r.ids.size(); i++) { if (i) ids += ","; ids += std::to_string(r.ids[i]); }
    ids += "]";
    return put("{\"context\":" + jstr(r.context) + ",\"facts\":" + jarr(r.facts) + ",\"conf\":" + jnum(r.conf) + ",\"intent\":" + jstr(r.intent) + ",\"ids\":" + ids + "}", out, cap);
}
KB_API int kb_dbg_chunks_json(kb_t* kb, char* out, int cap) {
    if (!kb) return put("[]", out, cap);
    KB_LOCK;
    S o = "[";
    bool first = true;
    for (size_t i = 0; i < kb->rag.chunks.size(); i++) {
        const Chunk& c = kb->rag.chunks[i];
        if (!c.live) continue;
        if (!first) o += ",";
        first = false;
        o += "{\"cid\":" + std::to_string(i) + ",\"src\":" + jstr(c.src) + ",\"conv\":" + jstr(c.conv) + ",\"i\":" + std::to_string(c.idx) + ",\"p\":" + std::to_string(c.pos) +
             ",\"r\":" + jstr(c.role) + ",\"t\":" + jstr(c.text) + ",\"ts\":" + std::to_string((long long)c.ts) + "}";
    }
    return put(o + "]", out, cap);
}
KB_API int kb_dbg_facts_json(kb_t* kb, char* out, int cap) {
    if (!kb) return put("[]", out, cap);
    KB_LOCK;
    kb->g.sync();
    S o = "[";
    bool first = true;
    for (size_t i = 0; i < kb->g.facts.size(); i++) {
        if (!kb->g.live((int)i)) continue;
        const Fact& f = kb->g.facts[i];
        if (!first) o += ",";
        first = false;
        o += "{\"cid\":" + std::to_string(f.cid) + ",\"text\":" + jstr(f.text) + ",\"who\":" + jstr(f.who) + ",\"ts\":" + std::to_string((long long)f.ts) + ",\"ents\":" + jarr(f.ents) +
             ",\"key\":" + (f.key.empty() ? S("null") : jstr(f.key)) + ",\"stale\":" + (f.stale ? "true" : "false") + "}";
    }
    return put(o + "]", out, cap);
}
}  // extern "C"
