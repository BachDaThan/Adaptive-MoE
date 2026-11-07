"""
vi_filter.py - lọc chất lượng văn bản web tiếng Việt (FineWeb-2 vie_Latn) trước khi đưa vào pre-train. Thuần Python, không cần thư viện ngoài.

Kiểu Gopher/C4 + heuristic cho web Việt (rác SEO, cá độ, bán hàng, trang liệt kê):
  - đủ dài, đủ số từ; độ dài từ trung bình hợp lý
  - tỉ lệ chữ cái, chữ có dấu tiếng Việt, chữ HOA, ký hiệu/emoji, chữ số
  - dòng lặp, đoạn lặp, cụm n-gram lặp (kiểu spam/keyword stuffing)
  - dòng quá ngắn (menu, breadcrumb, danh sách link), dòng kết thúc bằng "..." / "Xem thêm"
  - từ khoá rác (nhà cái, cá cược, xổ số, hotline, zalo...) và nội dung người lớn
  - URL / email dày đặc
Dùng:  from vi_filter import keep;  ok, why = keep(doc)     # why = lý do loại
Thử:   python vi_filter.py file.txt                          # in tỉ lệ giữ/loại theo từng lý do
Ngưỡng chỉnh bằng biến môi trường VF_* (xem THRESH) nếu loại quá tay/quá nhẹ.
"""
import os
import re
import sys
from collections import Counter

VI_MARKS = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")


def _f(name, default):
    return float(os.environ.get("VF_" + name, default))


THRESH = dict(
    min_chars=_f("MIN_CHARS", 300), min_words=_f("MIN_WORDS", 60), max_words=_f("MAX_WORDS", 40000),
    mean_word_lo=_f("MEAN_WORD_LO", 2.2), mean_word_hi=_f("MEAN_WORD_HI", 6.5),       # độ dài từ trung bình (ký tự/từ, tiếng Việt ~3.5-4.5)
    alpha_min=_f("ALPHA_MIN", 0.62), vi_marks_min=_f("VI_MARKS_MIN", 0.10),          # chữ cái / ký tự không-khoảng-trắng; chữ có dấu / chữ cái
    upper_max=_f("UPPER_MAX", 0.30), digit_max=_f("DIGIT_MAX", 0.22), symbol_max=_f("SYMBOL_MAX", 0.12),
    dup_line_frac=_f("DUP_LINE_FRAC", 0.25), dup_line_chars=_f("DUP_LINE_CHARS", 0.30),
    short_line_frac=_f("SHORT_LINE_FRAC", 0.55), ellipsis_line_frac=_f("ELLIPSIS_LINE_FRAC", 0.30),
    top_gram_frac=_f("TOP_GRAM_FRAC", 0.10),         # 1 cụm 3 từ chiếm quá ngần này tổng số cụm => lặp/keyword stuffing
    dup_gram_chars=_f("DUP_GRAM_CHARS", 0.35),       # tỉ lệ ký tự nằm trong các cụm 5 từ bị lặp
    spam_hits=_f("SPAM_HITS", 4), spam_density=_f("SPAM_DENSITY", 0.012), url_density=_f("URL_DENSITY", 0.012),
)

SPAM = re.compile(
    r"(nhà cái|cá cược|cá độ|soi cầu|xổ số|lô đề|đánh bài|game bài|tải app|link vào|nạp tiền|rút tiền|đại lý cấp|kèo nhà|casino|jackpot|slot game|bet\d|"
    r"mua ngay|đặt hàng|giao hàng toàn quốc|liên hệ hotline|hotline\b|zalo\b|giá rẻ nhất|uy tín số 1|bảo hành chính hãng|khuyến mãi sốc|chiết khấu|"
    r"xem phim sex|phim sex|clip nóng|gái gọi|massage kích dục|địt|đụ má|dương vật|âm đạo|làm tình|khiêu dâm|"
    r"click vào đây|đăng ký ngay|tải xuống miễn phí|download free|copyright ©|bản quyền thuộc về|all rights reserved)", re.I)
URL = re.compile(r"(https?://|www\.)\S+|\S+@\S+\.\S+", re.I)
TAIL_JUNK = re.compile(r"(xem thêm|đọc thêm|đọc tiếp|chi tiết|xem tiếp|\.\.\.|…)\s*[»>]*\s*$", re.I)
CITE = re.compile(r"(^|\n)\s*(tags?|từ khoá|từ khóa|chủ đề|bài viết liên quan|tin liên quan|có thể bạn quan tâm|nguồn)\s*:", re.I)


def _stats_fail(t, why):
    return False, why


def keep(doc):
    """-> (True, None) hoặc (False, lý do). Rẻ: ~vài chục µs-vài trăm µs / bài."""
    T = THRESH
    t = doc.strip()
    if len(t) < T["min_chars"]:
        return False, "quá ngắn"
    words = t.split()
    nw = len(words)
    if nw < T["min_words"]:
        return False, "ít từ"
    if nw > T["max_words"]:
        return False, "quá dài"
    mean_w = sum(len(w) for w in words) / nw
    if not T["mean_word_lo"] <= mean_w <= T["mean_word_hi"]:
        return False, "độ dài từ lạ"
    nonspace = [c for c in t if not c.isspace()]
    letters = [c for c in nonspace if c.isalpha()]
    if not letters:
        return False, "không có chữ"
    if len(letters) / len(nonspace) < T["alpha_min"]:
        return False, "ít chữ cái"
    if sum(c.lower() in VI_MARKS for c in letters) / len(letters) < T["vi_marks_min"]:
        return False, "thiếu dấu tiếng Việt"
    if sum(c.isupper() for c in letters) / len(letters) > T["upper_max"]:
        return False, "quá nhiều chữ HOA"
    if sum(c.isdigit() for c in nonspace) / len(nonspace) > T["digit_max"]:
        return False, "quá nhiều chữ số"
    if sum((not c.isalnum()) and c not in ".,;:!?-–—()\"'“”‘’%/" for c in nonspace) / len(nonspace) > T["symbol_max"]:
        return False, "nhiều ký hiệu/emoji"

    lines = [ln.strip() for ln in t.split("\n") if ln.strip()]
    if len(lines) >= 4:
        cl = Counter(lines)
        dup_lines = sum(c - 1 for c in cl.values() if c > 1)
        if dup_lines / len(lines) > T["dup_line_frac"]:
            return False, "lặp dòng"
        if sum(len(l) * (c - 1) for l, c in cl.items() if c > 1) / max(1, sum(len(l) for l in lines)) > T["dup_line_chars"]:
            return False, "lặp dòng"
        if sum(len(l.split()) < 5 for l in lines) / len(lines) > T["short_line_frac"]:
            return False, "nhiều dòng quá ngắn (menu/danh sách)"
        if sum(bool(TAIL_JUNK.search(l)) for l in lines) / len(lines) > T["ellipsis_line_frac"]:
            return False, "dòng cụt 'xem thêm'"
        if len(CITE.findall(t)) >= 3:
            return False, "nhiều khối tags/liên quan"

    low = [w.lower().strip(".,;:!?\"'()") for w in words]
    if nw >= 50:
        g3 = Counter(zip(low, low[1:], low[2:]))
        if g3.most_common(1)[0][1] / max(1, nw - 2) > T["top_gram_frac"]:
            return False, "lặp cụm từ"
        g5 = Counter(zip(low, low[1:], low[2:], low[3:], low[4:]))
        rep = sum(c for g, c in g5.items() if c > 1)
        if rep * 5 / max(1, nw) > T["dup_gram_chars"]:
            return False, "lặp đoạn"

    spam = len(SPAM.findall(t))
    if spam >= T["spam_hits"] or spam / nw > T["spam_density"]:
        return False, "rác/quảng cáo/người lớn"
    if len(URL.findall(t)) / nw > T["url_density"]:
        return False, "nhiều link"
    return True, None


class Counter2:
    """Gom thống kê lý do loại khi lọc luồng."""

    def __init__(self):
        self.kept = 0
        self.rej = Counter()

    def __call__(self, doc):
        ok, why = keep(doc)
        if ok:
            self.kept += 1
        else:
            self.rej[why] += 1
        return ok

    def report(self):
        tot = self.kept + sum(self.rej.values())
        if not tot:
            return "(chưa có bài)"
        top = ", ".join(f"{k} {v}" for k, v in self.rej.most_common(6))
        return f"giữ {self.kept}/{tot} ({100 * self.kept / tot:.0f}%); loại: {top or 'không'}"


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    c = Counter2()
    for blk in open(sys.argv[1], encoding="utf-8", errors="ignore").read().split("\n\n"):
        if blk.strip():
            c(blk)
    print(c.report())
