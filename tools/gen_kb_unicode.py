#!/usr/bin/env python3
"""Sinh native/kb_unicode.inc: bảng Unicode cho kb_core (không cần ICU, chạy được trên Android/Windows/Linux).
Mọi bảng lấy trực tiếp từ unicodedata/str của Python để C++ khớp Python từng ký tự:
  alnum (str.isalnum -> \\w), nd (\\d của re), isdigit (str.isdigit), upper/lower/title (str.isupper/islower trên 1 ký tự), space (str.isspace -> \\s, split()),
  lower_map (str.lower, có thể ra nhiều ký tự), fold_map (NFD rồi bỏ ký tự có combining() != 0; Hangul làm bằng thuật toán trong C++).
Chạy:  python3 tools/gen_kb_unicode.py > native/kb_unicode.inc
Python dùng để sinh bảng sẽ được ghi ở đầu file; sinh lại khi muốn đổi phiên bản Unicode (không ảnh hưởng gì trừ ký tự mới).
"""
import sys
import unicodedata

MAXCP = 0x110000


def runs(pred):
    out, start = [], None
    for cp in range(MAXCP):
        if 0xD800 <= cp <= 0xDFFF:
            ok = False
        else:
            ok = pred(chr(cp))
        if ok and start is None:
            start = cp
        elif not ok and start is not None:
            out.append((start, cp - 1))
            start = None
    if start is not None:
        out.append((start, MAXCP - 1))
    return out


def emit_ranges(name, rs):
    print(f"static const uint32_t {name}[][2] = {{")
    line = []
    for a, b in rs:
        line.append(f"{{0x{a:X},0x{b:X}}}")
        if len(line) == 8:
            print("  " + ",".join(line) + ",")
            line = []
    if line:
        print("  " + ",".join(line) + ",")
    print("};")
    print(f"static const size_t {name}_n = sizeof({name}) / sizeof({name}[0]);")


def emit_map(name, pairs):
    pool, ents = [], []
    for cp, s in pairs:
        ents.append((cp, len(pool), len(s)))
        pool.extend(ord(c) for c in s)
    print(f"static const uint32_t {name}_pool[] = {{")
    for i in range(0, len(pool), 12):
        print("  " + ",".join(f"0x{x:X}" for x in pool[i:i + 12]) + ",")
    print("};")
    print(f"static const uint32_t {name}_ent[][3] = {{")
    for i in range(0, len(ents), 6):
        print("  " + ",".join(f"{{0x{a:X},{b},{c}}}" for a, b, c in ents[i:i + 6]) + ",")
    print("};")
    print(f"static const size_t {name}_n = sizeof({name}_ent) / sizeof({name}_ent[0]);")


def fold1(c):
    c = c.replace("đ", "d").replace("Đ", "D")
    s = unicodedata.normalize("NFD", c)
    return "".join(x for x in s if not unicodedata.combining(x))


print(f"// TỰ SINH bởi tools/gen_kb_unicode.py (Python {sys.version.split()[0]}, Unicode {unicodedata.unidata_version}). KHÔNG sửa tay.")
emit_ranges("kb_alnum", runs(lambda c: c.isalnum()))
emit_ranges("kb_nd", runs(lambda c: unicodedata.category(c) == "Nd"))
emit_ranges("kb_isdigit", runs(lambda c: c.isdigit()))
emit_ranges("kb_upper", runs(lambda c: c.isupper()))
emit_ranges("kb_lower", runs(lambda c: c.islower()))
emit_ranges("kb_cased", runs(lambda c: c.isupper() or c.islower() or c.istitle()))
emit_ranges("kb_space", runs(lambda c: c.isspace()))
lm = []
fm = []
for cp in range(MAXCP):
    if 0xD800 <= cp <= 0xDFFF:
        continue
    c = chr(cp)
    lo = c.lower()
    if lo != c:
        lm.append((cp, lo))
    if 0xAC00 <= cp <= 0xD7A3:
        continue   # Hangul: C++ phân rã bằng thuật toán chuẩn
    fo = fold1(c)
    if fo != c:
        fm.append((cp, fo))
emit_map("kb_lowmap", lm)
emit_map("kb_foldmap", fm)
