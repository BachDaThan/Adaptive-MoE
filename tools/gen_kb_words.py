#!/usr/bin/env python3
"""Sinh native/kb_words.inc: các danh sách/mẫu của yui_rag.py + yui_graph.py cho kb_core (một nguồn sự thật, khỏi chép tay).
Chạy từ thư mục gốc repo:  python3 tools/gen_kb_words.py > native/kb_words.inc
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yui_graph as g  # noqa: E402
import yui_rag as r  # noqa: E402


def cstr(s):
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def words(name, items):
    items = sorted(items)
    print(f"static const char* const {name}[] = {{")
    for i in range(0, len(items), 8):
        print("  " + ", ".join(cstr(x) for x in items[i:i + 8]) + ",")
    print("};")
    print(f"static const size_t {name}_N = sizeof({name}) / sizeof({name}[0]);")


def rawstr(s):
    assert ")KBX\"" not in s
    return 'R"KBX(' + s + ')KBX"'


print("// TỰ SINH bởi tools/gen_kb_words.py từ yui_rag.py + yui_graph.py. KHÔNG sửa tay.")
print(f"static const int KB_RAG_VERSION = {r.VERSION};")
words("KB_STOP", r.STOP)
words("KB_STOP_ACC", r.STOP_ACC)
words("KB_COMMON_CAP", g.COMMON_CAP)
words("KB_SELF_FORMS", g.SELF_FORMS)
words("KB_QWORDS", g.QWORDS)
words("KB_TIME_WORDS", g.TIME_WORDS)
print("struct KbPat { const char* name; const char* rx; };")
print("static const KbPat KB_DURABLE[] = {")
for n, rx in g.DURABLE:
    print(f"  {{{cstr(n)}, {rawstr(rx)}}},")
print("};")
print("static const size_t KB_DURABLE_N = sizeof(KB_DURABLE) / sizeof(KB_DURABLE[0]);")
print("static const KbPat KB_INTENT[] = {")
for n, rx in g.INTENT:
    print(f"  {{{cstr(n)}, {rawstr(rx)}}},")
print("};")
print("static const size_t KB_INTENT_N = sizeof(KB_INTENT) / sizeof(KB_INTENT[0]);")
print("struct KbKi { const char* name; int k; };")
print("static const KbKi KB_K_BY_INTENT[] = {")
for n, k in g.K_BY_INTENT.items():
    print(f"  {{{cstr(n)}, {k}}},")
print("};")
print("static const size_t KB_K_BY_INTENT_N = sizeof(KB_K_BY_INTENT) / sizeof(KB_K_BY_INTENT[0]);")
print("struct KbMult { const char* unit; long long mul; };")
print("static const KbMult KB_MULT[] = {")
for n, m in g._MULT.items():
    print(f"  {{{cstr(n)}, {m}LL}},")
print("};")
print("static const size_t KB_MULT_N = sizeof(KB_MULT) / sizeof(KB_MULT[0]);")
