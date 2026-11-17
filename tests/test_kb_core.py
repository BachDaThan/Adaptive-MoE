"""Test lõi C++ kb_core so với bộ vàng sinh từ Python (tests/golden/kb_golden.json). Chạy:  python3 -m pytest -q tests/test_kb_core.py
Tự biên dịch native/kb_core.cpp thành thư viện tạm (cần g++/clang++). Sinh lại bộ vàng: TZ=UTC PYTHONHASHSEED=0 python3 tests/golden/gen_golden.py
"""
import ctypes
import json
import os
import shutil
import subprocess

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
GOLD = json.load(open(os.path.join(HERE, "golden", "kb_golden.json"), encoding="utf-8"))
TOL = 1e-9


@pytest.fixture(scope="session")
def lib(tmp_path_factory):
    cxx = os.environ.get("CXX") or shutil.which("g++") or shutil.which("clang++")
    if not cxx:
        pytest.skip("không có trình biên dịch C++")
    if os.environ.get("KB_LIB"):   # dùng thư viện dựng sẵn (vd bản sanitizer)
        out = os.environ["KB_LIB"]
    else:
        out = str(tmp_path_factory.mktemp("kb") / "libkb_core.so")
    src = os.path.join(ROOT, "native", "kb_core.cpp")
    if not os.environ.get("KB_LIB"):
        subprocess.run([cxx, "-O2", "-std=c++17", "-ffp-contract=off", "-fPIC", "-shared", "-Wall", "-Wextra", src, "-o", out], check=True)
    L = ctypes.CDLL(out)
    L.kb_new.restype = ctypes.c_void_p
    for n in ("kb_free", "kb_commit", "kb_clear"):
        getattr(L, n).argtypes = [ctypes.c_void_p]
    L.kb_set_tz.argtypes = [ctypes.c_void_p, ctypes.c_int]
    L.kb_add_chat_turn.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int64]
    L.kb_add_doc.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int64]
    L.kb_remember.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int64]
    L.kb_drop_source.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
    L.kb_recall.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    L.kb_dbg_fold.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    L.kb_dbg_tokens.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    L.kb_dbg_split_text.argtypes = [ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    for n in ("kb_dbg_entities", "kb_dbg_fact_key", "kb_dbg_intent", "kb_dbg_split_facts"):
        getattr(L, n).argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    L.kb_dbg_has_self.argtypes = [ctypes.c_char_p]
    L.kb_dbg_search_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    L.kb_dbg_query_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    L.kb_dbg_chunks_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    L.kb_dbg_facts_json.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    return L


def s(fn, *args):
    """Gọi hàm ghi chuỗi vào bộ đệm (tự nới bộ đệm) -> str."""
    cap = 1 << 14
    while True:
        buf = ctypes.create_string_buffer(cap)
        n = fn(*args, buf, cap)
        if n < cap:
            return buf.value.decode("utf-8")
        cap = n + 1


def b(x):
    return None if x is None else x.encode("utf-8")


def near(a, c):
    return abs(a - c) <= TOL * max(1.0, abs(a), abs(c))


def first_diff(cases, got_fn, label):
    bad = []
    for c in cases:
        got = got_fn(c)
        if got != c["out"]:
            bad.append((c["in"][:80], c["out"], got))
            if len(bad) >= 5:
                break
    assert not bad, f"{label}: lệch (đầu vào, python, c++): {bad}"


# ------------------------------------------------------------------ hàm đơn lẻ
def test_fold(lib):
    first_diff(GOLD["units"]["fold"], lambda c: s(lib.kb_dbg_fold, b(c["in"])), "fold")


def test_tokens(lib):
    first_diff(GOLD["units"]["tokens"], lambda c: json.loads(s(lib.kb_dbg_tokens, b(c["in"]), int(c["bigrams"]))), "tokens")


def test_split_text(lib):
    first_diff(GOLD["units"]["split_text"], lambda c: json.loads(s(lib.kb_dbg_split_text, b(c["in"]), c["maxlen"])), "split_text")


def test_entities(lib):
    first_diff(GOLD["units"]["entities"], lambda c: json.loads(s(lib.kb_dbg_entities, b(c["in"]))), "entities")


def test_fact_key(lib):
    first_diff(GOLD["units"]["fact_key"], lambda c: json.loads(s(lib.kb_dbg_fact_key, b(c["in"]))), "fact_key")


def test_intent(lib):
    first_diff(GOLD["units"]["intent"], lambda c: s(lib.kb_dbg_intent, b(c["in"])), "intent")


def test_has_self(lib):
    first_diff(GOLD["units"]["has_self"], lambda c: bool(lib.kb_dbg_has_self(b(c["in"]))), "has_self")


def test_split_facts(lib):
    first_diff(GOLD["units"]["split_facts"], lambda c: json.loads(s(lib.kb_dbg_split_facts, b(c["in"]))), "split_facts")


# ------------------------------------------------------------------ kịch bản kho
def replay(lib, kb, step):
    order = {"chat": 0, "notes": 1, "kb": 2}   # RAG.sync quét chat/, notes/, kb/ và sắp theo đường dẫn trong từng thư mục
    for f in sorted(step["files"], key=lambda f: (order[f["sub"]], f["rel"])):
        rel = f["rel"].encode()
        lib.kb_drop_source(kb, rel)
        if f["sub"] == "chat":
            for k, line in enumerate(l for l in f["text"].splitlines() if l.strip()):
                o = json.loads(line)
                for m in o["messages"]:
                    assert lib.kb_add_chat_turn(kb, rel, f"{f['rel']}#{k}".encode(), m["role"].encode(), m["content"].encode(), o["ts"]) >= 0
        else:
            assert lib.kb_add_doc(kb, b"note" if f["sub"] == "notes" else b"kb", rel, f["text"].encode(), f["mtime"]) >= 0
    for rel in step["remove"]:
        lib.kb_drop_source(kb, rel.encode())
    lib.kb_commit(kb)


def cmp_list(got, exp, key, label, floats=()):
    assert len(got) == len(exp), f"{label}: số phần tử {len(got)} != {len(exp)}"
    for i, (g, e) in enumerate(zip(got, exp)):
        for k in e:
            if k in floats:
                assert near(g[k], e[k]), f"{label}[{i}].{k}: c++ {g[k]} != python {e[k]}  ({key})"
            else:
                assert g[k] == e[k], f"{label}[{i}].{k}: c++ {g[k]!r} != python {e[k]!r}  ({key})"


@pytest.mark.parametrize("scn", GOLD["scenarios"], ids=[x["name"] for x in GOLD["scenarios"]])
def test_scenario(lib, scn):
    kb = lib.kb_new(b"Yui")
    lib.kb_set_tz(kb, 0)
    try:
        for si, step in enumerate(scn["steps"]):
            replay(lib, kb, step)
            ck = step["check"]
            cmp_list(json.loads(s(lib.kb_dbg_chunks_json, kb)), ck["chunks"], "", f"{scn['name']}#{si} chunks")
            cmp_list(json.loads(s(lib.kb_dbg_facts_json, kb)), ck["facts"], "", f"{scn['name']}#{si} facts")
            for x in ck["search"]:
                got = json.loads(s(lib.kb_dbg_search_json, kb, b(x["q"]), x["k"], b(x["role"])))
                cmp_list(got, x["hits"], x["q"], f"{scn['name']}#{si} search", floats=("score",))
            for x in ck["recall"]:
                got = s(lib.kb_recall, kb, b(x["q"]), x["chars"], x["k"], 0)
                assert got == x["out"], f"recall bm25 {x['q']!r}: c++ {got!r} != python {x['out']!r}"
            for x in ck["query"]:
                if x.get("mode") == "recall_fallback":
                    got = s(lib.kb_recall, kb, b(x["q"]), x["chars"], 0, 2)
                    assert got == x["context"], f"recall fallback {x['q']!r}: c++ {got!r} != python {x['context']!r}"
                    continue
                r = json.loads(s(lib.kb_dbg_query_json, kb, b(x["q"]), x["chars"], x["k"], int(x["abstain"])))
                tag = f"{scn['name']}#{si} query {x['q']!r} chars={x['chars']} k={x['k']} abstain={x['abstain']}"
                assert r["intent"] == x["intent"], tag
                assert r["context"] == x["context"], f"{tag}: c++ {r['context']!r} != python {x['context']!r}"
                assert r["facts"] == x["facts"], tag
                assert near(r["conf"], x["conf"]), f"{tag}: conf {r['conf']} != {x['conf']}"
    finally:
        lib.kb_free(kb)


def test_remember_equals_add_then_commit(lib):
    def build(use_remember):
        kb = lib.kb_new(b"Yui")
        if use_remember:
            assert lib.kb_remember(kb, b"chat/x.jsonl", b"chat/x.jsonl#0", "  mik nuôi mèo tên là Mướp  ".encode(), "dễ thương quá".encode(), 1_760_000_000) == 1
        else:
            lib.kb_add_chat_turn(kb, b"chat/x.jsonl", b"chat/x.jsonl#0", b"user", "mik nuôi mèo tên là Mướp".encode(), 1_760_000_000)
            lib.kb_add_chat_turn(kb, b"chat/x.jsonl", b"chat/x.jsonl#0", b"assistant", "dễ thương quá".encode(), 1_760_000_000)
            lib.kb_commit(kb)
        out = s(lib.kb_recall, kb, "mèo tên gì".encode(), 200, 0, 1)
        lib.kb_free(kb)
        return out
    assert build(True) == build(False) and "Mướp" in build(True)
    kb = lib.kb_new(None)
    assert lib.kb_remember(kb, b"a", b"a#0", b"  ", b"x", 1) == 0
    lib.kb_free(kb)


def test_drop_source_removes_from_results(lib):
    kb = lib.kb_new(b"Yui")
    lib.kb_add_doc(kb, b"note", b"notes/a.md", "Mướp là con mèo nhà mik".encode(), 1_760_000_000)
    lib.kb_commit(kb)
    assert "Mướp" in s(lib.kb_recall, kb, "mèo".encode(), 200, 0, 0)
    assert lib.kb_drop_source(kb, b"notes/a.md") == 1
    assert s(lib.kb_recall, kb, "mèo".encode(), 200, 0, 0) == "không tìm thấy trong lịch sử"
    lib.kb_free(kb)
