#!/usr/bin/env python3
"""Sinh bộ test vàng cho lõi C++ kb_core từ BẢN PYTHON (yui_rag.py + yui_graph.py): tests/golden/kb_golden.json.
Chạy từ thư mục gốc repo:   TZ=UTC PYTHONHASHSEED=0 python3 tests/golden/gen_golden.py
Gồm: (1) hàm đơn lẻ trên ~400 chuỗi (cố định + ngẫu nhiên có hạt giống), (2) các kịch bản kho: nạp file -> sync -> tìm BM25 / recall / Graphify, kể cả sửa/xoá file và nạp thêm.
Ghi chú: sum() của Python >= 3.12 dùng cộng bù; kb_core bắt chước đúng thuật toán đó. Python cũ hơn có thể lệch ở bit cuối (bộ so sánh có dung sai 1e-9).
"""
import json
import os
import random
import shutil
import sys
import tempfile
import time

os.environ["TZ"] = "UTC"
time.tzset()
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import graphify_bench as gb  # noqa: E402
import yui_graph as yg  # noqa: E402
from yui_graph import Graphify  # noqa: E402
from yui_rag import RAG, fold, split_text, tokens  # noqa: E402
from yui_vault import Workspace  # noqa: E402

# ----------------------------------------------------------------------------- 1. hàm đơn lẻ
FIXED = [
    "", " ", "Mèo", "mik nuôi chó", "cho mik hỏi", "Cho mik hỏi", "bạn Minh ở Huế", "mình ở Huế", "Đà Lạt đẹp lắm", "ĐƯỜNG Nguyễn Huệ", "đường", "Lan nuôi mèo tên là Xíu",
    "mik đi Đà Lạt chơi", "hôm nay mik ở nhà được nghỉ", "tiết kiệm 5 triệu", "sinh nhật 14/3", "sinh nhật 14 / 3 / 2025 nhé", "ngày 1/12/99 và 31/12/2026", "a12/3", "12/3b", "5kg gạo", "5 km", "3,5 triệu đồng",
    "giá 1.250.000 đồng", "mik 25 tuổi", "mik có 5 trong 10", "9" * 40, "1,2,3 nghìn", "5. hết", "100%", "5 % thôi", "12 TRIỆU", "bạn Lan đang ở Đà Lạt", "dạo này bạn Lan chuyển về Vũng Tàu rồi",
    "mèo nhà mik tên là A", "mèo của hàng xóm tên là B", "mèo nhà mik đổi tên thành B", "mik hay quên tên người ta", "số điện thoại của mik là 0912345678", "email của mik là a@b.com", "địa chỉ nhà mik ở Quận 1",
    "mik làm việc ở FPT", "món thích nhất của mik là phở", "lúc đầu mik ở đâu", "sinh nhật bạn Lan là ngày nào", "giá bao nhiêu", "ai là bạn thân của mik", "mik ở đâu", "liệt kê những nơi mik đã đi", "bây giờ bạn Lan đang ở đâu",
    "Hà Nội, Sài Gòn và Đà Nẵng", "Nguyễn Văn An gặp Trần Thị Bình", "Tôi tên là Hùng. Hôm nay Trời đẹp", "Ok Cảm ơn Chúc vui", "Hello World, This Is A Test", "ABC DEF", "Éa Ơi", "ǅ title", "İstanbul", "naïve café",
    "한국어 텍스트", "中文文本测试", "Ελληνικά κείμενο", "Привет мир", "e\u0301 a\u0300 o\u0302 (NFD)", "\u0301", "x" * 41 + " ngắn", "a. b! c? d… e; f, g: h",
    "Câu một. Câu hai! Câu ba? Câu bốn… Câu năm; câu sáu, câu bảy: câu tám.\nDòng mới\n\nĐoạn mới",
    "x" * 600, ("Một câu khá dài để thử. " * 30), ("từ " * 300), "a\n\n\nb\r\nc\td  e", "mik ở Đà Lạt\u00a0rất lâu", "A" * 180 + ", " + "b" * 50 + " " + "c" * 100, ("Mik đang kể một câu dài, có nhiều mệnh đề; và nhiều dấu: " * 6),
]
WORDS = ("mik minh mình bạn Lan Minh Hùng Thảo Đà Lạt Huế Vũng Tàu mèo chó cá nhà tên là gì ở đâu sống chuyển về sang đến tới đang dạo này hôm nay rồi nha nhé thích nhất ăn phở bún đi chơi cuối tuần "
         "sinh nhật ngày tháng năm tuổi giá triệu nghìn đồng k kg km tr trong phút giờ % email gmail sdt số điện thoại địa chỉ công việc làm nghề đổi gọi con của hàng xóm trước đây lúc đầu từng ai khi nào bao nhiêu "
         "những các tất cả liệt kê Mướp Bơ Xíu Đậu Sushi laptop điện thoại Nguyễn Trần Lê Phạm Hà Nội Sài Gòn Quận 1 2 3 7 14 25 100 2026").split()


def rand_strings(n, seed=7):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        parts = []
        for _ in range(rng.randint(1, 18)):
            w = rng.choice(WORDS)
            r = rng.random()
            if r < 0.08:
                w = w.upper()
            elif r < 0.14:
                w = w.lower()
            elif r < 0.2:
                import unicodedata
                w = "".join(c for c in unicodedata.normalize("NFD", w) if not unicodedata.combining(c))
            parts.append(w)
            if rng.random() < 0.2:
                parts.append(rng.choice([".", ",", ";", ":", "!", "?", "…", "\n", "  ", "/"]))
            if rng.random() < 0.1:
                parts.append(f"{rng.randint(1, 31)}/{rng.randint(1, 12)}" + (f"/{rng.randint(20, 2030)}" if rng.random() < 0.4 else ""))
            if rng.random() < 0.1:
                parts.append(f"{rng.choice(['3', '25', '5,5', '1.200.000', '70'])}{rng.choice(['', ' ', ' triệu', 'k', ' nghìn', ' tuổi', ' kg', ' tr', '%', ' đồng', ' phút'])}")
        out.append(" ".join(parts))
    return out


def unit_cases():
    texts = FIXED + rand_strings(350)
    cases = {"fold": [], "tokens": [], "split_text": [], "entities": [], "fact_key": [], "intent": [], "has_self": [], "split_facts": []}
    for t in texts:
        cases["fold"].append({"in": t, "out": fold(t)})
        for bg in (True, False):
            cases["tokens"].append({"in": t, "bigrams": bg, "out": tokens(t, bg)})
        for ml in (500, 40):
            cases["split_text"].append({"in": t, "maxlen": ml, "out": split_text(t, ml)})
        try:
            ent = yg.entities(t)
        except OverflowError:   # số quá lớn: bản Python ném lỗi, bản C++ bỏ qua thực thể đó (APP_KB.md §9)
            continue
        cases["entities"].append({"in": t, "out": ent})
        cases["fact_key"].append({"in": t, "out": yg.fact_key(t)})
        cases["intent"].append({"in": t, "out": yg.intent_of(t)})
        cases["has_self"].append({"in": t, "out": yg.has_self(t)})
        cases["split_facts"].append({"in": t, "out": yg.split_facts(t)})
    return cases


# ----------------------------------------------------------------------------- 2. kịch bản kho
def jl(rows):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows)


def write(ws, sub_rel, text, mtime):
    p = os.path.join(ws.root, sub_rel)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    os.utime(p, (mtime, mtime))


def dump_chunks(rag):
    return [{"cid": i, "src": c["s"], "conv": c["c"], "i": c["i"], "p": c["p"], "r": c["r"], "t": c["t"], "ts": c["ts"]} for i, c in enumerate(rag.chunks) if c is not None]


def dump_facts(g):
    g.sync()
    return [{"cid": f["cid"], "text": f["text"], "who": f["who"], "ts": f["ts"], "ents": f["ents"], "key": f["key"], "stale": f["stale"]} for f in g.facts if g.rag.chunks[f["cid"]] is not None]


def dump_hits(h):
    return [{"cid": x["cid"], "score": x["score"], "src": x["s"], "conv": x["c"], "i": x["i"], "p": x["p"], "r": x["r"], "t": x["t"], "ts": x["ts"]} for x in h]


def extra_queries(texts, n, seed):
    """Câu hỏi sinh ngẫu nhiên từ chính dữ liệu: vài từ của một lượt chat + từ để hỏi, để chạm các ngưỡng (từ chối, MMR, cầu nối, ý định)."""
    rng = random.Random(seed)
    ask = ["ở đâu", "tên gì", "bao nhiêu", "khi nào", "lúc đầu", "những", "bây giờ", "là ai", "giá", "ngày nào", "có không", "mik", "bạn"]
    out = []
    for _ in range(n):
        ws = rng.choice(texts).split()
        k = rng.randint(1, min(5, len(ws)))
        i = rng.randint(0, len(ws) - k)
        q = ws[i:i + k]
        if rng.random() < 0.6:
            q.insert(rng.randint(0, len(q)), rng.choice(ask))
        if rng.random() < 0.15:
            q.append(rng.choice(["xe máy", "căn cước", "con cáo", "trường đại học"]))
        out.append(" ".join(q))
    return out


def checks(rag, g, queries, extra=()):
    out = {"chunks": dump_chunks(rag), "facts": dump_facts(g), "search": [], "recall": [], "query": []}
    for q in queries:
        for k in (4, 8):
            out["search"].append({"q": q, "k": k, "role": None, "hits": dump_hits(rag.search(q, k))})
        out["search"].append({"q": q, "k": 4, "role": "user", "hits": dump_hits(rag.search(q, 4, role="user"))})
        for chars in (300, 90):
            out["recall"].append({"q": q, "chars": chars, "k": 4, "out": rag.recall(q, chars=chars, k=4)})
        for chars, k, ab in ((300, None, True), (150, 3, True), (300, None, False)):
            r = g.query(q, chars=chars, k=k, abstain=ab)
            out["query"].append({"q": q, "chars": chars, "k": k or 0, "abstain": ab, "context": r["context"], "facts": r["facts"], "conf": r["conf"], "intent": r["intent"]})
        out["query"].append({"q": q, "chars": 300, "k": 0, "mode": "recall_fallback", "context": g.recall(q, 300, None, fallback=True)})
    for q in extra:
        out["search"].append({"q": q, "k": 4, "role": None, "hits": dump_hits(rag.search(q, 4))})
        r = g.query(q, chars=300, k=None, abstain=True)
        out["query"].append({"q": q, "chars": 300, "k": 0, "abstain": True, "context": r["context"], "facts": r["facts"], "conf": r["conf"], "intent": r["intent"]})
    return out


def scenario_bench(seed, noise):
    turns, qs = gb.build(seed, noise)
    d = tempfile.mkdtemp()
    try:
        ws = Workspace(d)
        rows = [{"ts": ts, "messages": [{"role": "user", "content": u}, {"role": "assistant", "content": b}]} for ts, u, b in turns]
        text = jl(rows)
        write(ws, "chat/dai-han.jsonl", text, 1_760_000_000)
        rag = RAG(ws); rag.sync(); g = Graphify(rag)
        step = {"files": [{"sub": "chat", "rel": "chat/dai-han.jsonl", "mtime": 1_760_000_000, "text": text}], "remove": []}
        return {"name": f"bench-seed{seed}-noise{noise}", "steps": [dict(step, check=checks(rag, g, [q["q"] for q in qs]))]}
    finally:
        shutil.rmtree(d, ignore_errors=True)


NOTE1 = """# Mèo của mik

Mướp là con mèo nhà mik, nó 3 tuổi và rất thích ngủ trên ghế sofa mỗi buổi chiều khi trời nắng.

Bạn thân của mik là [[Lan]], Lan sống ở Đà Lạt. Sinh nhật Lan là ngày 14/3.

Ghi chú dài: """ + "Đây là một câu khá dài để kiểm tra việc cắt đoạn theo độ dài tối đa. " * 14 + """

Hết."""
NOTE2 = "Mik tiết kiệm được 5 triệu để mua laptop mới.\nGiá laptop khoảng 20 triệu.\n\nMik làm việc ở FPT từ năm 2024."
KB1 = "Tài liệu về Đà Lạt: thành phố ở Lâm Đồng, khí hậu mát mẻ quanh năm. Nhiều người ở Đà Lạt trồng hoa và dâu tây.\n\nHồ Xuân Hương nằm giữa trung tâm."
QS = ["mèo nhà mik tên gì", "bạn Lan ở đâu", "sinh nhật Lan là ngày nào", "laptop giá bao nhiêu", "mik làm việc ở đâu", "ten meo cua mik", "Đà Lạt có gì", "mik có con rùa tên gì", "bây giờ bạn Lan đang ở đâu",
      "lúc đầu bạn Lan ở đâu", "mik đã đi chơi những đâu", "dâu tây", "hồ Xuân Hương", "xe máy hiệu gì", "mik tiết kiệm bao nhiêu"]


def scenario_store():
    d = tempfile.mkdtemp()
    try:
        ws = Workspace(d)
        T = 1_760_000_000
        rows = [(1, "bạn Lan đang ở Đà Lạt", "mik nhớ rồi nha: bạn Lan đang ở Đà Lạt"), (30, "bạn Lan chuyển sang Huế sống", "ok nè"), (60, "dạo này bạn Lan chuyển về Vũng Tàu rồi", "mik nhớ rồi nha: dạo này bạn Lan chuyển về Vũng Tàu rồi"),
                (3, "mèo nhà mik tên là Mướp", "dễ thương quá"), (4, "mik đi Đà Lạt chơi cuối tuần", "vui nhỉ"), (5, "mik mới mua laptop giá 20 triệu", "xịn quá"),
                (6, "Cho mik hỏi: " + "một đoạn rất dài " * 40, "ừ " * 80)] + [(i, "hôm nay trời đẹp nhỉ", "ừ đẹp thật") for i in range(7, 25)]
        c1 = jl([{"ts": T + day * 86400, "messages": [{"role": "user", "content": u}, {"role": "assistant", "content": b}]} for day, u, b in rows])
        c2 = jl([{"ts": T + 40 * 86400, "messages": [{"role": "user", "content": "mik thích ăn phở nhất"}, {"role": "assistant", "content": "phở ngon thật"}]},
                 {"ts": T + 41 * 86400, "messages": [{"role": "user", "content": "địa chỉ nhà mik là 12 Lê Lợi"}, {"role": "assistant", "content": "ok"}]}])
        steps = []
        files = [("chat", "chat/a.jsonl", c1, T), ("chat", "chat/b.jsonl", c2, T + 5), ("notes", "notes/meo-cua-mik.md", NOTE1, T + 100), ("notes", "notes/Tiet Kiem.MD", NOTE2, T + 200), ("kb", "kb/_clean/da-lat.md", KB1, T + 300),
                 ("kb", "kb/_tomtat/da-lat.txt", "Đà Lạt mát mẻ, nhiều hoa.", T + 301)]
        for sub, rel, text, mt in files:
            write(ws, rel, text, mt)
        rag = RAG(ws); rag.sync(); g = Graphify(rag)
        steps.append({"files": [{"sub": s, "rel": r, "mtime": m, "text": t} for s, r, t, m in files], "remove": [], "check": checks(rag, g, QS, extra_queries([r[1] for r in rows if len(r[1].split()) > 3], 120, 5))})
        # bước 2: nạp thêm chat mới + sửa 1 ghi chú
        c3 = jl([{"ts": T + 90 * 86400, "messages": [{"role": "user", "content": "bạn Lan chuyển sang Hà Nội làm việc rồi"}, {"role": "assistant", "content": "ừ nhớ rồi"}]},
                 {"ts": T + 91 * 86400, "messages": [{"role": "user", "content": "mèo nhà mik đổi tên thành Bơ"}, {"role": "assistant", "content": "tên mới dễ thương"}]}])
        note1b = NOTE1 + "\n\nMướp vừa sinh 4 mèo con."
        for rel, text, mt in (("chat/c.jsonl", c3, T + 92 * 86400), ("notes/meo-cua-mik.md", note1b, T + 400)):
            write(ws, rel, text, mt)
        rag.sync()
        steps.append({"files": [{"sub": "chat", "rel": "chat/c.jsonl", "mtime": T + 92 * 86400, "text": c3}, {"sub": "notes", "rel": "notes/meo-cua-mik.md", "mtime": T + 400, "text": note1b}], "remove": [],
                      "check": checks(rag, g, QS)})
        # bước 3: xoá một file
        os.unlink(os.path.join(ws.root, "chat/b.jsonl"))
        rag.sync()
        steps.append({"files": [], "remove": ["chat/b.jsonl"], "check": checks(rag, g, QS)})
        return {"name": "store-mixed", "steps": steps}
    finally:
        shutil.rmtree(d, ignore_errors=True)


def main():
    out = {"python": sys.version.split()[0], "rag_version": __import__("yui_rag").VERSION, "units": unit_cases(), "scenarios": [scenario_store(), scenario_bench(1, 500), scenario_bench(3, 150)]}
    path = os.path.join(ROOT, "tests", "golden", "kb_golden.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, separators=(",", ":"))
    print("ghi", path, os.path.getsize(path), "byte")


if __name__ == "__main__":
    main()
