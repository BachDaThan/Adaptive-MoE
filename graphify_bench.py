#!/usr/bin/env python3
"""
graphify_bench.py - ĐO khoảng lệch giữa "tìm đúng" (Recall) và "trả lời đúng" (QA accuracy) của bộ nhớ dài hạn, và kiểm chứng graphify thu hẹp nó thế nào.

  python graphify_bench.py                       # chạy bộ dữ liệu tổng hợp, in bảng so sánh
  python graphify_bench.py --seed 3 --noise 800  # đổi dữ liệu / nhiều nhiễu hơn
  python graphify_bench.py --llm http://localhost:8080/v1 --model yui   # thêm người đọc LLM THẬT (OpenAI-compatible) cho cột "LLM acc"

BA HỆ THỐNG, cùng dữ liệu, cùng ngân sách ký tự cho prompt:
  bm25      recall BM25 thường (RAG.recall)
  dense     "đổ cả subgraph": hợp ứng viên BM25 + sự kiện chứa thực thể + lan truyền đồ thị, KHÔNG cắt tỉa/không thay cũ bằng mới, xếp theo điểm đồ thị  (đúng kiểu gây "ngợp ngữ cảnh")
  graphify  yui_graph.Graphify.recall (cắt tỉa, bỏ cũ, đa dạng, vừa ngân sách)
CHỈ SỐ
  recall    đáp án đúng CÓ MẶT trong ngữ cảnh (đo "tìm đúng")
  ctx       số ký tự ngữ cảnh trung bình (đo "độ đặc")
  stale     tỉ lệ câu hỏi mà ngữ cảnh chứa giá trị CŨ/NHIỄU cạnh đáp án đúng (đo "mâu thuẫn")
  acc       độ chính xác của NGƯỜI ĐỌC TRÍCH XUẤT (luật, cùng một người đọc cho cả ba hệ thống): chọn câu trùng ý nhất với câu hỏi rồi trích đáp án; hoà thì lấy câu đứng trước.
  gap       recall - acc (khoảng lệch)
  no-ans    với câu hỏi KHÔNG có đáp án: tỉ lệ từ chối đúng (không bịa)
TRUNG THỰC: dữ liệu và người đọc đều là TỔNG HỢP, nên số liệu chỉ cho thấy HƯỚNG và cơ chế (nhiễu/cũ/trùng làm hỏng người đọc), KHÔNG so sánh được với benchmark công bố của Graphify/Supermemory.
Người đọc luật không "hiểu" như LLM; dùng --llm để đo bằng model thật của bạn.
"""
import argparse
import collections
import math
import os
import random
import re
import shutil
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from yui_graph import QWORDS, Graphify, cap_phrases, entities, intent_of, num_value  # noqa: E402
from yui_rag import RAG, fold, tokens  # noqa: E402
from yui_vault import Workspace  # noqa: E402

CATS = ["Mướp", "Bơ", "Tèo", "Na", "Kem", "Sushi", "Đậu", "Mít", "Lu", "Bông", "Xíu", "Gạo"]
CITIES = ["Đà Lạt", "Huế", "Vũng Tàu", "Nha Trang", "Cần Thơ", "Hải Phòng", "Quy Nhơn", "Đà Nẵng", "Buôn Ma Thuột", "Phan Thiết", "Hà Giang", "Sa Pa"]
FRIENDS = ["Lan", "Minh", "Hùng", "Thảo", "Nam", "Vy", "Khang", "Trang", "Phúc", "Hạnh", "Tuấn", "Quỳnh"]
PETS = [("mèo", "nhà mik"), ("chó", "nhà mik"), ("cá", "trong bể của mik"), ("chim", "của mik"), ("thỏ", "của mik"), ("rùa", "của mik"), ("vẹt", "của mik"), ("hamster", "của mik")]
DEVICES = ["điện thoại", "laptop", "tai nghe", "đồng hồ", "máy tính bảng", "loa"]
NOISE = ["hôm nay trời đẹp nhỉ", "mik đang nghe nhạc", "kể chuyện cười đi", "mik buồn ngủ quá", "tối nay ăn gì ta", "mik vừa xem phim hay lắm", "dạo này công việc bận lắm",
         "mik thấy con mèo trên phố dễ thương", "điện thoại mik hay hết pin", "mik muốn đi chơi xa", "giá xăng tăng nữa rồi", "bạn có biết nhà nào đang cho thuê không",
         "mik hay quên tên người ta", "sinh nhật năm nay chắc mik ở nhà", "thời tiết hôm nay lạnh", "mik đang học nấu ăn", "con chó hàng xóm sủa hoài", "cuối tuần chưa biết làm gì"]
GEN_REPLY = ["ừ nghe hay đó", "mik hiểu rồi", "bạn kể thêm đi", "nghe vui quá", "ok bạn nhé"]


def norm(s):
    return re.sub(r"\s+", " ", fold(str(s).lower())).strip()


def build(seed=1, n_noise=500, reps=6):
    """reps <= 6: mỗi câu hỏi có chủ thể DUY NHẤT (mỗi thú cưng/thiết bị/bạn chỉ xuất hiện một lần) để đáp án không mơ hồ."""
    rng = random.Random(seed)
    t0, turns, qs = 1_760_000_000, [], []

    def say(day, user, bot=None, echo=True):
        turns.append((t0 + day * 86400 + rng.randint(0, 80000), user, bot or (f"mik nhớ rồi nha: {user}" if echo else rng.choice(GEN_REPLY))))
    pool = lambda xs, n: rng.sample(xs, n)  # noqa: E731
    for r in range(reps):
        pet, where = PETS[r % len(PETS)]
        a, b = pool(CATS, 2)
        say(rng.randint(1, 20), f"{pet} {where} tên là {a}")
        say(rng.randint(21, 40), f"{pet} của hàng xóm tên là {b}")
        qs.append(dict(q=f"{pet} {where} tên gì", type="name", gold=a, bad=[b]))
        dev = DEVICES[r % len(DEVICES)]; p1, p2 = rng.sample(range(2, 40), 2); fr = FRIENDS[r % len(FRIENDS)]
        say(rng.randint(1, 50), f"mik mới mua {dev} giá {p1} triệu")
        say(rng.randint(51, 90), f"bạn {fr} cũng mới mua {dev} giá {p2} triệu")
        qs.append(dict(q=f"{dev} mik mua giá bao nhiêu", type="number", gold=str(p1 * 10 ** 6), bad=[str(p2 * 10 ** 6)]))
        fr2 = FRIENDS[(r + 5) % len(FRIENDS)]; d1, m1, d2, m2 = rng.randint(1, 28), rng.randint(1, 12), rng.randint(1, 28), rng.randint(1, 12)
        say(rng.randint(1, 100), f"sinh nhật bạn {fr2} là ngày {d1}/{m1}")
        say(rng.randint(1, 100), f"sinh nhật mik là ngày {d2}/{m2}")
        qs.append(dict(q=f"sinh nhật bạn {fr2} là ngày nào", type="date", gold=f"{d1}/{m1}", bad=[f"{d2}/{m2}"]))
    for r in range(reps // 2):   # nơi ở thay đổi nhiều lần
        c = pool(CITIES, 3); days = sorted(rng.sample(range(1, 120), 3))
        who = FRIENDS[r]   # mỗi "người" một chuỗi nơi ở, để không lẫn nhau
        say(days[0], f"bạn {who} đang ở {c[0]}"); say(days[1], f"bạn {who} chuyển sang {c[1]} sống"); say(days[2], f"dạo này bạn {who} chuyển về {c[2]} rồi")
        qs.append(dict(q=f"bây giờ bạn {who} đang ở đâu", type="place", gold=c[2], bad=[c[0], c[1]]))
        qs.append(dict(q=f"lúc đầu bạn {who} ở đâu", type="place", gold=c[0], bad=[], history=True))
    for r in range(reps // 2):   # nhiều bước: bạn thân -> chó/mèo của bạn thân
        f1, f2 = pool(FRIENDS, 2); c1, c2 = pool(CATS, 2); noun = "chó" if r % 2 else "mèo"
        say(rng.randint(1, 60), f"{['bạn thân', 'người yêu', 'đồng nghiệp'][r % 3]} của mik là {f1}")
        say(rng.randint(61, 100), f"{f1} nuôi một con {noun} tên là {c1}")
        say(rng.randint(1, 100), f"{f2} nuôi một con {noun} tên là {c2}")
        qs.append(dict(q=f"{noun} của {['bạn thân', 'người yêu', 'đồng nghiệp'][r % 3]} mik tên gì", type="name", gold=c1, bad=[c2], hop=True))
    for r in range(reps // 3):   # liệt kê
        pl = pool(CITIES, 4)
        for i in range(3):
            say(rng.randint(1, 110), f"cuối tuần rồi mik đi {pl[i]} chơi")
        say(rng.randint(1, 110), f"mik muốn đi {pl[3]} nhưng chưa đi được")
        qs.append(dict(q="mik đã đi chơi những đâu", type="list", gold=sorted(pl[:3]), bad=[pl[3]], uniq=r))
        break   # một câu liệt kê duy nhất để không lẫn các lần
    for q in ("mik có xe máy hiệu gì", "mik học trường đại học nào", "số căn cước của mik là gì", "mik nuôi con cáo tên gì"):
        qs.append(dict(q=q, type="none", gold=None, bad=[]))
    for i in range(n_noise):
        say(rng.randint(1, 120), rng.choice(NOISE), echo=False)
    rng.shuffle(turns)
    return sorted(turns), qs


# ---------------- người đọc trích xuất (luật) ----------------
def content_tokens(q):
    t = {t for t in tokens(q, bigrams=False) if t.isascii()}
    return (t - QWORDS) or t


def extract(line, typ):
    body = line.split(": ", 1)[1] if ": " in line else line
    if typ == "name":
        m = re.search(r"tên (?:là )?(?:con )?(\w+)", body)
        if m and m.group(1)[:1].isupper():
            return m.group(1)
        caps = cap_phrases(body)
        return caps[-1][1] if caps and body.rstrip().endswith(caps[-1][1]) else None
    if typ == "number":
        m = re.search(r"\d[\d.,]*\s*(triệu|nghìn|k|đồng)", body)
        return num_value(m.group(0)) if m else None
    if typ == "date":
        m = re.search(r"\b(\d{1,2})/(\d{1,2})\b", body)
        return f"{int(m.group(1))}/{int(m.group(2))}" if m else None
    if typ == "place":
        m = re.search(r"(?:ở|sang|về|tới|đến|đi)\s+(\w)", body)
        if not m:
            return None
        caps = [p for pos, p in cap_phrases(body) if pos >= m.start(1)]
        return caps[0] if caps else None
    return None


def read(ctx, q, typ):
    lines = [ln for ln in ctx.split("\n") if ln.strip() and not ln.startswith("không tìm thấy")]
    if not lines:
        return None
    qt = content_tokens(q)
    if typ == "list":
        found = set()
        for ln in lines:
            if "chưa" in ln or "muốn" in ln:
                continue
            m = extract(ln, "place")
            if m:
                found.add(m)
        return sorted(found) or None
    scored = []
    for i, ln in enumerate(lines):
        lt = {t for t in tokens(ln, bigrams=False) if t.isascii()}
        ov = len(qt & lt) / max(1, len(qt))
        scored.append((ov, -i, ln))
    if typ == "name":   # người đọc biết "tên gì" cần câu có "tên" (áp dụng như nhau cho cả ba hệ thống)
        named = [x for x in scored if "tên" in x[2]]
        scored = named or scored
    scored.sort(reverse=True)
    if typ == "none":
        return scored[0][2] if scored[0][0] >= 0.6 else None       # trả lời bừa nếu có câu "na ná"
    for ov, _, ln in scored:
        if ov < 0.34:
            break
        a = extract(ln, typ)
        if a:
            return a
    return None


def correct(ans, qd):
    if qd["type"] == "none":
        return ans is None
    if ans is None:
        return False
    if qd["type"] == "list":
        return [norm(x) for x in ans] == [norm(x) for x in qd["gold"]]
    return norm(ans) == norm(qd["gold"])


def has(ctx, v):
    if isinstance(v, list):
        return all(has(ctx, x) for x in v)
    c = norm(ctx)
    return norm(v) in c or (str(v).isdigit() and len(str(v)) >= 6 and (str(int(v) // 10 ** 6) + " trieu") in c)


# ---------------- ba hệ thống ----------------
def sys_bm25(rag, g, q, chars):
    return rag.recall(q, chars=chars, k=4)


def sys_dense(rag, g, q, chars):
    """Đổ cả subgraph: ứng viên rộng (BM25 + thực thể + lan truyền) nhưng KHÔNG cắt tỉa, KHÔNG bỏ cũ; ngân sách gấp 3 (ngợp)."""
    g.sync()
    qt = {t for t in tokens(q, bigrams=False) if t.isascii()}
    qents = entities(q) + ["c:" + t for t in qt if ("c:" + t) in g.ent]
    cand = {}
    for h in rag.search(q, k=12):
        for fid in g.by_cid.get(h["cid"], ()):
            cand[fid] = cand.get(fid, 0) + 1.0
    seeds = {e: 1.0 for e in qents if e in g.ent}
    ppr = g._ppr(seeds)
    for e, w in sorted(ppr.items(), key=lambda kv: -kv[1])[:20]:
        for fid in g.ent.get(e, ())[-15:]:
            cand[fid] = cand.get(fid, 0) + w
    order = sorted(cand, key=cand.get, reverse=True)
    out, used = [], 0
    for fid in order:
        f = g.facts[fid]
        line = f"[{time.strftime('%d/%m', time.localtime(f['ts']))}] {f['who']}: {f['text']}"
        if used + len(line) > chars * 3:
            break
        out.append(line); used += len(line) + 1
    return "\n".join(out) or "không tìm thấy"


def sys_graphify(rag, g, q, chars):
    return g.recall(q, chars=chars)


SYSTEMS = {"bm25": sys_bm25, "dense": sys_dense, "graphify": sys_graphify}


def evaluate(seed=1, noise=500, chars=300, llm=None, quiet=False):
    d = tempfile.mkdtemp()
    try:
        ws = Workspace(d)
        turns, qs = build(seed, noise)
        import json
        with open(f"{d}/chat/dai-han.jsonl", "w", encoding="utf-8") as f:
            for ts, u, b in turns:
                f.write(json.dumps({"ts": ts, "messages": [{"role": "user", "content": u}, {"role": "assistant", "content": b}]}, ensure_ascii=False) + "\n")
        rag = RAG(ws); rag.sync(); g = Graphify(rag); g.sync()
        res = {}
        for name, fn in SYSTEMS.items():
            m = collections.defaultdict(list)
            for qd in qs:
                ctx = fn(rag, g, qd["q"], chars)
                ans = read(ctx, qd["q"], qd["type"])
                if qd["type"] == "none":
                    m["noans"].append(correct(ans, qd))
                    continue
                gold_in = has(ctx, qd["gold"])
                m["recall"].append(gold_in)
                m["acc"].append(correct(ans, qd))
                m["ctx"].append(len(ctx))
                m["stale"].append(gold_in and any(has(ctx, b) for b in qd["bad"]) if qd["bad"] else False)
                if llm:
                    m["llm"].append(llm(ctx, qd))
                m[f"acc_{qd['type']}"].append(correct(ans, qd))
                if qd.get("hop"):
                    m["acc_hop"].append(correct(ans, qd))
                    m["rec_hop"].append(gold_in)
                if qd.get("history"):
                    m["acc_hist"].append(correct(ans, qd))
            res[name] = {k: (sum(v) / len(v) if k != "ctx" else sum(v) / len(v)) for k, v in m.items()}
        res["_n"] = (len([q for q in qs if q["type"] != "none"]), len(turns))
        return res
    finally:
        shutil.rmtree(d, ignore_errors=True)


def table(res):
    n, turns = res["_n"]
    out = [f"{n} câu hỏi có đáp án + 4 câu không có đáp án, {turns} lượt chat (phần lớn là nhiễu).",
           f"{'hệ thống':<10}{'recall':>8}{'ctx(ký tự)':>12}{'stale':>8}{'acc':>8}{'gap':>8}{'no-ans':>8}{'hop acc':>9}{'history':>9}" + ("   LLM acc" if "llm" in res["bm25"] else "")]
    for k in SYSTEMS:
        r = res[k]
        out.append(f"{k:<10}{r['recall']*100:>7.0f}%{r['ctx']:>12.0f}{r['stale']*100:>7.0f}%{r['acc']*100:>7.0f}%{(r['recall']-r['acc'])*100:>7.0f}%{r['noans']*100:>7.0f}%{r.get('acc_hop', 0)*100:>8.0f}%{r.get('acc_hist', 0)*100:>8.0f}%"
                   + (f"   {r['llm']*100:.0f}%" if "llm" in r else ""))
    return "\n".join(out)


def make_llm(url, model):
    from yui_agent import OpenAICompat
    cli = OpenAICompat(url, model, None)

    def ask(ctx, qd):
        out = cli([{"role": "user", "content": f"Ngữ cảnh từ lịch sử chat:\n{ctx}\n\nCâu hỏi: {qd['q']}\nTrả lời thật ngắn; nếu ngữ cảnh không có thì nói \"không biết\"."}])
        gold = qd["gold"]
        if qd["type"] == "none":
            return "không biết" in norm(out)
        return has(out, gold) if not isinstance(gold, list) else all(norm(x) in norm(out) for x in gold)
    return ask


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--noise", type=int, default=500)
    ap.add_argument("--chars", type=int, default=300, help="ngân sách ký tự ngữ cảnh cho bm25/graphify (dense được gấp 3)")
    ap.add_argument("--seeds", type=int, default=1, help="chạy nhiều seed rồi lấy trung bình")
    ap.add_argument("--llm", default=None)
    ap.add_argument("--model", default="yui")
    a = ap.parse_args()
    llm = make_llm(a.llm, a.model) if a.llm else None
    runs = [evaluate(a.seed + i, a.noise, a.chars, llm) for i in range(a.seeds)]
    avg = {"_n": runs[0]["_n"]}
    for k in SYSTEMS:
        keys = set().union(*[r[k].keys() for r in runs])
        avg[k] = {x: sum(r[k].get(x, 0) for r in runs) / len(runs) for x in keys}
    print(table(avg))
