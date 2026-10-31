#!/usr/bin/env python3
"""
yui_graph.py - GRAPHIFY cho Yui: đồ thị tri thức mức SỰ KIỆN (fact) dựng trên lịch sử chat/ghi chú/tài liệu của RAG, chỉ thư viện chuẩn, hiểu tiếng Việt (kể cả không dấu).

VẤN ĐỀ CẦN GIẢI (khoảng lệch Recall ↔ độ chính xác trả lời): đồ thị giúp TÌM ĐÚNG nhiều hơn (recall cao), nhưng nếu đổ cả subgraph (nhiều nút, quan hệ, sự kiện cũ/mới lẫn lộn)
vào prompt thì model nhỏ bị "ngợp ngữ cảnh": nhiễu, mâu thuẫn, không biết câu nào đáng tin -> trả lời kém dù thông tin có mặt.
GIẢI PHÁP: TÁCH HAI VIỆC. Đồ thị dùng để MỞ RỘNG ỨNG VIÊN (recall); sau đó một tầng "người đọc thân thiện" nén lại thành vài câu thường, sạch:
  1. Ứng viên: BM25 trên đoạn + sự kiện chứa thực thể trong câu hỏi + lan truyền 2 bước trên đồ thị đồng xuất hiện (PageRank cá nhân hoá rút gọn) -> kéo được cả "bạn thân" -> "Lan" -> "mèo của Lan" (nhiều bước).
  2. Chấm điểm theo CẤP SỰ KIỆN (không phải cả đoạn): từ khoá, thực thể hiếm, điểm đồ thị, BM25 đoạn, khớp LOẠI câu hỏi (khi nào -> cần ngày; bao nhiêu -> cần số; ai/tên gì -> cần tên riêng), độ mới, vai (lời người dùng > lời Yui nhại lại).
  3. THAY THẾ SỰ KIỆN CŨ: thuộc tính bền (tên, tuổi, nơi ở, số điện thoại, email, địa chỉ, nghề...) bị bản mới hơn thay thế -> bản cũ bị đánh "cũ" và KHÔNG đưa vào (trừ khi hỏi "trước đây/lúc đầu/từng").
  4. CẮT TỈA: MMR (đa dạng, bỏ gần trùng), dừng khi đã phủ đủ ý câu hỏi (không cố nhồi cho đủ k), tối đa K câu theo loại câu hỏi, cắt mỗi câu quanh phần liên quan nhất, vừa khít ngân sách ký tự.
  5. KỂ LẠI bằng câu thường theo thứ tự thời gian: "[12/09] bạn: ..." (đúng định dạng recall cũ nên model đã học vẫn dùng được). KHÔNG có tam thể, mã nút, thuật ngữ đồ thị.
  6. TỰ TIN / TỪ CHỐI: tính độ phủ ý câu hỏi; quá thấp thì nói "không tìm thấy" (hoặc lùi về BM25 thường) thay vì đưa nhiễu để model bịa.
Kỳ vọng: recall gần bằng "đổ cả subgraph" nhưng ngữ cảnh ngắn hơn nhiều, sạch mâu thuẫn -> model nhỏ dễ chuyển thành câu trả lời hơn. Xem graphify_bench.py để đo (có chỗ cắm LLM thật).
Giới hạn trung thực: trích sự kiện/quan hệ bằng LUẬT (không dùng LLM) nên bỏ sót cách nói lạ; "thay thế cũ bằng mới" chỉ áp cho nhóm thuộc tính bền trong từ điển, không đoán bừa.
"""
import collections
import json
import math
import re
import threading
import time
import zlib

from yui_rag import _WORD, STOP, fold, tokens
from yui_vault import VaultError

VERSION = 2
SELF = {"mik", "minh", "toi", "to", "em", "tui"}
_LETTERS = re.compile(r"[^\W\d_]+", re.U)


def cap_phrases(text):
    """[(vị trí đầu, cụm)] các cụm từ VIẾT HOA chữ đầu liền nhau (tên riêng). Dùng str.isupper() chứ không dùng dải ký tự: dải À-Ỵ lẫn cả chữ thường có dấu (đ, ở, được...)."""
    out, cur, last_end = [], [], None
    for m in _LETTERS.finditer(text):
        w = m.group()
        if w[:1].isupper() and not w.isupper():
            if cur and text[last_end:m.start()].strip() == "":
                cur.append(w)
            else:
                if cur:
                    out.append((start, " ".join(cur)))
                cur, start = [w], m.start()
            last_end = m.end()
        else:
            if cur:
                out.append((start, " ".join(cur)))
            cur = []
    if cur:
        out.append((start, " ".join(cur)))
    return out

_NUM = re.compile(r"(\d[\d.,]*)\s*(triệu|trieu|tr|nghìn|nghin|ngàn|ngan|k|đồng|dong|vnđ|vnd|tuổi|tuoi|phút|phut|giờ|gio|ngày|ngay|tháng|thang|năm|nam|%|kg|km)?", re.I)
_DATE = re.compile(r"\b(\d{1,2})\s*/\s*(\d{1,2})(?:\s*/\s*(\d{2,4}))?\b")
_MULT = {"triệu": 10 ** 6, "trieu": 10 ** 6, "tr": 10 ** 6, "nghìn": 1000, "nghin": 1000, "ngàn": 1000, "ngan": 1000, "k": 1000}
COMMON_CAP = {"Mik", "Mình", "Tôi", "Tớ", "Hôm", "Tháng", "Thứ", "Ngày", "Năm", "Sáng", "Chiều", "Tối", "Trưa", "Nhưng", "Vì", "Nếu", "Khi", "Và", "Thì", "Cũng", "Rồi", "Ừ", "Ok", "Cảm", "Chúc"}
# thuộc tính BỀN: (tên thuộc tính, regex trên chữ không dấu). Khoá = thuộc tính|chủ thể. Bản mới thay bản cũ.
DURABLE = [("ten", r"\b(ten la|doi ten|goi la|ten (con|cua))\b"), ("tuoi", r"\b(\d+\s*tuoi|tuoi)\b"), ("noi o", r"\b(song o|dang o|o tai|chuyen (den|ve|sang|toi)|don (den|sang|ve))\b"),
           ("so dien thoai", r"\bso dien thoai\b|\bsdt\b"), ("email", r"\bemail\b|\bgmail\b"), ("dia chi", r"\bdia chi\b"), ("cong viec", r"\b(lam nghe|lam viec (o|tai)|cong viec)\b"),
           ("thich nhat", r"\bthich nhat\b|\byeu thich nhat\b")]
SUBJECT_NOUNS = ["ban than", "ban gai", "ban trai", "meo", "cho", "me", "ba", "bo", "anh", "chi", "em", "con", "vo", "chong", "sep", "thay", "co giao", "dong nghiep", "ban"]
INTENT = [("history", r"\b(truoc day|luc dau|ban dau|tung|truoc kia|ngay xua|cu|hoi do)\b"), ("when", r"\b(khi nao|ngay nao|luc nao|hom nao|thang may|nam nao|bao gio|ngay may)\b"),
          ("howmany", r"\b(bao nhieu|may (nguoi|cai|lan|con|mon|nam|tuoi)|so luong|het bao nhieu|gia)\b"), ("who", r"\b(ai|ten gi|ten la gi|la ai)\b"),
          ("where", r"\b(o dau|noi nao|di dau|tai dau)\b"), ("list", r"\b(nhung|liet ke|tat ca|co nhung|cac)\b")]
# từ để HỎI (không mang nội dung cần tìm): không tính vào độ phủ/điểm từ khoá, tránh từ chối oan "bây giờ ... ở đâu"
QWORDS = set(("bay gio dau nao gi bao nhieu ai khi luc hom may thang nam ngay truoc day tien hien tai la nhung cac tat ca liet ke biet noi nhac nho tung con gia sao "
              "phai khong the nhu tai dai ban dau xua cu moi").split())   # (không có "cho": "chó" bỏ dấu là "cho")
K_BY_INTENT = {"list": 6, "history": 4, "when": 3, "howmany": 3, "who": 3, "where": 3, "other": 4}


_PLACEY_W = re.compile(r"(?:ở|sang|về|tới|đến|chuyển)\s+(\w)")


class _Placey:
    @staticmethod
    def search(text):
        return any(m.group(1).isupper() for m in _PLACEY_W.finditer(text))


_PLACEY = _Placey()


SELF_FORMS = {"mik", "mình", "minh", "tôi", "toi", "tui", "tớ"}


def has_self(text):
    """Có đại từ tự xưng (mik/mình/tôi...) viết THƯỜNG không. 'Minh' viết hoa là tên người, không phải 'mình'."""
    return any(w in SELF_FORMS for w in _WORD.findall(text) if not w[:1].isupper())


def intent_of(q):
    f = fold(q.lower())
    return next((n for n, rx in INTENT if re.search(rx, f)), "other")


def num_value(txt):
    m = _NUM.search(txt)
    if not m:
        return None
    try:
        v = float(m.group(1).replace(".", "").replace(",", "."))
    except ValueError:
        return None
    return str(int(v * _MULT.get((m.group(2) or "").lower(), 1)))


def entities(text):
    """-> list thực thể: 'p:da lat' (tên riêng), 'n:5000000' (số/tiền), 'd:14/3' (ngày). Thực thể khái niệm (từ khoá) thêm ở tầng chỉ mục."""
    out = []
    for pos, w in cap_phrases(text):
        first = pos == 0 or text[:pos].rstrip().endswith((".", "!", "?", ":", "\n"))
        if w in COMMON_CAP or (first and " " not in w and len(text) > 90):
            continue   # từ đầu câu viết hoa trong câu DÀI là chữ thường bị viết hoa (không phải tên); câu ngắn kiểu chat ("Lan nuôi mèo...") thì giữ
        parts = [p for p in w.split() if p not in COMMON_CAP]
        if parts:
            out.append("p:" + fold(" ".join(parts).lower()))
    for m in _DATE.finditer(text):
        out.append(f"d:{int(m.group(1))}/{int(m.group(2))}")
    for m in _NUM.finditer(text):
        v = num_value(m.group(0))
        if v and not _DATE.fullmatch(m.group(0).strip()) and (len(v) >= 2 or m.group(2)):
            out.append("n:" + v)
    return list(dict.fromkeys(out))


TIME_WORDS = {"dao", "nay", "hom", "gio", "bay", "vua", "moi", "luc", "dau", "roi", "cung", "van", "con", "sau", "truoc", "do", "nua", "thi", "ma"}


def fact_key(text):
    """Khoá thay thế cho thuộc tính bền, dạng 'thuộc tính|chủ thể' (vd 'ten|meo', 'noi o|ban lan', 'noi o|user').
    Chủ thể = các từ nội dung đứng TRƯỚC cụm thuộc tính (bỏ đại từ/từ chỉ thời gian), nên 'mèo nhà mik tên là A' và 'mèo của hàng xóm tên là B' KHÔNG thay thế nhau.
    None nếu không phải thuộc tính bền."""
    f = fold(text.lower())
    for attr, rx in DURABLE:
        m = re.search(rx, f)
        if m:
            subj = sorted({t for t in tokens(f[:m.start()], bigrams=False) if t.isascii() and t not in TIME_WORDS})
            return f"{attr}|{' '.join(subj[:4]) or 'user'}"
    return None


def split_facts(text, maxlen=170):
    text = " ".join(text.split())
    if len(text) <= maxlen:
        return [text] if len(text) >= 8 else []
    parts = [s.strip() for s in re.split(r"(?<=[.!?…;])\s+", text) if len(s.strip()) >= 8]
    out = []
    for s in parts:
        while len(s) > maxlen:
            cut = max(s.rfind(",", 0, maxlen), s.rfind(" ", 0, maxlen))
            out.append(s[:cut if cut > 40 else maxlen].strip()); s = s[(cut if cut > 40 else maxlen):].strip(" ,")
        if s:
            out.append(s)
    return out


class Graphify:
    def __init__(self, rag):
        self.rag, self.lock = rag, threading.RLock()
        self._reset()
        self.load()

    def _reset(self):
        self.facts = []            # {cid, text, who, ts, ents, key, stale, conv, pos}
        self.by_cid = {}           # cid -> [fid]
        self.ent = collections.defaultdict(list)    # thực thể -> [fid]
        self.co = collections.defaultdict(dict)     # thực thể -> {thực thể: trọng số}
        self.by_key = {}           # khoá thuộc tính bền -> [fid] theo thời gian
        self.mark = 0              # đã xử lý các chunk có cid < mark
        self.dirty = False
        self._saved = 0.0

    # ---------------- lưu / nạp ----------------
    def save(self):
        with self.lock:
            import os
            blob = zlib.compress(json.dumps({"v": VERSION, "mark": self.mark, "facts": [[f["cid"], f["text"], f["who"], f["ts"], f["ents"], f["key"], f["conv"], f["pos"]] for f in self.facts]},
                                            ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 6)
            for stale in ("graphify.bin", "graphify.bin.enc"):
                p = os.path.join(self.rag.ws.root, "index", stale)
                if os.path.exists(p):
                    os.unlink(p)
            self.rag.ws.write_bytes("graphify.bin", blob, sub="index")
            self.dirty = False

    def load(self):
        p = self.rag.ws.find("graphify.bin", "index")
        if not p:
            return
        try:
            d = json.loads(zlib.decompress(self.rag.ws.read_bytes(p)).decode("utf-8"))
            if d.get("v") != VERSION:
                raise ValueError("phiên bản cũ")
            rows = d["facts"]
            chunks = self.rag.chunks
            if d["mark"] > len(chunks):
                raise ValueError("chỉ mục RAG đã dựng lại")
            for cid, text, *_ in rows[:: max(1, len(rows) // 60)]:   # lấy mẫu: sự kiện phải khớp đúng đoạn RAG (chỉ mục RAG có thể đã dựng lại với mã đoạn khác)
                ch = chunks[cid] if cid < len(chunks) else None
                if ch is not None and text[:30] not in " ".join(ch["t"].split()):
                    raise ValueError("không khớp chỉ mục RAG")
            self._reset()
            self.mark = d["mark"]
            for cid, text, who, ts, ents, key, conv, pos in rows:
                self._add(cid, text, who, ts, ents, key, conv, pos)
            self._supersede()
        except (VaultError, OSError, ValueError, KeyError, IndexError, zlib.error) as e:
            self.rag.ws.log_error("graph.load", f"đồ thị hỏng, dựng lại: {e}")
            self._reset()

    # ---------------- dựng ----------------
    def _add(self, cid, text, who, ts, ents, key, conv, pos):
        fid = len(self.facts)
        self.facts.append({"cid": cid, "text": text, "who": who, "ts": ts, "ents": ents, "key": key, "stale": False, "conv": conv, "pos": pos})
        self.by_cid.setdefault(cid, []).append(fid)
        for e in ents:
            self.ent[e].append(fid)
        for i, a in enumerate(ents):
            for b in ents[i + 1:]:
                self.co[a][b] = self.co[a].get(b, 0) + 1
                self.co[b][a] = self.co[b].get(a, 0) + 1
        return fid

    def sync(self):
        """Trích sự kiện cho các đoạn mới của RAG (chưa xử lý). Gọi rẻ: chỉ làm phần mới."""
        with self.lock:
            rag, n_new = self.rag, 0
            for cid in range(self.mark, len(rag.chunks)):
                ch = rag.chunks[cid]
                if ch is None or self._is_echo(cid, ch):
                    continue
                for s in split_facts(ch["t"].split(": ", 1)[1] if ch["r"] in ("note", "kb") and ": " in ch["t"][:80] else ch["t"]):
                    ents = entities(s)
                    # thực thể khái niệm: từ khoá hiếm (idf cao) trong câu, dạng không dấu
                    toks = [t for t in tokens(s, bigrams=False) if t.isascii() and len(t) >= 3 and not t.isdigit()]
                    ents += ["c:" + t for t in sorted(set(toks), key=lambda t: len(rag.inv.get(t, ())))[:5]]
                    self._add(cid, s, ch["r"], ch["ts"], list(dict.fromkeys(ents)), fact_key(s) if ch["r"] in ("user", "note", "kb") else None, ch["c"], ch["p"])
                    n_new += 1
            self.mark = len(rag.chunks)
            if n_new:
                self._supersede()
                self.dirty = True
            if self.dirty and time.time() - self._saved > 30:
                self.save()
                self._saved = time.time()
            if len(self.facts) > 400 and sum(1 for f in self.facts if rag.chunks[f["cid"]] is None) > len(self.facts) * 0.35:
                self._compact()
            for e in [e for e, nb in self.co.items() if len(nb) > 60]:
                self.co[e] = dict(sorted(self.co[e].items(), key=lambda kv: -kv[1])[:30])
            return n_new

    def _is_echo(self, cid, ch):
        """Lời Yui lặp lại gần nguyên câu người dùng vừa nói ("mik nhớ rồi nha: ...") không mang thông tin mới, chỉ gây trùng/nhiễu -> bỏ."""
        if ch["r"] != "assistant":
            return False
        lst = self.rag.convs.get(ch["c"], [])
        j = ch["p"] - 1
        if not (0 <= j < len(lst)):
            return False
        prev = self.rag.chunks[lst[j]]
        if prev is None or prev["r"] != "user":
            return False
        a, b = set(tokens(ch["t"], bigrams=False)), set(tokens(prev["t"], bigrams=False))
        return bool(b) and len(a & b) / len(b) >= 0.7

    def _compact(self):
        live = [f for f in self.facts if self.rag.chunks[f["cid"]] is not None]
        mark = self.mark
        self._reset()
        self.mark = mark
        for f in live:
            self._add(f["cid"], f["text"], f["who"], f["ts"], f["ents"], f["key"], f["conv"], f["pos"])
        self._supersede()
        self.dirty = True

    def _supersede(self):
        groups = collections.defaultdict(list)
        for fid, f in enumerate(self.facts):
            f["stale"] = False
            if f["key"] and self.rag.chunks[f["cid"]] is not None:
                groups[f["key"]].append(fid)
        self.by_key = {}
        for key, fids in groups.items():
            fids.sort(key=lambda i: (self.facts[i]["ts"], i))
            self.by_key[key] = fids
            for i in fids[:-1]:
                self.facts[i]["stale"] = True

    # ---------------- truy vấn ----------------
    def _live(self, fid):
        return self.rag.chunks[self.facts[fid]["cid"]] is not None

    def _ppr(self, seeds, hops=2, damp=0.5):
        score = collections.defaultdict(float)
        frontier = dict(seeds)
        for e, w in seeds.items():
            score[e] += w
        for _ in range(hops):
            nxt = collections.defaultdict(float)
            for e, w in frontier.items():
                nb = self.co.get(e)
                if not nb:
                    continue
                tot = sum(nb.values())
                for e2, c in nb.items():
                    nxt[e2] += w * damp * c / tot
            for e2, w in nxt.items():
                score[e2] += w
            frontier = nxt
        return score

    def query(self, q, chars=300, k=None, abstain=True):
        """-> {'context': chuỗi gọn, 'facts': [...], 'conf': 0..1, 'intent': ...}. Context đã được nén cho model nhỏ."""
        with self.lock:
            self.sync()
            rag = self.rag
            it = intent_of(q)
            k = k or K_BY_INTENT[it]
            qt = [t for t in tokens(q, bigrams=False) if t.isascii()]
            qset = set(qt) - QWORDS or set(qt)
            wants_place = bool(re.search(r"\b(o dau|dang o|song o|o)\b", fold(q.lower())))
            q_self = has_self(q)   # "mik/mình/tôi..." trong câu hỏi: hỏi về CHÍNH người dùng
            if not qset or not self.facts:
                return {"context": "không tìm thấy trong lịch sử", "facts": [], "conf": 0.0, "intent": it}
            N = max(2, rag.N)
            idf = {t: math.log(1 + N / (1 + len(rag.inv.get(t, ())))) for t in qset}
            typed = {fold(w): w for w in _WORD.findall(q.lower()) if fold(w) != w}   # từ người dùng gõ CÓ DẤU: khi khớp phải đúng dấu ("hiệu" ≠ "hiểu")

            def covers(text):
                ft = set(tokens(text, bigrams=False))
                return {t for t in qset if (typed[t] in ft if t in typed else t in ft)}
            qents = [e for e in entities(q)] + ["c:" + t for t in qset if ("c:" + t) in self.ent]
            # ---- 1. ứng viên ----
            cand = {}
            bm_chunks = {h["cid"]: h["score"] for h in rag.search(q, k=12)}
            bm_max = max(bm_chunks.values(), default=1.0) or 1.0
            for cid in bm_chunks:
                for fid in self.by_cid.get(cid, ()):
                    cand[fid] = 1
            for e in qents:
                for fid in self.ent.get(e, ())[-40:]:
                    cand[fid] = 1
            seeds = {e: 1.0 for e in qents if e in self.ent}
            top_bm = sorted(bm_chunks, key=bm_chunks.get, reverse=True)[:3]
            for cid in top_bm:
                for fid in self.by_cid.get(cid, ()):
                    for e in self.facts[fid]["ents"]:
                        if not e.startswith("c:"):
                            seeds[e] = max(seeds.get(e, 0), 0.5)
            ppr = self._ppr(seeds) if seeds else {}
            for e, w in sorted(ppr.items(), key=lambda kv: -kv[1])[:12]:
                if w > 0.08:
                    for fid in self.ent.get(e, ())[-12:]:
                        cand[fid] = 1
            cand = [fid for fid in cand if self._live(fid)]
            # ---- nhiều bước: sự kiện giai đoạn 1 (khớp từ trong câu hỏi) cho TÊN RIÊNG làm "cầu nối" sang sự kiện đích ----
            def lex_of(fid):
                ft = set(t for t in tokens(self.facts[fid]["text"], bigrams=False) if t.isascii())
                return sum(idf[t] for t in qset & ft) / (sum(idf.values()) or 1), ft
            rare = sorted((t for t in qset if t in rag.inv), key=lambda t: -idf[t])[:2]   # từ HIẾM nhất của câu hỏi thường là từ chỉ quan hệ ("thân", "yêu", "đồng nghiệp")
            stage1 = sorted((fid for fid in cand if any(t in lex_of(fid)[1] for t in rare)) or cand, key=lambda fid: -(lex_of(fid)[0] + 0.5 * bm_chunks.get(self.facts[fid]["cid"], 0.0) / bm_max))[:6]
            bridges = {}
            for fid in stage1:
                lx, ft = lex_of(fid)
                if lx <= 0.2:
                    continue
                for e in self.facts[fid]["ents"]:
                    if e.startswith("p:") and e not in qents:
                        bridges.setdefault(e, set()).update(ft)
            for e in bridges:
                for fid in self.ent.get(e, ())[-15:]:
                    if self._live(fid) and fid not in cand:
                        cand.append(fid)
            if not cand:
                return {"context": "không tìm thấy trong lịch sử", "facts": [], "conf": 0.0, "intent": it}
            # ---- 2. chấm điểm cấp sự kiện ----
            tss = [self.facts[f]["ts"] for f in cand]
            lo, hi = min(tss), max(tss)
            raw = {}
            for fid in cand:
                f = self.facts[fid]
                ft = set(t for t in tokens(f["text"], bigrams=False) if t.isascii())
                lex = sum(idf[t] for t in covers(f["text"])) / (sum(idf.values()) or 1)
                eo = sum(1.0 / math.sqrt(1 + len(self.ent.get(e, ()))) for e in f["ents"] if e in qents) / (len(qents) or 1) if qents else 0.0
                gr = sum(ppr.get(e, 0.0) for e in f["ents"]) / math.sqrt(1 + len(f["ents"]))
                bm = bm_chunks.get(f["cid"], 0.0) / bm_max
                bonus = 0.0
                has_d, has_n = any(e.startswith("d:") for e in f["ents"]), any(e.startswith("n:") for e in f["ents"])
                if it == "when" and has_d:
                    bonus += 0.3
                if it == "howmany" and has_n:
                    bonus += 0.3
                if it in ("who", "where") and any(e.startswith("p:") for e in f["ents"]):
                    bonus += 0.15
                rec = (f["ts"] - lo) / (hi - lo) if hi > lo else 0
                bonus += (0.2 * (1 - rec) if it == "history" else 0.1 * rec) + (0.08 if f["who"] == "user" else -0.05 if f["who"] == "assistant" else 0.02)
                if f["stale"] and it != "history":
                    bonus -= 0.6
                if wants_place and ((f["key"] or "").startswith("noi o") or _PLACEY.search(f["text"])):
                    bonus += 0.4   # câu hỏi "ở đâu" cần sự kiện nói về NƠI Ở (các từ "ở/đang" là từ dừng nên không tự đóng góp điểm)
                if q_self:   # hỏi về bản thân: ưu tiên lời tự kể, hạ sự kiện nói về NGƯỜI KHÁC (tên riêng không có trong câu hỏi)
                    selfy = has_self(f["text"])
                    others = [e for e in f["ents"] if e.startswith("p:") and e not in qents and e not in bridges]
                    bonus += 0.3 if selfy else 0.0
                    bonus -= 0.3 if others and not selfy else 0.0
                    if re.search(r"\bban\s+\w+", fold(f["text"].lower())) and not selfy and not any(e in f["ents"] for e in qents):
                        bonus -= 0.15
                br = 0.0   # khớp cầu nối VÀ phủ phần câu hỏi mà sự kiện cầu nối chưa phủ -> sự kiện đích của bước 2
                for e, covered_by_bridge in bridges.items():
                    if e in f["ents"]:
                        rest = qset - covered_by_bridge
                        br = max(br, len(rest & ft) / max(1, len(rest)))
                raw[fid] = (lex, eo, gr, bm, bonus, br)
            mx = [max(r[i] for r in raw.values()) or 1.0 for i in range(4)]
            final = {fid: 1.0 * r[0] / mx[0] + 0.8 * r[1] / mx[1] + 0.5 * r[2] / mx[2] + 0.6 * r[3] / mx[3] + r[4] + 1.2 * r[5] for fid, r in raw.items()}
            # ---- 3. cắt tỉa: MMR + dừng khi đã phủ đủ ý ----
            order = sorted(final, key=final.get, reverse=True)
            top = final[order[0]]
            self.last_scores = [(round(final[f], 2), self.facts[f]["text"], [round(x, 2) for x in raw[f]]) for f in order[:8]]   # để gỡ lỗi/giải thích: điểm + (lex, thực thể, đồ thị, bm25, thưởng, cầu nối)
            chosen, covered = [], set()
            toks_of = {fid: set(t for t in tokens(self.facts[fid]["text"], bigrams=False) if t.isascii()) for fid in order[:40]}
            while len(chosen) < k:
                best, bs = None, -1e9
                for fid in order[:40]:
                    if fid in chosen:
                        continue
                    sim = max((len(toks_of[fid] & toks_of[c]) / max(1, len(toks_of[fid] | toks_of[c])) for c in chosen), default=0.0)
                    if sim > 0.8:
                        continue                      # gần trùng (lời Yui nhại lại câu của người dùng)
                    s = 0.75 * final[fid] - 0.25 * sim * top
                    if s > bs:
                        best, bs = fid, s
                if best is None or final[best] < 0.32 * top:
                    break
                chosen.append(best)
                covered |= covers(self.facts[best]["text"])
                cov = sum(idf[t] for t in covered) / (sum(idf.values()) or 1)
                if cov >= 0.9 and len(chosen) >= (2 if it in ("list", "history") else 1) and it != "list":
                    break
            if it == "history":   # hỏi về quá khứ: thêm giá trị CŨ NHẤT của thuộc tính (lúc đầu/ban đầu), giữ cả bản hiện tại để model thấy diễn tiến
                for fid in list(chosen):
                    grp = self.by_key.get(self.facts[fid]["key"] or "", [])
                    live = [x for x in grp if self._live(x)]
                    if live and live[0] not in chosen and len(chosen) < k + 1:
                        chosen.append(live[0])
            # thay sự kiện cũ bằng bản mới nhất cùng khoá (đảm bảo giá trị hiện tại)
            if it != "history":
                for i, fid in enumerate(chosen):
                    f = self.facts[fid]
                    if f["stale"] and f["key"]:
                        newest = max((x for x in range(len(self.facts)) if self.facts[x]["key"] == f["key"] and self._live(x)), key=lambda x: (self.facts[x]["ts"], x))
                        chosen[i] = newest
                chosen = list(dict.fromkeys(chosen))
            # ---- 4. độ tự tin ----
            cov_ents = {e for fid in chosen for e in self.facts[fid]["ents"]}
            got = {t for fid in chosen for t in covers(self.facts[fid]["text"])}
            cov = sum(idf[t] for t in got) / (sum(idf.values()) or 1)
            cov = max(cov, 0.5 * len(set(qents) & cov_ents) / (len(qents) or 1) if qents else 0)
            conf = round(min(1.0, 0.7 * cov + 0.3 * min(1.0, top)), 2)
            # từ khoá phân biệt nhất của câu hỏi ("cáo", "căn cước", "xe máy") mà KHÔNG sự kiện nào chứa -> thông tin chưa từng được kể: từ chối thay vì đưa nhiễu cho model bịa
            keyt = [t for t in qset if idf[t] >= 0.75 * max(idf.values())]
            if abstain and (cov < 0.34 or (keyt and not (set(keyt) & got) and not ({e for e in qents if not e.startswith("c:")} & cov_ents))):
                return {"context": "không tìm thấy trong lịch sử", "facts": [], "conf": conf, "intent": it}
            # ---- 5. kể lại gọn, theo thời gian ----
            chosen.sort(key=lambda x: (self.facts[x]["ts"], self.facts[x]["conv"], self.facts[x]["pos"]))
            lines, budget = [], chars
            for j, fid in enumerate(chosen):
                f = self.facts[fid]
                who = "bạn" if f["who"] == "user" else rag.bot if f["who"] == "assistant" else "tài liệu" if f["who"] == "kb" else "ghi chú"
                tag = " (cũ)" if f["stale"] else ""
                head = f"[{time.strftime('%d/%m', time.localtime(f['ts']))}] {who}{tag}: "
                room = budget - len(head) - (len(chosen) - j - 1) * 30
                if room < 25:
                    break
                lines.append(head + focus(f["text"], qset, min(room, budget - len(head))))
                budget -= len(lines[-1]) + 1
            return {"context": "\n".join(lines) or "không tìm thấy trong lịch sử", "facts": [self.facts[f]["text"] for f in chosen], "conf": conf, "intent": it,
                    "ids": chosen}

    def recall(self, q, chars=300, k=None, fallback=False):
        """Thay thế trực tiếp RAG.recall: cùng định dạng '[dd/mm] bạn: ...' nhưng gọn, sạch cũ/trùng. Lùi về BM25 khi đồ thị không đủ chắc."""
        r = self.query(q, chars, k)
        if r["facts"] or not fallback:
            return r["context"]
        return self.rag.recall(q, chars=min(chars, 160), k=2)

    def related(self, term, n=8):
        with self.lock:
            self.sync()
            f = fold(term.strip().lower())
            keys = [e for e in (f"p:{f}", f"c:{f}", f"d:{f}", f"n:{f}") if e in self.ent]
            out = collections.Counter()
            for e in keys:
                for e2, c in self.co.get(e, {}).items():
                    out[e2] += c
            show = [f"{self.rag.form.get(e[2:], e[2:])} ({c})" for e, c in out.most_common(n)]
            return ", ".join(show) if show else "không có từ liên quan"

    def flush(self):
        """Ghi đồ thị xuống đĩa nếu có thay đổi (gọi khi đóng app)."""
        with self.lock:
            if self.dirty:
                self.save()
                self._saved = time.time()

    def stats(self):
        live = [f for f in self.facts if self.rag.chunks[f["cid"]] is not None]
        return {"facts": len(live), "entities": len(self.ent), "edges": sum(len(v) for v in self.co.values()) // 2, "superseded": sum(1 for f in live if f["stale"])}


def focus(text, qset, maxlen):
    """Cắt câu dài quanh mệnh đề liên quan nhất tới câu hỏi (giữ nguyên nếu đã ngắn)."""
    text = " ".join(text.split())
    if len(text) <= maxlen:
        return text
    clauses = [c.strip() for c in re.split(r"(?<=[,;:])\s+", text) if c.strip()]
    best = max(range(len(clauses)), key=lambda i: sum(1 for t in tokens(clauses[i], bigrams=False) if t in qset))
    out = clauses[best]
    i, j = best, best
    while True:
        grew = False
        if i > 0 and len(clauses[i - 1]) + len(out) + 1 <= maxlen:
            i -= 1; out = clauses[i] + " " + out; grew = True
        if j < len(clauses) - 1 and len(out) + len(clauses[j + 1]) + 1 <= maxlen:
            j += 1; out = out + " " + clauses[j]; grew = True
        if not grew:
            break
    return out if len(out) <= maxlen else out[:maxlen - 1].rstrip() + "…"
