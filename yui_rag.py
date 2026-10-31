#!/usr/bin/env python3
"""
yui_rag.py - RAG mini TỰ CHẾ cho model Yui (chỉ thư viện chuẩn, chạy được trên điện thoại): tìm lại lịch sử chat + ghi chú trong thư mục làm việc.

  * Nạp: mọi file trong chat/ (.json .jsonl .txt .md; nhận cả file xuất của ChatGPT) và notes/ (.md). Chép thêm file vào thư mục => lần mở sau tự nạp (sync).
    Đổi điện thoại: chỉ cần chép lại thư mục (hoặc chỉ chat/ + notes/ + skills/), chỉ mục tự dựng lại.
  * Tìm: BM25 trên từ + cặp từ, hiểu tiếng Việt KHÔNG DẤU (gõ "ten meo" vẫn ra "tên mèo"), ưu tiên nhẹ nội dung mới hơn.
  * Đồ thị: đồ thị đồng xuất hiện giữa các từ khoá (cùng nằm trong một đoạn) + liên kết [[ghi chú]]. Dùng để MỞ RỘNG truy vấn bằng từ liên quan và kéo thêm
    câu hỏi/câu trả lời liền kề trong cùng cuộc chat; có thể xem trực tiếp bằng related().
  * Ghi nhớ: remember(user, assistant) thêm lượt chat mới vào chat/yui-<ngày>.jsonl (mã hoá nếu kho có khoá) và cập nhật chỉ mục ngay.
  * Ghi chú: add_note(tiêu đề, nội dung) -> notes/<tiêu-đề>.md (nối thêm nếu đã có).
  * Chỉ mục lưu ở index/index.bin (nén, mã hoá nếu có khoá). Xoá đi cũng được, tự dựng lại từ chat/ và notes/.
Nội dung tìm được là DỮ LIỆU KHÔNG TIN CẬY (có thể chứa chữ <tool>...): yui_agent.sanitize() vô hiệu hoá thẻ trước khi đưa cho model.
"""
import collections
import functools
import json
import math
import os
import re
import threading
import time
import unicodedata
import zlib

from yui_vault import VaultError

_WORD = re.compile(r"[0-9A-Za-zÀ-ỹ]+", re.U)
_SENT = re.compile(r"(?<=[.!?…])\s+|\n+")
_LINK = re.compile(r"\[\[([^\[\]\n]{1,60})\]\]")
STOP = set("va la cua co khong cho the nay do duoc mot nhung cac voi trong khi da se rat cung nhu thi ma o tu ra vao den lai con nen vi neu hay roi "
           "ban toi minh mik oi nha a day ne nhe the nao gi sao anh chi em ho no ay thoi luon van dang bi boi ve len xuong "
           "the a an of to in on is are was were be and or for it that this".split())
# Từ dừng CÓ DẤU: chỉ bỏ khi người dùng thực sự gõ dạng này. Bỏ dấu làm "chó" trùng "cho", "đá" trùng "da"..., nên KHÔNG được coi "chó" là từ dừng chỉ vì dạng không dấu của nó là "cho".
STOP_ACC = set("và là của có không cho thế này đó được một những các với trong khi đã sẽ rất cũng như thì mà ở từ ra vào đến lại còn nên vì nếu hay rồi bạn tôi mình ơi nhé nhỉ nè đây "
               "thế nào gì sao anh chị em họ nó ấy thôi luôn vẫn đang bị bởi về lên xuống".split())
VERSION = 4   # tăng khi đổi cách tách từ: chỉ mục cũ tự dựng lại


@functools.lru_cache(maxsize=65536)
def fold(w):
    """Bỏ dấu tiếng Việt: 'Mèo' -> 'meo', 'đường' -> 'duong'."""
    s = unicodedata.normalize("NFD", w.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if not unicodedata.combining(c))


def tokens(text, bigrams=True):
    """Từ khoá để lập chỉ mục/tìm kiếm: dạng KHÔNG DẤU (khớp mọi cách gõ) + dạng có dấu (khớp chính xác hơn) + cặp từ liền nhau."""
    out, prev = [], None
    for i, orig in enumerate(_WORD.findall(text)):
        w = orig.lower()
        if len(w) > 40:
            prev = None
            continue
        f = fold(w)
        name = i > 0 and orig[:1].isupper() and not orig.isupper()   # tên riêng giữa câu ("bạn Minh"): không phải đại từ "mình"
        if (f in STOP and not name and (w == f or w in STOP_ACC)) or (len(f) == 1 and not f.isdigit()):
            prev = None
            continue
        out.append(f)
        if w != f:
            out.append(w)
        if bigrams and prev:
            out.append(prev + "_" + f)
        prev = f
    return out


def split_text(text, maxlen=500):
    text = text.strip()
    if len(text) <= maxlen:
        return [text] if text else []
    parts, cur = [], ""
    for s in _SENT.split(text):
        if not s:
            continue
        while len(s) > maxlen:
            if cur:
                parts.append(cur)
                cur = ""
            parts.append(s[:maxlen])
            s = s[maxlen:]
        if len(cur) + len(s) + 1 > maxlen and cur:
            parts.append(cur)
            cur = s
        else:
            cur = (cur + " " + s).strip()
    if cur:
        parts.append(cur)
    return parts


# ============================ đọc nhiều định dạng lịch sử chat ============================
_ROLE = {"user": "user", "human": "user", "ban": "user", "mik": "user", "me": "user", "assistant": "assistant", "bot": "assistant", "gpt": "assistant",
         "model": "assistant", "ai": "assistant", "yui": "assistant"}
_MARK = re.compile(r"^\s*(user|human|bạn|ban|mik|assistant|bot|ai|yui)\s*[:：]\s*(.*)$", re.I)


def _role(x):
    return _ROLE.get(fold(str(x).strip().lower()))


def _text_of(m):
    c = m.get("content", m.get("value", m.get("text", "")))
    if isinstance(c, dict):
        c = c.get("parts") or c.get("text") or ""
    if isinstance(c, list):
        c = " ".join(p if isinstance(p, str) else str(p.get("text", "")) if isinstance(p, dict) else "" for p in c)
    return str(c or "").strip()


def _ts_of(m, default):
    for k in ("ts", "create_time", "time", "timestamp", "created_at"):
        v = m.get(k)
        if isinstance(v, (int, float)) and v > 1e9:
            return float(v if v < 1e12 else v / 1000)
    return default


def _conv_from_msgs(msgs, ts):
    out = []
    for m in msgs:
        if not isinstance(m, dict):
            continue
        a = m.get("author")
        raw = m.get("role") or m.get("from") or (a.get("role") if isinstance(a, dict) else a)
        r = _role(raw) if raw else None
        t = _text_of(m)
        if r and t:
            out.append((r, t, _ts_of(m, ts)))
    return out


def _convs_from_obj(o, ts):
    convs = []
    if isinstance(o, list):
        if o and all(isinstance(x, dict) and ("role" in x or "from" in x) for x in o):
            c = _conv_from_msgs(o, ts)
            return [c] if c else []
        for x in o:
            convs += _convs_from_obj(x, ts)
    elif isinstance(o, dict):
        if isinstance(o.get("mapping"), dict):   # file xuất của ChatGPT
            nodes = [n.get("message") for n in o["mapping"].values() if isinstance(n, dict) and isinstance(n.get("message"), dict)]
            nodes.sort(key=lambda m: m.get("create_time") or 0)
            c = _conv_from_msgs(nodes, _ts_of(o, ts))
            return [c] if c else []
        for key in ("messages", "conversations", "history", "chat"):
            if isinstance(o.get(key), list):
                return _convs_from_obj(o[key], _ts_of(o, ts))
        pair = [(a, b) for a, b in (("user", "assistant"), ("prompt", "response"), ("question", "answer"), ("instruction", "output")) if a in o and b in o]
        if pair:
            a, b = pair[0]
            return [[("user", str(o[a]).strip(), _ts_of(o, ts)), ("assistant", str(o[b]).strip(), _ts_of(o, ts))]]
    return [c for c in convs if c]


def parse_chat(text, name, ts):
    """-> list các cuộc hội thoại, mỗi cuộc là [(role, nội dung, thời gian)]. role: user | assistant | note."""
    text = text.lstrip("\ufeff")
    low = name.lower()
    if low.endswith(".json") or low.endswith(".jsonl") or text.lstrip()[:1] in "[{":
        try:
            return _convs_from_obj(json.loads(text), ts)
        except ValueError:
            pass
        convs = []
        for ln in text.splitlines():
            ln = ln.strip()
            if ln[:1] in "[{":
                try:
                    convs += _convs_from_obj(json.loads(ln), ts)
                except ValueError:
                    continue
        if convs:
            return convs
    turns, cur = [], None
    for ln in text.splitlines():
        m = _MARK.match(ln)
        if m:
            cur = [_role(m.group(1)) or "user", m.group(2)]
            turns.append(cur)
        elif cur is not None:
            cur[1] += " " + ln.strip()
        elif ln.strip():
            cur = ["note", ln.strip()]
            turns.append(cur)
    turns = [(r, " ".join(t.split()), ts) for r, t in turns if t.strip()]
    return [turns] if turns else []


# ============================ chỉ mục ============================
class RAG:
    def __init__(self, ws, bot_name="Yui"):
        self.ws, self.bot, self.lock = ws, bot_name, threading.RLock()
        self._reset()
        self.load()

    def _reset(self):
        self.chunks, self.terms, self.dl = [], [], []
        self.inv, self.files, self.graph, self.form, self.links, self.convs = {}, {}, {}, {}, {}, {}
        self.N, self.totlen, self.dirty = 0, 0, False

    # ---------------- lưu / nạp ----------------
    def _index_rel(self):
        return "index.bin"

    def save(self):
        with self.lock:
            blob = zlib.compress(json.dumps({"v": VERSION, "chunks": self.chunks, "dl": self.dl, "files": self.files, "inv": {t: [[c, f] for c, f in p.items()] for t, p in self.inv.items()},
                                             "graph": self.graph, "form": self.form, "links": self.links}, ensure_ascii=False, separators=(",", ":")).encode("utf-8"), 6)
            for stale in (os.path.join(self.ws.root, "index", "index.bin"), os.path.join(self.ws.root, "index", "index.bin.enc")):
                if os.path.exists(stale):
                    os.unlink(stale)
            self.ws.write_bytes("index.bin", blob, sub="index")
            self.dirty = False

    def load(self):
        p = self.ws.find("index.bin", "index")
        if not p:
            return
        try:
            d = json.loads(zlib.decompress(self.ws.read_bytes(p)).decode("utf-8"))
            if d.get("v") != VERSION:
                raise ValueError("phiên bản chỉ mục cũ")
            self.chunks, self.dl, self.files, self.graph, self.form, self.links = d["chunks"], d["dl"], d["files"], d["graph"], d["form"], d["links"]
            self.inv = {t: {c: f for c, f in post} for t, post in d["inv"].items()}
            tl = [[] for _ in self.chunks]
            for t, post in self.inv.items():
                for c in post:
                    tl[c].append(t)
            self.terms = [tuple(x) for x in tl]
            self.convs, self.N, self.totlen = {}, 0, 0
            for cid, ch in enumerate(self.chunks):
                if ch is None:
                    continue
                self.N += 1
                self.totlen += self.dl[cid]
                self.convs.setdefault(ch["c"], []).append(cid)
        except (VaultError, OSError, ValueError, KeyError, IndexError, zlib.error) as e:
            self.ws.log_error("rag.load", f"chỉ mục hỏng, sẽ dựng lại: {e}")
            self._reset()

    # ---------------- thêm / xoá ----------------
    def _add_chunk(self, src, conv, idx, role, text, ts):
        cid = len(self.chunks)
        toks = tokens(text)
        tf = collections.Counter(toks)
        lst = self.convs.setdefault(conv, [])
        self.chunks.append({"s": src, "c": conv, "i": idx, "p": len(lst), "r": role, "t": text, "ts": int(ts)})
        lst.append(cid)
        self.terms.append(tuple(tf))
        self.dl.append(len(toks))
        self.totlen += len(toks)
        self.N += 1
        for t, c in tf.items():
            self.inv.setdefault(t, {})[cid] = c
        if len(self.form) < 60000:
            for w in set(_WORD.findall(text.lower())):
                f = fold(w)
                if f != w and len(w) > 2:
                    self.form[f] = w
        return cid

    def _drop_file(self, rel):
        info = self.files.pop(rel, None)
        self.links.pop(rel, None)
        for cid in (info or {}).get("cids", []):
            ch = self.chunks[cid]
            if ch is None:
                continue
            for t in self.terms[cid]:
                d = self.inv.get(t)
                if d:
                    d.pop(cid, None)
                    if not d:
                        del self.inv[t]
            lst = self.convs.get(ch["c"], [])
            if cid in lst:
                lst.remove(cid)
            self.totlen -= self.dl[cid]
            self.N -= 1
            self.chunks[cid], self.terms[cid] = None, ()
        self.dirty = True

    def _graph_batch(self, cids):
        N = max(2, self.N)
        g = self.graph
        for cid in cids:
            if self.chunks[cid] is None:
                continue
            cand = [t for t in self.terms[cid] if t.isascii() and "_" not in t and len(t) >= 3 and not t.isdigit()]
            cand.sort(key=lambda t: math.log(N / (1 + len(self.inv.get(t, ())))), reverse=True)
            top = cand[:8]
            for i, a in enumerate(top):
                for b in top[i + 1:]:
                    g.setdefault(a, {})[b] = g.get(a, {}).get(b, 0) + 1
                    g.setdefault(b, {})[a] = g.get(b, {}).get(a, 0) + 1
        for t in [t for t, nb in g.items() if len(nb) > 80]:   # giữ đồ thị gọn: mỗi từ tối đa 40 láng giềng mạnh nhất
            g[t] = dict(sorted(g[t].items(), key=lambda kv: -kv[1])[:40])

    def _ingest(self, rel, text, mtime, sub):
        cids = []
        if sub in ("notes", "kb"):
            title = re.sub(r"\.(md|txt|enc)$", "", os.path.basename(rel), flags=re.I)
            title = ("tóm tắt " + title) if rel.startswith("kb/_tomtat/") else title
            if sub == "notes":
                self.links[rel] = list(dict.fromkeys(_LINK.findall(text)))[:30]
            for i, part in enumerate(p for para in re.split(r"\n\s*\n", text) for p in split_text(para)):
                cids.append(self._add_chunk(rel, rel, i, "note" if sub == "notes" else "kb", f"{title}: {part}" if i == 0 or sub == "kb" else part, mtime))
        else:
            for k, conv in enumerate(parse_chat(text, rel, mtime)):
                i = 0
                for role, t, ts in conv:
                    for part in split_text(" ".join(t.split())):
                        cids.append(self._add_chunk(rel, f"{rel}#{k}", i, role, part, ts))
                        i += 1
        return cids

    def sync(self):
        """Quét chat/ và notes/: nạp file mới/đổi, bỏ file đã xoá. Trả số file thay đổi."""
        with self.lock:
            changed, new, seen = 0, [], set()
            for sub in ("chat", "notes", "kb"):
                for p, mt, sz in self.ws.list_files(sub):
                    rel = self.ws.rel(p)
                    if sub == "kb" and not rel.startswith(("kb/_clean/", "kb/_tomtat/")):
                        continue   # chỉ nạp bản đã lọc rác (yui_kb.py), không nạp file gốc
                    seen.add(rel)
                    sig = [int(mt), sz]
                    old = self.files.get(rel)
                    if old and old["sig"] == sig:
                        continue
                    if old:
                        self._drop_file(rel)
                    try:
                        text = self.ws.read_text(p)
                        cids = self._ingest(rel, text, mt, sub)
                    except (VaultError, OSError, ValueError, MemoryError) as e:
                        self.ws.log_error("rag.sync", f"{rel}: {e}")
                        cids = []
                    self.files[rel] = {"sig": sig, "cids": cids}
                    new += cids
                    changed += 1
            for rel in [r for r in self.files if r not in seen]:
                self._drop_file(rel)
                changed += 1
            if new:
                self._graph_batch(new)
            if changed or self.dirty:
                self.save()
            return changed

    def _reindex(self, rel, sub):
        p = os.path.join(self.ws.root, rel)
        st = os.stat(p)
        if rel in self.files:
            self._drop_file(rel)
        cids = self._ingest(rel, self.ws.read_text(p), st.st_mtime, sub)
        self.files[rel] = {"sig": [int(st.st_mtime), st.st_size], "cids": cids}
        self._graph_batch(cids)
        self.save()

    # ---------------- ghi nhớ ----------------
    def remember(self, user, assistant, ts=None):
        """Lưu một lượt chat mới vào chat/yui-<ngày>.jsonl và cập nhật chỉ mục ngay. Trả đường dẫn tương đối của file."""
        user, assistant = " ".join(str(user).split())[:4000], " ".join(str(assistant).split())[:4000]
        if not user or not assistant:
            return None
        with self.lock:
            ts = ts or time.time()
            day, n = time.strftime("%Y%m%d", time.localtime(ts)), 1
            while True:
                name = f"yui-{day}" + (f"-{n}" if n > 1 else "") + ".jsonl"
                p = self.ws.find(name, "chat")
                old = self.ws.read_text(p) if p else ""
                if old.count("\n") < 400:
                    break
                n += 1
            line = json.dumps({"ts": int(ts), "messages": [{"role": "user", "content": user}, {"role": "assistant", "content": assistant}]}, ensure_ascii=False)
            if p and not self.ws.encrypted and not p.endswith(".enc"):
                with open(p, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
                rel = self.ws.rel(p)
            else:
                rel = self.ws.write_text(name, old + line + "\n", sub="chat")
                rel = rel if rel.startswith("chat/") else "chat/" + rel
            k = old.count("\n")
            cids, i = [], 0
            for role, t in (("user", user), ("assistant", assistant)):
                for part in split_text(t):
                    cids.append(self._add_chunk(rel, f"{rel}#{k}", i, role, part, ts))
                    i += 1
            full = os.path.join(self.ws.root, rel)
            st = os.stat(full)
            info = self.files.setdefault(rel, {"sig": [0, 0], "cids": []})
            info["sig"], info["cids"] = [int(st.st_mtime), st.st_size], info["cids"] + cids
            self._graph_batch(cids)
            self.dirty = True
            if len(self.chunks) % 20 == 0:
                self.save()
            return rel

    def add_note(self, title, text):
        """Tạo/nối ghi chú notes/<tiêu-đề>.md. Trả tên ghi chú."""
        from yui_vault import slug
        title, text = " ".join(str(title).split())[:80], str(text).strip()[:2000]
        if not title or not text:
            raise VaultError("ghi chú cần tiêu đề và nội dung")
        with self.lock:
            nm, n = slug(title), 1
            while True:
                fn = f"{nm}.md" if n == 1 else f"{nm}-{n}.md"
                p = self.ws.find(fn, "notes")
                old = self.ws.read_text(p) if p else ""
                if len(old) + len(text) < 20000:
                    break
                n += 1
            body = (old.rstrip() + "\n\n" + text) if old else f"# {title}\n\n{text}"
            rel = self.ws.write_text(fn, body, sub="notes")
            self._reindex(rel if rel.startswith("notes/") else "notes/" + rel, "notes")
            return fn[:-3]

    # ---------------- tìm kiếm ----------------
    def search(self, query, k=4, role=None):
        """-> danh sách chunk (dict) xếp theo độ liên quan, đã mở rộng bằng đồ thị + câu liền kề."""
        with self.lock:
            qt = tokens(query)
            if not qt or not self.N:
                return []
            base = [t for t in qt if t.isascii() and "_" not in t]
            w = {}
            for t in qt:
                w[t] = 1.6 if "_" in t else 1.0
            for t in base:   # mở rộng bằng đồ thị: từ hay đi cùng
                for nb, c in sorted(self.graph.get(t, {}).items(), key=lambda kv: -kv[1])[:3]:
                    if c >= 2 and nb not in w:
                        w[nb] = 0.35
            avg, N, k1, b = self.totlen / max(1, self.N), self.N, 1.5, 0.75
            sc = collections.defaultdict(float)
            for t, wq in w.items():
                post = self.inv.get(t)
                if not post:
                    continue
                idf = math.log(1 + (N - len(post) + 0.5) / (len(post) + 0.5))
                for cid, tf in post.items():
                    sc[cid] += wq * idf * tf * (k1 + 1) / (tf + k1 * (1 - b + b * self.dl[cid] / avg))
            if not sc:
                return []
            tss = [self.chunks[c]["ts"] for c in sc]
            lo, hi = min(tss), max(tss)
            for c in list(sc):
                ch = self.chunks[c]
                if role and ch["r"] != role:
                    del sc[c]
                    continue
                if hi > lo:
                    sc[c] *= 1 + 0.15 * (ch["ts"] - lo) / (hi - lo)
            ranked = sorted(sc.items(), key=lambda kv: -kv[1])[:k]
            final = dict(ranked)
            for cid, s in ranked[:max(1, k // 2)]:   # kéo câu liền kề (hỏi <-> đáp) của kết quả tốt nhất
                ch = self.chunks[cid]
                lst = self.convs.get(ch["c"], [])
                for d in (-1, 1):
                    j = ch["p"] + d
                    if 0 <= j < len(lst) and lst[j] not in final and self.chunks[lst[j]] is not None and self.chunks[lst[j]]["r"] != ch["r"]:
                        final[lst[j]] = s * 0.45
            top = sorted(final.items(), key=lambda kv: -kv[1])[:k + 2]
            return [dict(self.chunks[c], score=round(s, 3), cid=c) for c, s in sorted(top, key=lambda kv: (self.chunks[kv[0]]["c"], self.chunks[kv[0]]["i"]))]

    def fmt(self, ch):
        who = "bạn" if ch["r"] == "user" else self.bot if ch["r"] == "assistant" else "tài liệu" if ch["r"] == "kb" else "ghi chú"
        return f"[{time.strftime('%d/%m', time.localtime(ch['ts']))}] {who}: {' '.join(ch['t'].split())}"

    def recall(self, query, chars=300, k=4):
        """Chuỗi ngắn đưa cho model: các đoạn liên quan nhất, vừa khít `chars` ký tự."""
        hits = self.search(query, k)
        if not hits:
            return "không tìm thấy trong lịch sử"
        best = sorted(hits, key=lambda h: -h["score"])
        lines, budget = {}, chars
        for h in best:
            line = self.fmt(h)
            if budget < 40:
                break
            if len(line) > budget:
                line = line[:budget - 1] + "…"
            lines[h["cid"]] = line
            budget -= len(line) + 1
        return "\n".join(lines[h["cid"]] for h in hits if h["cid"] in lines)

    def related(self, term, n=8):
        """Từ/ghi chú liên quan tới `term` theo đồ thị."""
        with self.lock:
            f = fold(term.strip().lower())
            nb = sorted(self.graph.get(f, {}).items(), key=lambda kv: -kv[1])[:n]
            out = [f"{self.form.get(t, t)} ({c})" for t, c in nb]
            for rel, ls in self.links.items():
                if f in (fold(os.path.basename(rel).lower()[:-3]), ) and ls:
                    out += [f"[[{x}]]" for x in ls[:n]]
            return ", ".join(out) if out else "không có từ liên quan"

    @property
    def graphify(self):
        """Đồ thị sự kiện (yui_graph.py): recall gọn, sạch cũ/trùng, hợp model nhỏ."""
        if getattr(self, "_g", None) is None:
            from yui_graph import Graphify
            self._g = Graphify(self)
        return self._g

    def stats(self):
        return {"chunks": self.N, "files": len(self.files), "terms": len(self.inv), "graph_nodes": len(self.graph), "encrypted": self.ws.encrypted}
