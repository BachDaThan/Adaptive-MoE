#!/usr/bin/env python3
"""
yui_kb.py - NẠP TÀI LIỆU CÁ NHÂN vào kho tri thức, có THANH TIẾN ĐỘ, lọc rác và cô đọng để model nhỏ (1B) xử lý nhanh và chính xác. Chỉ thư viện chuẩn.

  Người dùng chép tài liệu vào  <kho>/kb/  (.txt .md .html .json .jsonl .csv .tsv .docx; .pdf nếu máy có pypdf). Chạy ingest() (hoặc `python yui_agent.py mem kb`) thì mỗi file qua 5 bước:
    1. đọc      trích chữ theo định dạng (HTML bỏ script/menu/chân trang; docx đọc XML; csv thành "cột: giá trị")
    2. dọn      chuẩn hoá Unicode, bỏ số trang / dòng lặp (đầu-cuối trang) / dòng chỉ có link / lời mời đăng ký-cookie, nối dòng bị ngắt giữa câu, bỏ đoạn trùng và gần trùng
    3. tóm tắt  chọn các câu giàu thông tin nhất (điểm TF-IDF + vị trí + có số/ngày) trong một ngân sách ký tự -> bản "tóm tắt" ngắn + từ khoá
    4. ghi      kb/_clean/<tên>.txt (chữ sạch, vẫn đủ chi tiết để tìm) và kb/_tomtat/<tên>.md (bản tóm tắt); mã hoá nếu kho có khoá
    5. chỉ mục  RAG nạp 2 thư mục trên (KHÔNG nạp file gốc, nên rác không lọt vào kết quả tìm)
  Model dùng: `recall` tìm trong chữ sạch; `kb` liệt kê tài liệu; `kb <tên>` đọc bản tóm tắt ngắn (vừa ngữ cảnh nhỏ).
  Chỉ xử lý file mới/đổi (so chữ ký). Có cờ `stop` (threading.Event): bấm dừng thì dừng ngay giữa chừng, file đang làm dở không bị ghi hỏng (ghi nguyên tử).
  on_progress(phần_trăm, bước, tên_file) được gọi liên tục để vẽ thanh tiến độ (CLI có sẵn bản chữ).
"""
import collections
import html as _html
import io
import json
import math
import os
import re
import time
import unicodedata
import zipfile
from html.parser import HTMLParser

from yui_rag import fold, tokens
from yui_vault import VaultError, slug

KB_EXT = (".txt", ".md", ".html", ".htm", ".json", ".jsonl", ".csv", ".tsv", ".docx", ".pdf")
MAX_IN = 8 * 1024 * 1024
CLEAN_DIR, SUM_DIR = "kb/_clean", "kb/_tomtat"
BOILER = re.compile(r"(cookie|đăng nhập|đăng ký nhận|subscribe|newsletter|all rights reserved|bản quyền|chính sách bảo mật|privacy policy|terms of (use|service)|"
                    r"click here|read more|xem thêm|chia sẻ|share on|follow us|theo dõi chúng tôi|skip to content|menu chính|back to top|advertisement|quảng cáo)", re.I)
_SENT = re.compile(r"(?<=[.!?…])\s+(?=[A-ZÀ-Ỹ0-9\"“(])|\n+")


class Cancelled(Exception):
    pass


class KBError(ValueError):
    pass


# ============================ 1. đọc ============================
class _Strip(HTMLParser):
    SKIP = {"script", "style", "noscript", "nav", "footer", "header", "aside", "form", "svg", "iframe", "template"}
    BLOCK = {"p", "div", "br", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "table", "ul", "ol", "blockquote", "pre"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag in self.BLOCK:
            self.out.append("\n")
        if tag in ("h1", "h2", "h3"):
            self.out.append("\n# ")

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.skip:
            self.skip -= 1
        elif tag in self.BLOCK:
            self.out.append("\n")

    def handle_data(self, d):
        if not self.skip:
            self.out.append(d)


def _from_html(s):
    p = _Strip()
    p.feed(s)
    return "".join(p.out)


def _from_docx(data):
    with zipfile.ZipFile(io.BytesIO(data)) as z:
        info = z.getinfo("word/document.xml")
        if info.file_size > 40 * 1024 * 1024:
            raise KBError("docx quá lớn")
        xml = z.read("word/document.xml").decode("utf-8", "replace")
    xml = re.sub(r"</w:p>", "\n", xml)
    xml = re.sub(r"<w:(tab|br)\b[^>]*/>", " ", xml)
    return _html.unescape(re.sub(r"<[^>]+>", "", xml))


def _from_pdf(data):
    try:
        from pypdf import PdfReader
    except ImportError:
        raise KBError("đọc PDF cần: pip install pypdf (hoặc xuất PDF sang .txt/.docx)")
    try:
        r = PdfReader(io.BytesIO(data))
        return "\n".join((pg.extract_text() or "") for pg in r.pages[:400])
    except Exception as e:  # noqa: BLE001 - pypdf ném nhiều loại lỗi cho PDF hỏng/có mật khẩu
        raise KBError(f"không đọc được PDF ({type(e).__name__})")


def _rows(text, delim):
    import csv
    rd = csv.reader(io.StringIO(text), delimiter=delim)
    head, out = None, []
    for i, row in enumerate(rd):
        if i > 5000:
            break
        if head is None:
            head = [h.strip() for h in row]
            continue
        out.append("; ".join(f"{h or 'cột'+str(j+1)}: {c.strip()}" for j, (h, c) in enumerate(zip(head, row)) if c.strip()))
    return "\n".join(out)


def _from_json(text):
    from yui_condense import render
    try:
        obj = json.loads(text)
    except ValueError:
        objs = []
        for ln in text.splitlines():
            try:
                objs.append(json.loads(ln))
            except ValueError:
                continue
        obj = objs
    if isinstance(obj, list):
        return "\n".join(render(x if isinstance(x, list) else [x] if isinstance(x, dict) else x, 1, 8, 400) if isinstance(x, (dict, list)) else str(x) for x in obj[:5000])
    return render(obj, 400, 12, 400) if isinstance(obj, (dict, list)) else str(obj)


def extract_text(name, data):
    ext = os.path.splitext(name.lower().removesuffix(".enc"))[1]
    if ext == ".docx":
        return _from_docx(data)
    if ext == ".pdf":
        return _from_pdf(data)
    text = data.decode("utf-8", "replace")
    if ext in (".html", ".htm") or text.lstrip()[:15].lower().startswith(("<!doctype", "<html")):
        return _from_html(text)
    if ext == ".csv":
        return _rows(text, ",")
    if ext == ".tsv":
        return _rows(text, "\t")
    if ext in (".json", ".jsonl"):
        return _from_json(text)
    return text


# ============================ 2. dọn ============================
def _shingles(s, k=4):
    w = fold(s.lower()).split()
    return {hash(" ".join(w[i:i + k])) for i in range(max(1, len(w) - k + 1))}


def clean(text):
    """-> list đoạn văn sạch (đã bỏ rác, trùng lặp)."""
    text = unicodedata.normalize("NFC", text.replace("\r\n", "\n").replace("\r", "\n"))
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f\u200b\ufeff]", "", text)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)                      # nối từ bị ngắt bằng gạch nối cuối dòng
    lines = [re.sub(r"[ \t]+", " ", ln).strip() for ln in text.split("\n")]
    freq = collections.Counter(ln.lower() for ln in lines if 3 < len(ln) < 90)
    keep = []
    for ln in lines:
        if not ln:
            keep.append("")
            continue
        low = ln.lower()
        if re.fullmatch(r"[-–—_=*·•.\s\d/|]+", ln) or re.fullmatch(r"(trang|page)\s*\d+(\s*(/|of|trên)\s*\d+)?", low):
            continue                                                    # số trang, đường kẻ
        if freq.get(low, 0) >= 3 and len(ln) < 90 and not ln.startswith("#"):
            continue                                                    # tiêu đề đầu/cuối trang lặp
        if re.fullmatch(r"(https?://|www\.)\S+", ln) or (BOILER.search(ln) and len(ln) < 120):
            continue
        keep.append(ln)
    paras, cur = [], []
    for ln in keep + [""]:
        if not ln:
            if cur:
                paras.append(" ".join(cur)); cur = []
            continue
        if ln.startswith("#") or (cur and (re.match(r"^([-•*]|\d+[.)])\s", ln) or len(cur[-1]) < 40 and cur[-1].startswith("#"))):
            if cur:
                paras.append(" ".join(cur)); cur = []
        cur.append(ln)
    out, seen, sh = [], set(), []
    for p in paras:
        if len(p.split()) < 3 and not p.startswith("#"):
            continue
        k = fold(p.lower())
        if k in seen:
            continue
        s = _shingles(p)
        if len(s) > 3 and any(len(s & t) / max(1, len(s | t)) > 0.8 for t in sh[-200:]):
            continue                                                    # gần trùng (Jaccard > 0.8)
        seen.add(k); sh.append(s); out.append(p)
    return out


# ============================ 3. tóm tắt ============================
def distill(paras, budget, title=""):
    """Chọn các câu giàu thông tin nhất (giữ thứ tự gốc) trong `budget` ký tự. -> (bản tóm tắt, từ khoá)."""
    sents = []
    for pi, p in enumerate(paras):
        if p.startswith("#"):
            sents.append((pi, 0, p.lstrip("# ").strip(), True))
            continue
        for si, s in enumerate(x.strip() for x in _SENT.split(p) if x.strip()):
            if len(s) >= 25:
                sents.append((pi, si, s, False))
    if not sents:
        return "", []
    toks = [[t for t in tokens(s, bigrams=False) if t.isascii() and not t.isdigit() and len(t) > 2] for _, _, s, _ in sents]
    df = collections.Counter(t for ts in toks for t in set(ts))
    n = len(sents)
    idf = {t: math.log(1 + n / c) for t, c in df.items()}
    tf = collections.Counter(t for ts in toks for t in ts)
    kw = [t for t, _ in sorted(((t, tf[t] * idf[t]) for t in tf), key=lambda x: -x[1])[:12]]
    kws = set(kw)
    scored = []
    for i, ((pi, si, s, is_h), ts) in enumerate(zip(sents, toks)):
        if is_h or not ts:
            scored.append((0.0, i)); continue
        sc = sum(idf[t] for t in set(ts)) / (len(ts) ** 0.6)
        sc *= 1 + 0.5 * len(kws & set(ts)) / 4 + (0.35 if si == 0 else 0) + (0.2 if re.search(r"\d", s) else 0) + (0.15 if pi < 3 else 0)
        scored.append((sc, i))
    chosen, used = set(), 0
    for sc, i in sorted(scored, reverse=True):
        s = sents[i][2]
        if sents[i][3] or used + len(s) + 1 > budget:
            continue
        chosen.add(i); used += len(s) + 1
    out, last_h = [], None
    for i, (pi, si, s, is_h) in enumerate(sents):
        if is_h:
            last_h = s
        elif i in chosen:
            if last_h:
                out.append(f"## {last_h}"); last_h = None
            out.append(s)
    return "\n".join(out), [(k) for k in kw]


# ============================ pipeline ============================
def _sig(path):
    st = os.stat(path)
    return [int(st.st_mtime), st.st_size]


class KB:
    def __init__(self, ws, rag=None, ratio=0.2, cap=1800, floor=300):
        self.ws, self.rag, self.ratio, self.cap, self.floor = ws, rag, ratio, cap, floor

    def _manifest(self):
        p = self.ws.find("kb.json", "index")
        try:
            return json.loads(self.ws.read_text(p)) if p else {}
        except (VaultError, OSError, ValueError):
            return {}

    def _save_manifest(self, m):
        for stale in (os.path.join(self.ws.root, "index", "kb.json"), os.path.join(self.ws.root, "index", "kb.json.enc")):
            if os.path.exists(stale):
                os.unlink(stale)
        self.ws.write_text("kb.json", json.dumps(m, ensure_ascii=False), sub="index")

    def sources(self):
        return [(p, mt, sz) for p, mt, sz in self.ws.list_files("kb", exts=KB_EXT) if not self.ws.rel(p).startswith(("kb/_clean/", "kb/_tomtat/"))]

    def ingest(self, on_progress=None, stop=None, force=False):
        """Xử lý các file mới/đổi trong kb/. Trả báo cáo. on_progress(pct, bước, tên). stop: threading.Event."""
        t0 = time.time()
        rep = {"files": 0, "skipped": 0, "errors": [], "chars_in": 0, "chars_clean": 0, "chars_summary": 0}
        man, srcs = self._manifest(), self.sources()

        def prog(i, frac, step, name):
            if on_progress:
                on_progress(min(99, int(100 * (i + frac) / max(1, len(srcs)))), step, name)

        def check():
            if stop is not None and stop.is_set():
                raise Cancelled()

        try:
            for i, (p, mt, sz) in enumerate(srcs):
                rel, name = self.ws.rel(p), os.path.basename(p).removesuffix(".enc")
                check()
                if not force and man.get(rel, {}).get("sig") == [int(mt), sz]:
                    rep["skipped"] += 1
                    continue
                try:
                    prog(i, 0.05, "đọc", name)
                    if sz > MAX_IN:
                        raise KBError("file quá lớn (tối đa 8MB)")
                    raw = extract_text(name, self.ws.read_bytes(p))
                    check(); prog(i, 0.25, "dọn", name)
                    paras = clean(raw)
                    clean_txt = "\n\n".join(paras)
                    if len(clean_txt) < 20:
                        raise KBError("không trích được chữ")
                    check(); prog(i, 0.5, "tóm tắt", name)
                    budget = int(min(self.cap, max(self.floor, len(clean_txt) * self.ratio)))
                    summ, kws = distill(paras, budget, name)
                    check(); prog(i, 0.8, "ghi", name)
                    base = slug(os.path.splitext(name)[0], "tai-lieu")
                    self.ws.write_text(f"{base}.txt", clean_txt, sub="kb/_clean")
                    head = f"# {os.path.splitext(name)[0]}\nnguồn: {name} | gốc {len(raw)} ký tự -> sạch {len(clean_txt)} -> tóm tắt {len(summ)}\ntừ khoá: {', '.join(kws)}\n\n"
                    self.ws.write_text(f"{base}.md", head + summ, sub="kb/_tomtat")
                    man[rel] = {"sig": [int(mt), sz], "name": base, "chars": [len(raw), len(clean_txt), len(summ)]}
                    rep["files"] += 1
                    rep["chars_in"] += len(raw); rep["chars_clean"] += len(clean_txt); rep["chars_summary"] += len(summ)
                except Cancelled:
                    raise
                except Exception as e:  # noqa: BLE001 - một file hỏng không được làm hỏng cả đợt nạp
                    rep["errors"].append(f"{name}: {e}")
                    self.ws.log_error("kb.ingest", f"{name}: {e}")
                    man[rel] = {"sig": [int(mt), sz], "name": "", "error": str(e)[:100]}
            for rel in [r for r in man if not os.path.exists(os.path.join(self.ws.root, r))]:   # file gốc đã xoá -> bỏ bản sạch/tóm tắt
                for sub, ext in (("kb/_clean", ".txt"), ("kb/_tomtat", ".md")):
                    for cand in (os.path.join(self.ws.root, sub, man[rel].get("name", "") + ext), os.path.join(self.ws.root, sub, man[rel].get("name", "") + ext + ".enc")):
                        if man[rel].get("name") and os.path.exists(cand):
                            os.unlink(cand)
                man.pop(rel)
            check(); prog(len(srcs), 0, "lập chỉ mục", "")
            self._save_manifest(man)
            if self.rag is not None:
                self.rag.sync()
            if on_progress:
                on_progress(100, "xong", "")
        except Cancelled:
            self._save_manifest(man)   # giữ phần đã xong để lần sau làm tiếp
            rep["cancelled"] = True
        rep["seconds"] = round(time.time() - t0, 1)
        return rep

    # ---- cho công cụ `kb` ----
    def listing(self, chars=300):
        man = self._manifest()
        items = [f"{v['name']} ({v['chars'][0] // 1000}k→{v['chars'][2]})" for v in man.values() if v.get("name")]
        return ("; ".join(items) if items else "chưa có tài liệu: chép file vào thư mục kb/ rồi chạy 'mem kb'")[:chars]

    def summary(self, name, chars=700):
        key = slug(name)
        man = self._manifest()
        cands = [v["name"] for v in man.values() if v.get("name")]
        hit = key if key in cands else next((c for c in cands if key in c or c in key), None)
        if not hit:
            return "không có tài liệu này. " + self.listing(200)
        p = self.ws.find(f"{hit}.md", "kb/_tomtat")
        txt = self.ws.read_text(p)
        body = " ".join(ln for ln in txt.splitlines() if ln.strip() and not ln.startswith(("# ", "nguồn:")))
        return f"{hit}: {' '.join(body.split())}"[:chars]


def text_bar(width=24):
    """on_progress dạng chữ cho CLI: [██████░░░░] 45% dọn - ten-file.pdf"""
    def cb(pct, step, name):
        n = int(width * pct / 100)
        print(f"\r[{'█' * n}{'░' * (width - n)}] {pct:3d}% {step:<11} {name[:30]:<30}", end="" if pct < 100 else "\n", flush=True)
    return cb
