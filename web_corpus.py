#!/usr/bin/env python3
"""
web_corpus.py - DATA_MODE=web_only: tạo data train + fine-tune CHỈ từ web, KHÔNG gọi LLM nào, KHÔNG Wikipedia, KHÔNG kho dữ liệu Hugging Face.

  make_data.py finetune --data_mode web_only --tag 1B --files 3     # Fine-tune/1B/N.txt  (cặp hội thoại rút từ cấu trúc Markdown của bài web)
  make_data.py pretrain --data_mode web_only --tag 1B --files 3     # data/1B/N.txt       (văn xuôi sạch)
  DATA_MODE=web_only DATA_KINDS=finetune,pretrain,tools make_data.py auto --plan   # chỉ in kế hoạch + ước tính credit (không cần key LLM)

Luồng: tự sinh từ khoá (SEED_THEMES x ANGLES, mã xác định) -> Tavily search lấy URL -> Tavily extract (gộp tới 20 URL/lần, markdown) -> vi_filter.keep
-> khử trùng lặp gần giống bằng MinHash+LSH (thuần Python) -> (a) Markdown ##/###/danh sách -> cặp hội thoại, (b) văn xuôi sạch.
Tavily hết key/hạn mức -> tự dự phòng sang chuỗi Exa/Linkup/Firecrawl của make_data (chỉ có đoạn trích tìm kiếm, ngắn hơn trang đầy đủ).
Mọi tài liệu đã dùng được ghi vào  <out_dir>/web_sources.jsonl  (URL, tiêu đề, ngày, SHA-256, dùng cho loại nào) và  <out_dir>/DATA_SOURCES.md  (tự sinh).

LƯU Ý PHÁP LÝ (đọc trước khi công bố model/data): bỏ Teacher-LLM chỉ loại MỘT rủi ro điều khoản. Dữ liệu web vẫn có bản quyền và điều khoản của nguồn/Tavily;
câu trả lời SFT ở đây là đoạn TRÍCH NGUYÊN VĂN từ bài gốc. Không đưa data/model vào Kho công khai khi chưa đọc điều khoản nguồn (xem docs/LEGAL_REGISTER.md §J).
"""
import hashlib
import json
import os
import random
import re
import time
import unicodedata
import urllib.error
import urllib.parse
import urllib.request

# --------------------------------------------------------------------------- cấu hình
BLOCK_DOMAINS = ["wikipedia.org", "wikimedia.org", "wikidata.org", "wiktionary.org", "huggingface.co"]
CREDIT_SEARCH, CREDIT_EXTRACT_PER_5, CREDIT_MAP_PER_10 = 1.0, 1.0, 1.0   # ước tính theo tài liệu Tavily (cơ bản); gói của bạn có thể khác
EXTRACT_BATCH = 20
URLS_PER_SEARCH = 10

SEED_THEMES = {
    "máy tính & hệ điều hành": ["cài đặt hệ điều hành", "máy tính chạy chậm", "driver card đồ họa", "Wi-Fi không kết nối được", "BIOS và UEFI",
                                "ổ cứng SSD", "sao lưu dữ liệu", "bảo mật máy tính cá nhân", "lệnh terminal cơ bản", "mạng LAN gia đình", "pin laptop",
                                "máy tính cũ cấu hình thấp"],
    "lập trình": ["học lập trình cho người mới", "Python cơ bản", "Git cơ bản", "cơ sở dữ liệu là gì", "thuật toán sắp xếp", "viết mã sạch",
                  "lập trình web", "API là gì", "kiểm thử phần mềm", "học máy cơ bản"],
    "điện thoại & ứng dụng": ["điện thoại Android bị chậm", "dung lượng bộ nhớ điện thoại", "ứng dụng chạy ngầm", "bảo mật tài khoản", "pin điện thoại",
                              "chia sẻ mạng từ điện thoại", "chuyển dữ liệu sang điện thoại mới"],
    "đời sống": ["nấu ăn đơn giản", "dọn dẹp nhà cửa", "tiết kiệm điện", "chăm sóc cây cảnh", "quản lý thời gian", "ngủ đủ giấc", "chi tiêu cá nhân",
                 "mua sắm thông minh", "sửa chữa đồ dùng gia đình", "nuôi thú cưng"],
    "học tập": ["ngữ pháp tiếng Việt", "cách học từ vựng tiếng Anh", "kỹ năng ghi chép", "phương pháp học hiệu quả", "toán lớp 9", "vật lý đại cương",
                "viết luận", "luyện thi", "kỹ năng thuyết trình", "học ngoại ngữ"],
    "khoa học": ["thiên văn học cơ bản", "hóa học đời sống", "sinh học tế bào", "biến đổi khí hậu", "năng lượng tái tạo", "vật liệu mới",
                 "cơ thể con người", "khí tượng", "địa chất Việt Nam"],
    "văn hóa & lịch sử Việt Nam": ["lịch sử Việt Nam", "lễ hội truyền thống", "ẩm thực ba miền", "di sản văn hóa", "phong tục Tết", "nhạc cụ dân tộc",
                                   "làng nghề truyền thống", "danh lam thắng cảnh"],
    "du lịch": ["kinh nghiệm du lịch bụi", "du lịch Đà Lạt", "du lịch Phú Quốc", "chuẩn bị hành lý", "đặt phòng khách sạn", "du lịch Hà Giang"],
    "tài chính cá nhân": ["lập ngân sách", "tiết kiệm hằng tháng", "lãi suất là gì", "bảo hiểm cơ bản", "lạm phát", "quỹ khẩn cấp"],
    "công nghệ & AI": ["trí tuệ nhân tạo là gì", "mô hình ngôn ngữ lớn", "quyền riêng tư dữ liệu", "điện toán đám mây", "robot", "xe điện",
                       "mạng 5G", "blockchain là gì"],
    "sức khỏe đời thường": ["tập thể dục tại nhà", "ăn uống cân bằng", "giảm căng thẳng", "vệ sinh giấc ngủ", "uống đủ nước", "tư thế ngồi làm việc"],
    "thể thao & giải trí": ["bóng đá cơ bản", "cờ vua cho người mới", "chạy bộ", "nhạc cụ cho người mới", "chơi cờ tướng", "đọc sách hiệu quả"],
}
ANGLES = ["là gì", "hướng dẫn từng bước", "mẹo hay", "lỗi thường gặp và cách khắc phục", "giải thích dễ hiểu", "so sánh ưu nhược điểm",
          "kinh nghiệm thực tế", "câu hỏi thường gặp"]

GENERIC_HEADINGS = re.compile(r"^(giới thiệu|kết luận|tổng quan|mục lục|bình luận|tham khảo|liên hệ|lời kết|tóm lại|nguồn|xem thêm|bài viết liên quan|tags?)$", re.I)
_URL = re.compile(r"(https?://|www\.)\S+|\S+@\S+\.\S+", re.I)
_IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")
_LINK = re.compile(r"\[([^\]]+)\]\([^)]*\)")
_HTML = re.compile(r"<[^>]+>")
_BOILER = re.compile(r"(đăng nhập|đăng ký|cookie|chính sách bảo mật|bản quyền|copyright|©|theo dõi chúng tôi|chia sẻ (bài viết )?(lên|trên)|bình luận|"
                     r"tin liên quan|bài viết liên quan|liên hệ quảng cáo|tải ứng dụng|xem thêm)", re.I)
_H = re.compile(r"^(#{1,4})\s+(.*\S)\s*$")
_LIST = re.compile(r"^\s*(?:[-*+•]|\d{1,2}[.)])\s+(.*\S)\s*$")
_SENT = re.compile(r"(?<=[.!?…])\s+")
VI_MARKS = set("àáảãạăằắẳẵặâầấẩẫậèéẻẽẹêềếểễệìíỉĩịòóỏõọôồốổỗộơờớởỡợùúủũụưừứửữựỳýỷỹỵđ")


# --------------------------------------------------------------------------- từ khoá
def gen_seed_queries(n, rng, extra=None):
    """n từ khoá khác nhau (chủ đề x góc nhìn), thứ tự xáo theo rng (xác định theo seed). `extra`: chủ đề riêng của bạn (list[str])."""
    topics = [p for ps in SEED_THEMES.values() for p in ps] + [str(e).strip() for e in (extra or []) if str(e).strip()]
    combos = [f"{t} {a}" for t in topics for a in ANGLES]
    rng.shuffle(combos)
    return combos[:max(0, n)]


def blocked(url, extra=()):
    host = urllib.parse.urlparse(url).netloc.lower().split(":")[0]
    return any(host == d or host.endswith("." + d) for d in list(BLOCK_DOMAINS) + [e.strip().lower() for e in extra if e.strip()])


def strip_accents(s):
    s = unicodedata.normalize("NFD", s.replace("đ", "d").replace("Đ", "D"))
    return "".join(c for c in s if unicodedata.category(c) != "Mn")


def norm_key(s):
    return re.sub(r"\W+", " ", strip_accents(str(s)).lower()).strip()


# --------------------------------------------------------------------------- MinHash + LSH (thuần Python)
class MinHashLSH:
    """Khử trùng lặp GẦN GIỐNG: chữ ký MinHash (num_perm hàm băm) trên shingle `k` từ; LSH `bands` x `rows`. add(text)->True nếu bài MỚI (không giống bài nào đã thêm)."""

    P = (1 << 61) - 1

    def __init__(self, num_perm=64, bands=16, k=5, threshold=0.8, seed=12345):
        assert num_perm % bands == 0
        self.n, self.bands, self.rows, self.k, self.threshold = num_perm, bands, num_perm // bands, k, threshold
        r = random.Random(seed)
        self.ab = [(r.randrange(1, self.P), r.randrange(0, self.P)) for _ in range(num_perm)]
        self.sigs, self.buckets, self.exact = [], [dict() for _ in range(bands)], set()

    def _shingles(self, text):
        w = re.findall(r"\w+", strip_accents(text).lower())
        if len(w) < self.k:
            return {" ".join(w)} if w else set()
        return {" ".join(w[i:i + self.k]) for i in range(len(w) - self.k + 1)}

    def signature(self, text):
        sh = self._shingles(text)
        if not sh:
            return None
        hs = [int.from_bytes(hashlib.blake2b(s.encode("utf-8"), digest_size=8).digest(), "big") for s in sh]
        return tuple(min((a * h + b) % self.P for h in hs) for a, b in self.ab)

    @staticmethod
    def jaccard(s1, s2):
        return sum(1 for a, b in zip(s1, s2) if a == b) / len(s1)

    def add(self, text):
        ex = hashlib.sha1(norm_key(text).encode("utf-8")).hexdigest()
        if ex in self.exact:
            return False
        sig = self.signature(text)
        if sig is None:
            return False
        cand = set()
        for bi in range(self.bands):
            key = sig[bi * self.rows:(bi + 1) * self.rows]
            cand.update(self.buckets[bi].get(key, ()))
        if any(self.jaccard(sig, self.sigs[c]) >= self.threshold for c in cand):
            return False
        idx = len(self.sigs)
        self.sigs.append(sig)
        for bi in range(self.bands):
            self.buckets[bi].setdefault(sig[bi * self.rows:(bi + 1) * self.rows], []).append(idx)
        self.exact.add(ex)
        return True


# --------------------------------------------------------------------------- Markdown -> sạch -> SFT / văn xuôi
def clean_markdown(md):
    md = str(md).replace("\r", "")
    md = re.sub(r"```.*?```", "", md, flags=re.S)
    md = _HTML.sub("", _LINK.sub(r"\1", _IMG.sub("", md)))
    out = []
    for ln in md.split("\n"):
        s = ln.strip()
        if not s:
            out.append("")
            continue
        if _BOILER.search(s) and len(s) < 120:
            continue
        if s.count("|") >= 3 or re.fullmatch(r"[-*_=\s]{3,}", s):
            continue
        out.append(re.sub(r"[ \t]{2,}", " ", ln.rstrip()))
    return re.sub(r"\n{3,}", "\n\n", "\n".join(out)).strip()


def split_sections(md):
    """-> [(tiêu_đề_gần_nhất, [các cấp tiêu đề], nội dung)] theo # .. ####."""
    secs, stack, body = [], [], []

    def flush():
        txt = "\n".join(body).strip()
        if txt and stack:
            secs.append((stack[-1][1], [t for _, t in stack], txt))

    for ln in md.split("\n"):
        m = _H.match(ln)
        if m:
            flush()
            body.clear()
            lvl, title = len(m.group(1)), re.sub(r"[*_`]+", "", m.group(2)).strip(" #:.")
            while stack and stack[-1][0] >= lvl:
                stack.pop()
            stack.append((lvl, title))
        else:
            body.append(ln)
    flush()
    return secs


def _strip_md(t):
    return re.sub(r"[*_`]+", "", t).strip()


def _first_sentences(text, max_chars=420, min_chars=60):
    sents = [s.strip() for s in _SENT.split(re.sub(r"\s+", " ", text)) if s.strip()]
    out = ""
    for s in sents:
        if len(out) + len(s) + 1 > max_chars:
            break
        out = (out + " " + s).strip()
    return out if len(out) >= min_chars else ""


def _items(body):
    return [_strip_md(m.group(1)) for ln in body.split("\n") if (m := _LIST.match(ln))]


Q_TEXT = ["{h} là gì vậy?", "cho mik hỏi về {h}", "giải thích giúp mik {h} với", "{h} thế nào vậy bạn?", "mik muốn biết về {h}", "nói sơ qua về {h} đi",
          "bạn ơi, {h} là sao?", "ê, {h} nghĩa là gì?"]
Q_LIST = ["các ý chính của {h} là gì?", "{h} gồm những gì vậy?", "tóm tắt {h} giúp mik", "liệt kê {h} giúp mik nha"]


def _ok_text(t):
    return bool(t) and not _URL.search(t) and "©" not in t and len(t) <= 600


def md_to_sft(md, title, rng):
    """Markdown của một bài -> [(câu hỏi, câu trả lời)]. Quy tắc xác định (không LLM); câu trả lời là đoạn trích nguyên văn từ bài (chưa xử lý bản quyền: xem cảnh báo đầu tệp)."""
    pairs = []
    t = _strip_md(title or "")
    secs = split_sections(md)
    for head, path, body in secs:
        h = _strip_md(head)
        if not 4 <= len(h) <= 80 or GENERIC_HEADINGS.match(h):
            continue
        items = _items(body)
        if len(items) >= 3:
            ans = "; ".join(f"{i + 1}) {x.rstrip('.;')}" for i, x in enumerate(items[:6]))
            q = rng.choice(Q_LIST).format(h=h.rstrip("?").lower() if not h.isupper() else h.lower())
        else:
            prose = "\n".join(ln for ln in body.split("\n") if not _LIST.match(ln))
            ans = _first_sentences(_strip_md(prose))
            q = h if h.endswith("?") else rng.choice(Q_TEXT).format(h=h.lower())
        ans = ans[:560]
        if _ok_text(ans) and len(ans) >= 60 and _ok_text(q) and ans.lower().rstrip(" .") != q.lower().rstrip(" ?."):
            pairs.append((q, ans))
    if t and 4 <= len(t) <= 60 and secs:
        first = _first_sentences(_strip_md(re.sub(r"^#.*$", "", md, flags=re.M)), 360)
        if _ok_text(first):
            pairs.append((f"{t.lower()} là gì vậy?", first))
    return pairs


def prose_paragraphs(md, min_chars=150, max_chars=1500):
    out = []
    for blk in re.split(r"\n{2,}", md):
        b = _strip_md(re.sub(r"\s*\n\s*", " ", blk))
        if len(b) < min_chars or len(b) > max_chars or b.startswith("#") or _LIST.match(blk.split("\n")[0]) or _URL.search(b):
            continue
        letters = [c for c in b if c.isalpha()]
        if not letters or sum(c in VI_MARKS for c in b.lower()) / len(letters) < 0.06:   # chưa giống tiếng Việt
            continue
        out.append(b)
    return out


# --------------------------------------------------------------------------- Tavily search -> extract (xoay key, tự bỏ key hỏng)
class TavilyWeb:
    def __init__(self, keys, base="https://api.tavily.com", delay=0.3, post=None):
        self.keys = [k if isinstance(k, dict) else {"k": k, "cool": 0.0, "dead": False} for k in keys]
        self.base, self.delay, self.fails, self.credits, self.calls = base.rstrip("/"), delay, 0, 0.0, 0
        self._post = post or self._http_post

    @property
    def available(self):
        return any(not k["dead"] for k in self.keys) and self.fails < 5

    def _http_post(self, path, key, body):
        req = urllib.request.Request(self.base + path, method="POST", data=json.dumps(body).encode("utf-8"),
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            return 599, str(e)

    def _call(self, path, body, cost):
        for _ in range(len(self.keys) * 2 + 1):
            alive = [k for k in self.keys if not k["dead"] and k["cool"] <= time.time()]
            if not alive:
                return None
            key = random.choice(alive)
            time.sleep(self.delay)
            self.calls += 1
            st, txt = self._post(path, key["k"], body)
            if st == 200:
                try:
                    res = json.loads(txt)
                except ValueError:
                    self.fails += 1
                    return None
                self.fails, self.credits = 0, self.credits + cost
                return res
            if st in (401, 403, 432, 433):
                key["dead"] = True
                print(f"  ⚠ Tavily: key ...{key['k'][-4:]} bị từ chối/hết hạn mức ({st}), bỏ key này", flush=True)
                continue
            if st == 429:
                key["cool"] = time.time() + 20
                continue
            self.fails += 1
            return None
        return None

    def search_urls(self, query, max_results=URLS_PER_SEARCH):
        res = self._call("/search", {"query": query, "search_depth": "basic", "max_results": max_results, "include_answer": False}, CREDIT_SEARCH)
        return [(r.get("url", ""), r.get("title", "")) for r in (res or {}).get("results", []) if r.get("url")]

    def extract(self, urls):
        urls = urls[:EXTRACT_BATCH]
        if not urls:
            return {}
        res = self._call("/extract", {"urls": urls, "format": "markdown", "extract_depth": "basic"}, CREDIT_EXTRACT_PER_5 * len(urls) / 5)
        return {r["url"]: r.get("raw_content") or "" for r in (res or {}).get("results", []) if r.get("url")}

    def map_site(self, url, limit=50):
        res = self._call("/map", {"url": url, "limit": limit, "max_depth": 1}, CREDIT_MAP_PER_10 * limit / 10)
        return [u for u in (res or {}).get("results", []) if isinstance(u, str)]


# --------------------------------------------------------------------------- kho tài liệu đã cào (cache dùng chung giữa finetune/pretrain, và nhật ký nguồn)
class SourceLog:
    """<out_dir>/web_sources.jsonl: mỗi dòng {url,title,at,sha256,used,md?}. 'used' = null (chưa dùng) | 'finetune' | 'pretrain' | 'rejected:<lý do>'."""

    def __init__(self, out_dir):
        self.path = os.path.join(out_dir, "web_sources.jsonl")
        os.makedirs(out_dir, exist_ok=True)
        self.rows = {}
        if os.path.exists(self.path):
            for ln in open(self.path, encoding="utf-8"):
                try:
                    r = json.loads(ln)
                    self.rows[r["url"]] = r
                except (ValueError, KeyError):
                    pass

    def has(self, url):
        return url in self.rows

    def add(self, url, title, md):
        self.rows[url] = {"url": url, "title": title, "at": time.strftime("%Y-%m-%d"), "sha256": hashlib.sha256(md.encode("utf-8")).hexdigest(), "used": None, "md": md}

    def unused(self):
        return [r for r in self.rows.values() if r.get("used") is None and r.get("md")]

    def mark(self, url, used):
        if url in self.rows:
            self.rows[url]["used"] = used
            if used.startswith("rejected"):
                self.rows[url]["md"] = ""   # bỏ nội dung bài bị loại cho nhẹ

    def save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in self.rows.values():
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        os.replace(tmp, self.path)
        lines = ["# DATA_SOURCES (tự sinh bởi web_corpus.py)", "",
                 "Nguồn web đã dùng để tạo data. **Chưa kiểm giấy phép/điều khoản từng nguồn** — đọc trước khi công bố data hoặc model huấn luyện từ chúng.", "",
                 "| URL | Tiêu đề | Ngày | SHA-256 (nội dung đã cào) | Dùng cho |", "|---|---|---|---|---|"]
        for r in self.rows.values():
            if r.get("used") and not str(r["used"]).startswith("rejected"):
                lines.append(f"| {r['url']} | {str(r.get('title', ''))[:60].replace('|', ' ')} | {r['at']} | `{r['sha256'][:16]}…` | {r['used']} |")
        with open(os.path.join(os.path.dirname(self.path), "DATA_SOURCES.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")


def title_of(md, fallback=""):
    m = re.search(r"^#\s+(.+)$", md, flags=re.M)
    return _strip_md(m.group(1)) if m else fallback


# --------------------------------------------------------------------------- điều phối
def collect(tav, chain, queries, log, need_docs, deadline=float("inf"), extra_block=(), vi_keep=None, say=print):
    """Cào thêm tới khi có `need_docs` tài liệu chưa dùng. tav: TavilyWeb hoặc None; chain: WebChain của make_data hoặc None (dự phòng, chỉ có đoạn trích)."""
    tried = 0
    for q in queries:
        if len(log.unused()) >= need_docs or time.time() > deadline:
            break
        tried += 1
        got = False
        if tav is not None and tav.available:
            urls = [(u, t) for u, t in tav.search_urls(q) if not blocked(u, extra_block) and not log.has(u)]
            titles = dict(urls)
            for i in range(0, len(urls), EXTRACT_BATCH):
                batch = [u for u, _ in urls[i:i + EXTRACT_BATCH]]
                for u, md in tav.extract(batch).items():
                    md = clean_markdown(md)
                    ok, why = (vi_keep(md) if vi_keep else (True, None))
                    log.add(u, title_of(md, titles.get(u, "")), md if ok else "")
                    if ok:
                        got = True
                    else:
                        log.mark(u, "rejected:" + str(why))
        if not got and chain is not None and chain.available:   # dự phòng: đoạn trích tìm kiếm (ngắn)
            d = chain.doc(q)
            if d and d.get("text") and not log.has("chain:" + norm_key(d.get("title", q))):
                md = clean_markdown("# " + str(d.get("title", q)) + "\n\n" + d["text"])
                ok, why = (vi_keep(md) if vi_keep else (True, None))
                log.add("chain:" + norm_key(d.get("title", q)), d.get("title", q), md if ok else "")
                if not ok:
                    log.mark("chain:" + norm_key(d.get("title", q)), "rejected:" + str(why))
        log.save()
    say(f"  🔎 đã thử {tried} từ khoá; kho còn {len(log.unused())} tài liệu chưa dùng", flush=True) if say is print else None
    return tried


def build_items(kind, log, dedup, seen, rng, per_file, files, say=print, keyfn=norm_key, carry=None, allow_partial=True):
    """Dùng các tài liệu chưa dùng để tạo mục (JSON line cho finetune, đoạn văn cho pretrain) -> danh sách các chuỗi file cần up.
    keyfn PHẢI trùng hàm tạo `seen` (make_data.norm_key khi chạy thật), nếu không câu hỏi đã có trên repo sẽ không bị bỏ trùng.
    carry: danh sách mục dư giữ lại giữa các vòng; allow_partial=False thì file thiếu mục KHÔNG được ghi (mục dư vào carry)."""
    buf, chunks = (carry if carry is not None else []), []
    for r in list(log.unused()):
        if len(chunks) >= files:
            break
        md = r["md"]
        if not dedup.add(md):
            log.mark(r["url"], "rejected:trùng gần giống")
            continue
        added = 0
        if kind == "finetune":
            for q, a in md_to_sft(md, r.get("title", ""), rng):
                k = keyfn(q)
                if not k or k in seen:
                    continue
                seen.add(k)
                buf.append(json.dumps({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}, ensure_ascii=False))
                added += 1
        else:
            for p in prose_paragraphs(md):
                k = keyfn(p[:80])
                if k in seen:
                    continue
                seen.add(k)
                buf.append(p)
                added += 1
        log.mark(r["url"], kind if added else "rejected:không rút được mục nào")
        while len(buf) >= per_file and len(chunks) < files:
            chunks.append(buf[:per_file])
            buf = buf[per_file:]
    if allow_partial and buf and len(chunks) < files and len(buf) >= max(10, per_file // 5):
        chunks.append(list(buf))
        buf.clear()
    elif carry is not None:
        carry[:] = buf
    sep = "\n" if kind == "finetune" else "\n\n"
    return [sep.join(c) + "\n" for c in chunks]


def estimate_credits(files, per_file, kind, docs_per_query=3.0):
    """Ước tính credit Tavily: mục/tài liệu ~ finetune 4, pretrain 8 (kinh nghiệm, chưa đo) -> số tài liệu -> số lượt search + extract."""
    items_per_doc = 4 if kind == "finetune" else 8
    docs = files * per_file / items_per_doc
    queries = docs / docs_per_query
    return dict(docs=int(docs), queries=int(queries + 0.999), credits=round(queries * CREDIT_SEARCH + docs / 5 * CREDIT_EXTRACT_PER_5, 1))


def run_web_only(args, md, tav=None, chain=None, say=print):
    """args: Namespace của make_data (mode finetune|pretrain). md: module make_data (Publisher, LocalRepo/HFRepo, existing_state, build_scrapers...)."""
    import vi_filter
    from data_layout import data_prefix, finetune_prefix
    kind = args.mode
    rng = random.Random(args.seed)
    md.load_kaggle_secrets()
    if tav is None and chain is None:
        chain, parts = md.build_scrapers(args)
        say("🔑 " + (", ".join(parts) or "không có key cào web"), flush=True)
        if chain is not None:
            for s in chain.sources:
                if isinstance(s, md.Tavily):
                    tav = TavilyWeb(s.keys, s.base)   # dùng chung danh sách key + trạng thái (cooldown/dead) với chuỗi cào cũ
    if tav is None and (chain is None or not chain.available):
        say("❌ web_only cần ít nhất một key cào web (TAVILY_API_KEY1.. / EXA / LINKUP / FIRECRAWL). Không tạo gì.", flush=True)
        return 0
    if args.tag:
        prefix = data_prefix(args.tag) if kind == "pretrain" else finetune_prefix(args.tag)
    else:
        prefix = "pretrain/" if kind == "pretrain" else ""
    remote = md.LocalRepo(args.repo_dir) if args.repo_dir else (None if args.no_upload else md.HFRepo(args.repo, args.repo_type))
    pub = md.Publisher(remote, args.out_dir, prefix, upload=not args.no_upload)
    seen, _ = md.existing_state(remote, prefix)
    log = SourceLog(args.out_dir)
    per_file = args.per_file or (100 if kind == "finetune" else 150)
    items_per_doc = 4 if kind == "finetune" else 8
    need_docs = max(1, int(args.files * per_file / items_per_doc))
    topics = [t.strip() for t in re.split(r"[;\n]", args.topics or "") if t.strip()]
    queries = gen_seed_queries(max(20, need_docs * 3), rng, topics)
    say(f"📁 {prefix or '(gốc repo)'} | web_only: cần ~{need_docs} tài liệu, {len(log.unused())} đã có sẵn trong kho; {len(queries)} từ khoá sẵn sàng", flush=True)
    extra_block = [x for x in re.split(r"[,\s]+", os.environ.get("WEB_BLOCK_DOMAINS", "")) if x]
    deadline = getattr(md, "DEADLINE", float("inf"))
    dedup, carry, chunks = MinHashLSH(), [], []
    for rnd in range(1, 6):   # thu thập -> tạo mục; thiếu thì thu thập thêm (tối đa 5 vòng, hết từ khoá/thời gian thì dừng)
        tried = collect(tav, chain, queries, log, max(need_docs, len(log.unused()) + 1) if rnd > 1 else need_docs, deadline, extra_block, vi_filter.keep, say)
        queries = queries[tried:]
        last = rnd == 5 or not queries or time.time() > deadline or (tried == 0 and not log.unused())
        chunks += build_items(kind, log, dedup, seen, rng, per_file, args.files - len(chunks), say, keyfn=md.norm_key, carry=carry, allow_partial=last)
        if len(chunks) >= args.files or last:
            break
        need_docs = max(2, need_docs // 2)
        say(f"  ↻ vòng {rnd}: mới có {len(chunks)}/{args.files} file, thu thập thêm...", flush=True)
    for c in chunks:
        pub.publish(c)
    log.save()
    if tav is not None:
        say(f"💳 Tavily: {tav.calls} lượt gọi, ước tính ~{tav.credits:.1f} credit", flush=True)
    say(f"\n🎉 Xong: {len(chunks)} file mới (web_only, {kind}).", flush=True)
    return len(chunks)
