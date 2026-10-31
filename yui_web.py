#!/usr/bin/env python3
"""
yui_web.py - TÌM WEB NHIỀU TẦNG cho công cụ `search`: một tầng hỏng thì tự sang tầng sau, cuối cùng luôn có câu trả lời (không bao giờ "câm").

  Tầng 1  API có key của người dùng: Tavily, Exa, Linkup, Firecrawl. Mỗi dịch vụ nhận NHIỀU key, xoay vòng (mỗi lần tìm bắt đầu từ key kế tiếp để chia đều hạn mức);
          key bị giới hạn (429/432/433) nghỉ 1-30 phút tăng dần, key sai (401/403) nghỉ 6 giờ, lỗi mạng/5xx nghỉ ngắn; hết key thì bỏ qua dịch vụ đó.
          Key đặt bằng biến môi trường  TAVILY_API_KEY1, TAVILY_API_KEY2, ... (hoặc TAVILY_API_KEYS=a,b,c; tương tự EXA_ LINKUP_ FIRECRAWL_) hoặc cấu hình web.keys.
  Tầng 2  DuckDuckGo (bản HTML, miễn phí) bằng IP của chính người dùng; có curl_cffi (pip install curl_cffi) thì giả dấu vân tay trình duyệt để ít bị chặn,
          không có thì dùng HTTP thường của yui_agent.
  Tầng 2b Brave / Bing (và Google, TẮT mặc định) đọc trang kết quả HTML bằng IP của chính người dùng, KHÔNG cần key. Lưu ý trung thực: các trang này cấm truy vấn tự động trong điều khoản
          và hay đổi giao diện/hiện CAPTCHA, nên có GIỮ NHỊP (cách nhau >= min_interval_s, trần/ngày), tự nghỉ khi bị chặn rồi nhường tầng sau. Bản parse chưa thử với trang thật.
          Có key thì nên dùng bản chính thức: Brave Search API (BRAVE_SEARCH_API_KEY1..), Google Programmable Search (GOOGLE_CSE_API_KEY1="khoá:cx"); Bing Search API đã bị Microsoft ngừng (2025).
  Tầng 3  SearXNG công cộng: danh sách web.searxng (hoặc biến YUI_SEARXNG=url1,url2) - xáo trộn, instance lỗi/khoá JSON nghỉ riêng.
  Tầng 4  Wikipedia (vi rồi en) và kho tri thức cá nhân (RAG: lịch sử chat, ghi chú, tài liệu đã nạp) - điểm tựa cuối, chạy cả khi mất mạng.
Mọi lời gọi mạng đi qua yui_agent.safe_http: chỉ https, tên miền phải nằm trong net.allow (host SearXNG do người dùng liệt kê được tự cho phép), chống SSRF/DNS rebinding.
Bản parse từng dịch vụ viết theo tài liệu công khai và đã thử bằng máy chủ giả; chưa thử với key thật của từng dịch vụ.
Kết quả có kèm tên miền nguồn (ghi công Wikipedia/CC BY-SA và để người dùng kiểm chứng). Có bộ nhớ đệm ngắn (mặc định 10 phút) để đỡ tốn hạn mức.
"""
import html as _html
import json
import os
import random
import re
import threading
import time
import urllib.parse

PROVIDERS = ("tavily", "exa", "linkup", "firecrawl", "braveapi", "gcse")
ENV = {"braveapi": "BRAVE_SEARCH", "gcse": "GOOGLE_CSE"}   # tiền tố biến môi trường nếu khác tên tầng
SCRAPERS = ("ddg", "brave", "bing", "google")
DEFAULT_ORDER = ["tavily", "exa", "linkup", "firecrawl", "braveapi", "gcse", "ddg", "brave", "bing", "searxng", "wikipedia", "kb"]   # google: bật bằng web.google=true hoặc thêm vào web.order
BASES = {"braveapi": "https://api.search.brave.com", "gcse": "https://www.googleapis.com", "brave": "https://search.brave.com", "bing": "https://www.bing.com", "google": "https://www.google.com",
         "tavily": "https://api.tavily.com", "exa": "https://api.exa.ai", "linkup": "https://api.linkup.so", "firecrawl": "https://api.firecrawl.dev",
         "ddg": "https://html.duckduckgo.com", "wikipedia": "https://vi.wikipedia.org", "wikipedia_en": "https://en.wikipedia.org"}
BROWSER_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
_LOCK = threading.Lock()
_STATE = {}      # (tầng, khoá) -> {"until": mốc monotonic, "fails": n}   dùng chung cả tiến trình
_RR = {}         # tầng -> bộ đếm xoay vòng
_CACHE = {}      # truy vấn -> (hết hạn, văn bản)
_LAST = {}       # tầng đào HTML -> mốc monotonic của lần gọi gần nhất (giữ nhịp)
_COUNT = {}      # (tầng, ngày) -> số lần đã gọi hôm nay


class TierError(Exception):
    def __init__(self, kind, msg=""):
        super().__init__(msg or kind)
        self.kind = kind   # quota | auth | error | blocked | empty


def reset_state():
    with _LOCK:
        _STATE.clear(); _RR.clear(); _CACHE.clear(); _LAST.clear(); _COUNT.clear()


# ---------------- khoá + nghỉ ----------------
def keys_for(provider, extra=None):
    out = list(extra or [])
    pat = re.compile(rf"{ENV.get(provider, provider.upper())}_API_KEYS?(_?\d+)?", re.I)
    for k, v in sorted(os.environ.items()):
        if pat.fullmatch(k) and v.strip():
            out += [x for x in re.split(r"[,;\s]+", v) if x]
    return list(dict.fromkeys(out))


def _cool(tier, key, kind):
    with _LOCK:
        s = _STATE.setdefault((tier, key), {"until": 0.0, "fails": 0})
        s["fails"] += 1
        base, cap = {"quota": (60, 1800), "auth": (21600, 21600), "blocked": (120, 900), "error": (20, 300)}.get(kind, (20, 300))
        s["until"] = time.monotonic() + min(cap, base * 2 ** min(s["fails"] - 1, 5))


def _good(tier, key):
    with _LOCK:
        _STATE.pop((tier, key), None)


def _ready(tier, key):
    with _LOCK:
        return _STATE.get((tier, key), {}).get("until", 0.0) <= time.monotonic()


def rotate(tier, keys):
    """Các key đang sẵn sàng, bắt đầu từ key kế tiếp của vòng xoay."""
    with _LOCK:
        i = _RR.get(tier, 0)
        _RR[tier] = i + 1
    n = len(keys)
    return [keys[(i + j) % n] for j in range(n) if _ready(tier, keys[(i + j) % n])] if n else []


def key_status():
    now = time.monotonic()
    return {f"{t}:{(k[:4] + '…') if k != '-' else '-'}": max(0, int(s['until'] - now)) for (t, k), s in _STATE.items() if s["until"] > now}


# ---------------- định dạng ----------------
def _domain(url):
    try:
        return (urllib.parse.urlsplit(url).hostname or "").removeprefix("www.")
    except ValueError:
        return ""


def _clean(s):
    s = _html.unescape(re.sub(r"(?s)<[^>]+>", " ", str(s or "")))
    return re.sub(r"\s+", " ", s).strip()


def fmt_results(results, chars):
    """[{title,url,snippet}] -> chuỗi gọn: 'đoạn trích (nguồn) | đoạn trích 2 (nguồn)' vừa `chars`."""
    out, left = [], chars
    for r in results:
        sn = _clean(r.get("snippet") or r.get("title"))
        if not sn:
            continue
        tail = f" ({_domain(r.get('url', ''))})" if r.get("url") else ""
        room = left if not out else left - 3
        if room < 60:
            break
        piece = sn if len(sn) + len(tail) <= room else sn[: max(20, room - len(tail) - 1)].rstrip() + "…"
        out.append(piece + tail)
        left -= len(piece) + len(tail) + 3
    return " | ".join(out)


# ---------------- phân tích từng dịch vụ ----------------
def _json(data):
    try:
        return json.loads(data.decode("utf-8", "replace"))
    except ValueError:
        raise TierError("error", "phản hồi không phải JSON")


def parse_tavily(d):
    return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")} for r in d.get("results", [])]


def parse_exa(d):
    return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("text") or r.get("summary") or (r.get("highlights") or [""])[0]} for r in d.get("results", [])]


def parse_linkup(d):
    return [{"title": r.get("name", r.get("title", "")), "url": r.get("url", ""), "snippet": r.get("content", r.get("snippet", ""))} for r in d.get("results", []) if isinstance(r, dict)]


def parse_firecrawl(d):
    data = d.get("data", [])
    data = data.get("web", []) if isinstance(data, dict) else data
    return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("description") or r.get("markdown", "")[:600]} for r in data if isinstance(r, dict)]


def parse_searxng(d):
    return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("content", "")} for r in d.get("results", [])]


_A = re.compile(r"<a\b([^>]*)>(.*?)</a>", re.S | re.I)


def parse_ddg(page):
    """Trang html.duckduckgo.com/html: các thẻ <a class=result__a> (tiêu đề + link bọc uddg) và <a class=result__snippet>."""
    if re.search(r"anomaly|captcha|unusual traffic", page, re.I) and "result__a" not in page:
        raise TierError("blocked", "DuckDuckGo yêu cầu xác minh")
    res = []
    for attrs, inner in _A.findall(page):
        cls = re.search(r'class="([^"]*)"', attrs)
        cls = cls.group(1) if cls else ""
        if "result__a" in cls:
            m = re.search(r'href="([^"]+)"', attrs)
            href = _html.unescape(m.group(1)) if m else ""
            if "uddg=" in href:
                href = urllib.parse.parse_qs(urllib.parse.urlsplit(href).query).get("uddg", [href])[0]
            elif href.startswith("//"):
                href = "https:" + href
            res.append({"title": _clean(inner), "url": href, "snippet": ""})
        elif "result__snippet" in cls and res and not res[-1]["snippet"]:
            res[-1]["snippet"] = _clean(inner)
    return res


def parse_braveapi(d):
    return [{"title": r.get("title", ""), "url": r.get("url", ""), "snippet": r.get("description", "")} for r in (d.get("web") or {}).get("results", [])]


def parse_gcse(d):
    return [{"title": r.get("title", ""), "url": r.get("link", ""), "snippet": r.get("snippet", "")} for r in d.get("items", [])]


def _unwrap_bing(href):
    """Bing bọc link kết quả trong bing.com/ck/a?...&u=a1<base64url của URL thật>."""
    href = _html.unescape(href)
    if "bing.com/ck/a" in href:
        u = urllib.parse.parse_qs(urllib.parse.urlsplit(href).query).get("u", [""])[0]
        if u.startswith("a1"):
            import base64
            try:
                return base64.urlsafe_b64decode(u[2:] + "=" * (-len(u[2:]) % 4)).decode("utf-8", "replace")
            except ValueError:
                pass
    return href


_CAPTCHA = re.compile(r"captcha|unusual traffic|not a robot|verify you are human|enable javascript|enablejs|/sorry/|consent\.google|xác minh", re.I)


def parse_bing(page):
    res = []
    for blk in re.findall(r'<li[^>]*class="[^"]*\bb_algo\b[^"]*"[^>]*>(.*?)</li>', page, re.S | re.I):
        a = re.search(r'<h2[^>]*>\s*<a[^>]+href="([^"]+)"[^>]*>(.*?)</a>', blk, re.S | re.I)
        if not a:
            continue
        sn = re.search(r'<p[^>]*>(.*?)</p>', blk, re.S | re.I) or re.search(r'class="b_caption[^"]*"[^>]*>(.*?)</div>', blk, re.S | re.I)
        res.append({"title": _clean(a.group(2)), "url": _unwrap_bing(a.group(1)), "snippet": _clean(sn.group(1)) if sn else ""})
    return res


def parse_brave(page):
    res = []
    for blk in re.split(r'(?=<div[^>]*class="[^"]*\bsnippet\b[^"]*"[^>]*data-type="web")', page)[1:]:
        a = re.search(r'<a[^>]+href="(https?://[^"]+)"[^>]*>(.*?)</a>', blk, re.S | re.I)
        if not a:
            continue
        t = re.search(r'class="[^"]*\btitle\b[^"]*"[^>]*>(.*?)</(?:div|span|h\d)>', blk, re.S | re.I)
        d = re.search(r'class="[^"]*(?:snippet-description|snippet-content|description)[^"]*"[^>]*>(.*?)</(?:div|p)>', blk, re.S | re.I)
        res.append({"title": _clean(t.group(1)) if t else _clean(a.group(2)), "url": _html.unescape(a.group(1)), "snippet": _clean(d.group(1)) if d else ""})
    return res


def parse_google(page):
    """Bản HTML cơ bản của Google: link dạng /url?q=<URL thật>&sa=...; đoạn trích là chữ ngay sau thẻ <a>."""
    res, seen = [], set()
    for m in re.finditer(r'<a[^>]+href="/url\?q=([^"&]+)[^"]*"[^>]*>(.*?)</a>(?=(.{0,700}))', page, re.S | re.I):
        url = urllib.parse.unquote(m.group(1))
        if not url.startswith("http") or "google." in _domain(url) or url in seen:
            continue
        seen.add(url)
        rest = re.split(r'<a[^>]+href="/url\?q=', m.group(3))[0]
        res.append({"title": _clean(m.group(2)), "url": url, "snippet": _clean(rest)[:300]})
    return res


SCRAPE_URL = {"brave": lambda b, q: f"{b}/search?" + urllib.parse.urlencode({"q": q, "source": "web"}),
              "bing": lambda b, q: f"{b}/search?" + urllib.parse.urlencode({"q": q, "setlang": "vi", "cc": "VN"}),
              "google": lambda b, q: f"{b}/search?" + urllib.parse.urlencode({"q": q, "hl": "vi", "num": 5, "gbv": 1})}
SCRAPE_PARSE = {"brave": parse_brave, "bing": parse_bing, "google": parse_google}


class WebSearch:
    def __init__(self, prof, kb=None, http=None, cfg=None):
        self.prof, self.kb = prof, kb
        self.cfg = dict(getattr(prof, "web", None) or {}, **(cfg or {}))
        self._http = http

    # ---- hạ tầng ----
    def http(self, method, url, body=None, headers=None, max_kb=256, timeout=8):
        if self._http:
            return self._http(method, url, body=body, headers=headers, max_kb=max_kb, timeout=timeout)
        import yui_agent as ya
        return ya.safe_http(method, url, self.wprof, body=body, headers=headers, max_kb=max_kb, timeout=timeout)

    @property
    def wprof(self):
        """Hồ sơ mạng dùng cho tìm web: như hồ sơ chính + host SearXNG mà người dùng đã liệt kê (liệt kê = đồng ý)."""
        import copy
        p = copy.copy(self.prof)
        p.net_allow = list(self.prof.net_allow) + [urllib.parse.urlsplit(u).hostname for u in self.searx_urls() if urllib.parse.urlsplit(u).hostname] if self.prof.net_allow else []
        return p

    def base(self, name):
        attr = {"wikipedia": "wiki_base", "tavily": "tavily_base"}.get(name, f"{name}_base")
        return str(self.cfg.get(f"{name}_base") or getattr(self.prof, attr, None) or BASES[name]).rstrip("/")

    def searx_urls(self):
        v = self.cfg.get("searxng") or [u for u in os.environ.get("YUI_SEARXNG", "").split(",") if u.strip()]
        return [u.strip().rstrip("/") for u in v if str(u).startswith("https://")]

    def allowed(self, url):
        import yui_agent as ya
        host = urllib.parse.urlsplit(url).hostname
        return bool(host) and ya._host_allowed(host, self.wprof)

    # ---- các tầng ----
    def _keyed(self, prov, q, n):
        keys = keys_for(prov, (self.cfg.get("keys") or {}).get(prov))
        if not keys:
            raise TierError("empty", "không có key")
        if not self.allowed(self.base(prov)):
            raise TierError("empty", "tên miền chưa được cho phép")
        last = None
        for key in rotate(prov, keys):
            try:
                st, _, data = self.http(*self._req(prov, q, n, key))
            except (ValueError, OSError) as e:
                _cool(prov, key, "error"); last = TierError("error", str(e)[:80]); continue
            if st == 200:
                try:
                    res = {"tavily": parse_tavily, "exa": parse_exa, "linkup": parse_linkup, "firecrawl": parse_firecrawl, "braveapi": parse_braveapi, "gcse": parse_gcse}[prov](_json(data))
                except TierError as e:
                    _cool(prov, key, "error"); last = e; continue
                _good(prov, key)
                if res:
                    return res
                raise TierError("empty", "không có kết quả")
            kind = "quota" if st in (429, 432, 433, 402) else "auth" if st in (401, 403) else "error"
            _cool(prov, key, kind); last = TierError(kind, f"HTTP {st}")
        raise last or TierError("quota", "mọi key đang nghỉ")

    def _req(self, prov, q, n, key):
        b, js = self.base(prov), {"Content-Type": "application/json"}
        if prov == "tavily":
            return ("POST", b + "/search", json.dumps({"query": q, "max_results": n, "search_depth": "basic"}), {**js, "Authorization": "Bearer " + key}, 128, 10)
        if prov == "exa":
            return ("POST", b + "/search", json.dumps({"query": q, "numResults": n, "contents": {"text": {"maxCharacters": 600}}}), {**js, "x-api-key": key}, 128, 10)
        if prov == "linkup":
            return ("POST", b + "/v1/search", json.dumps({"q": q, "depth": "standard", "outputType": "searchResults"}), {**js, "Authorization": "Bearer " + key}, 128, 12)
        if prov == "braveapi":
            return ("GET", b + "/res/v1/web/search?" + urllib.parse.urlencode({"q": q, "count": n}), None, {"Accept": "application/json", "X-Subscription-Token": key}, 128, 10)
        if prov == "gcse":   # khoá dạng "API_KEY:CX"
            k, _, cx = key.partition(":")
            return ("GET", b + "/customsearch/v1?" + urllib.parse.urlencode({"key": k, "cx": cx, "q": q, "num": min(n, 10)}), None, {"Accept": "application/json"}, 128, 10)
        return ("POST", b + "/v1/search", json.dumps({"query": q, "limit": n}), {**js, "Authorization": "Bearer " + key}, 128, 12)

    def _ddg(self, q, n):
        url = self.base("ddg") + "/html/?" + urllib.parse.urlencode({"q": q})
        if not self.allowed(url):
            raise TierError("empty", "tên miền chưa được cho phép")
        if not _ready("ddg", "-"):
            raise TierError("blocked", "đang nghỉ")
        try:
            default_host = url.startswith(BASES["ddg"])
            try:
                if not default_host or self._http:
                    raise ImportError
                import yui_agent as ya
                from curl_cffi import requests as cr   # tuỳ chọn: giả dấu vân tay TLS của trình duyệt
                ya.check_url(url, self.wprof)           # vẫn chặn SSRF/tên miền ngoài danh sách
                r = cr.get(url, impersonate="chrome", timeout=8, allow_redirects=False)
                st, data = r.status_code, r.content[:262144]
            except ImportError:
                st, _, data = self.http("GET", url, headers={"User-Agent": BROWSER_UA, "Accept": "text/html"}, max_kb=256, timeout=8)
        except (ValueError, OSError) as e:
            _cool("ddg", "-", "error"); raise TierError("error", str(e)[:80])
        if st in (202, 403, 429):
            _cool("ddg", "-", "blocked"); raise TierError("blocked", f"HTTP {st}")
        if st != 200:
            _cool("ddg", "-", "error"); raise TierError("error", f"HTTP {st}")
        try:
            res = parse_ddg(data.decode("utf-8", "replace"))
        except TierError:
            _cool("ddg", "-", "blocked"); raise
        if not res:
            raise TierError("empty", "không có kết quả")
        _good("ddg", "-")
        return res[:n]

    def _throttle(self, engine):
        """Giữ nhịp cho tầng đào HTML: cách nhau >= min_interval_s (mặc định 4s) và tối đa daily_cap lần/ngày (mặc định 150)."""
        gap, cap, today = float(self.cfg.get("min_interval_s", 4)), int(self.cfg.get("daily_cap", 150)), time.strftime("%Y%m%d")
        with _LOCK:
            wait = _LAST.get(engine, -1e9) + gap - time.monotonic()
            used = _COUNT.get((engine, today), 0)
            if used >= cap:
                raise TierError("blocked", "đủ số lần hôm nay")
            if wait > 3:
                raise TierError("blocked", "giữ nhịp")
            _LAST[engine] = time.monotonic() + max(0, wait)
            _COUNT[(engine, today)] = used + 1
        if wait > 0:
            time.sleep(wait + random.uniform(0, 0.4))

    def _scrape(self, engine, q, n):
        url = SCRAPE_URL[engine](self.base(engine), q)
        if not self.allowed(url):
            raise TierError("empty", "tên miền chưa được cho phép")
        if not _ready(engine, "-"):
            raise TierError("blocked", "đang nghỉ")
        self._throttle(engine)
        st, page = self._fetch_html(url, engine)
        if st in (202, 403, 429, 503) or (st in (301, 302, 303, 307) and re.search(r"sorry|consent|captcha", page, re.I)):
            _cool(engine, "-", "blocked"); raise TierError("blocked", f"HTTP {st}")
        if st != 200:
            _cool(engine, "-", "error"); raise TierError("error", f"HTTP {st}")
        res = SCRAPE_PARSE[engine](page)
        if not res:
            if _CAPTCHA.search(page):
                _cool(engine, "-", "blocked"); raise TierError("blocked", "cần xác minh/JavaScript")
            _cool(engine, "-", "error"); raise TierError("error", "không đọc được trang kết quả (giao diện có thể đã đổi)")
        _good(engine, "-")
        return res[:n]

    def _fetch_html(self, url, engine):
        try:
            try:
                if not url.startswith(BASES.get(engine, "x")) or self._http:
                    raise ImportError
                import yui_agent as ya
                from curl_cffi import requests as cr
                ya.check_url(url, self.wprof)
                r = cr.get(url, impersonate="chrome", timeout=8, allow_redirects=False, headers={"Accept-Language": "vi,en;q=0.8"})
                return r.status_code, r.content[:393216].decode("utf-8", "replace")
            except ImportError:
                st, _, data = self.http("GET", url, headers={"User-Agent": BROWSER_UA, "Accept": "text/html", "Accept-Language": "vi,en;q=0.8"}, max_kb=384, timeout=8)
                return st, data.decode("utf-8", "replace")
        except (ValueError, OSError) as e:
            _cool(engine, "-", "error"); raise TierError("error", str(e)[:80])

    def _searx(self, q, n):
        urls = [u for u in self.searx_urls() if self.allowed(u) and _ready("searxng", u)]
        if not urls:
            raise TierError("empty", "chưa có/không còn instance SearXNG")
        random.shuffle(urls)
        for u in urls[:4]:
            try:
                st, ct, data = self.http("GET", u + "/search?" + urllib.parse.urlencode({"q": q, "format": "json", "language": "vi"}), headers={"Accept": "application/json"}, max_kb=256, timeout=8)
            except (ValueError, OSError):
                _cool("searxng", u, "error"); continue
            if st != 200 or "json" not in ct.lower():   # nhiều instance công cộng khoá định dạng JSON (403)
                _cool("searxng", u, "blocked" if st in (403, 429) else "error"); continue
            try:
                res = parse_searxng(_json(data))
            except TierError:
                _cool("searxng", u, "error"); continue
            _good("searxng", u)
            if res:
                return res[:n]
        raise TierError("error", "SearXNG không trả được kết quả")

    def _wiki(self, q, n):
        chars = max(200, getattr(self.prof, "result_chars", 300) * 2)
        for which in ("wikipedia", "wikipedia_en"):
            if which == "wikipedia_en" and not self.base("wikipedia").startswith(BASES["wikipedia"]):
                continue   # base tuỳ biến (máy chủ thử/nội bộ): không tự ý hỏi thêm Wikipedia tiếng Anh
            url = self.base(which) + "/w/api.php?" + urllib.parse.urlencode({"action": "query", "format": "json", "formatversion": "2", "generator": "search", "gsrsearch": q, "gsrlimit": 1,
                                                                             "prop": "extracts", "exintro": 1, "explaintext": 1, "exchars": chars})
            if not self.allowed(url) or not _ready(which, "-"):
                continue
            try:
                st, _, data = self.http("GET", url, max_kb=128, timeout=8)
            except (ValueError, OSError):
                _cool(which, "-", "error"); continue
            if st != 200:
                _cool(which, "-", "error"); continue
            pages = _json(data).get("query", {}).get("pages", [])
            if pages and pages[0].get("extract"):
                _good(which, "-")
                return [{"title": pages[0].get("title", ""), "url": self.base(which) + "/wiki/" + urllib.parse.quote(str(pages[0].get("title", "")).replace(" ", "_")), "snippet": pages[0]["extract"]}]
        raise TierError("empty", "Wikipedia không có bài phù hợp")

    def _kb(self, q, n):
        if self.kb is None:
            raise TierError("empty", "chưa có kho")
        txt = self.kb.recall(q, chars=max(120, getattr(self.prof, "result_chars", 300)), k=n)
        if txt.startswith("không tìm thấy"):
            raise TierError("empty", "kho không có")
        return [{"title": "kho cá nhân", "url": "", "snippet": txt}]

    # ---- điều phối ----
    def search(self, query, chars=None, n=3):
        q = " ".join(str(query).split())[:200]
        if not q:
            raise ValueError("thiếu từ khoá")
        chars = chars or getattr(self.prof, "result_chars", 300)
        ttl = float(self.cfg.get("cache_s", 600))
        sig = hash(tuple(tuple(keys_for(p, (self.cfg.get("keys") or {}).get(p))) for p in PROVIDERS) + (tuple(self.searx_urls()), tuple(self.cfg.get("order") or ())))
        ck = (q.lower(), chars, sig, self.base("wikipedia"))   # đổi key/cấu hình thì không dùng nhầm kết quả cũ
        with _LOCK:
            hit = _CACHE.get(ck)
        if hit and hit[0] > time.monotonic():
            return hit[1]
        order = list(self.cfg.get("order") or DEFAULT_ORDER)
        if self.cfg.get("google") and "google" not in order:   # Google là tầng đào HTML dễ bị chặn nhất: chỉ chạy khi người dùng bật
            order.insert(order.index("bing") + 1 if "bing" in order else 0, "google")
        budget, t0, tried, why = float(self.cfg.get("budget_s", 25)), time.monotonic(), [], {}
        for tier in order:
            if time.monotonic() - t0 > budget:
                break
            try:
                if tier in PROVIDERS:
                    res = self._keyed(tier, q, n)
                elif tier == "ddg":
                    res = self._ddg(q, n)
                elif tier in ("brave", "bing", "google"):
                    res = self._scrape(tier, q, n)
                elif tier == "searxng":
                    res = self._searx(q, n)
                elif tier == "wikipedia":
                    res = self._wiki(q, n)
                elif tier == "kb":
                    res = self._kb(q, n)
                else:
                    continue
            except TierError as e:
                if e.kind != "empty" or tier in ("wikipedia",):
                    why[tier] = e.kind
                continue
            txt = fmt_results(res, chars)
            if txt:
                if tier != "kb" and ttl > 0:
                    with _LOCK:
                        if len(_CACHE) > 200:
                            _CACHE.clear()
                        _CACHE[ck] = (time.monotonic() + ttl, txt)
                return (txt if tier != "kb" else "(offline) " + txt)
        hint = ", ".join(f"{t}:{k}" for t, k in why.items())
        return "không tra được trên web lúc này" + (f" ({hint})" if hint else "") + ", bạn thử lại sau hoặc hỏi mik theo kiến thức sẵn có nha"
