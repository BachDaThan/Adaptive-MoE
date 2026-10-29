"""
providers.py - các nhà cung cấp LLM và nguồn cào web dùng cho make_data.py (không cần SDK, gọi thẳng REST bằng urllib).

LLM (thứ tự mặc định, đổi bằng biến LLM_ORDER, vd "cohere,openrouter,mistral,gemini"):
  Cohere -> OpenRouter -> Mistral -> Gemini
  - Provider nào không có key thì bị bỏ qua. Provider đứng trước được dùng trước; lỗi/hết quota/quá tải thì tự rơi xuống provider sau.
  - Provider đang phải chờ rate-limit lâu (> SPILL_SECONDS, mặc định 15s) thì KHÔNG đứng chờ: chuyển ngay sang provider kế,
    hết chờ thì tự quay lại provider ưu tiên cao hơn (vì make_data luôn chọn model xếp hạng cao nhất đang rảnh).
  - Model đi theo dạng  "<provider>::<model>"  (vd cohere::command-a-03-2025, openrouter::meta-llama/llama-3.3-70b-instruct:free).

Cào web (thứ tự mặc định, đổi bằng SCRAPE_ORDER): Tavily -> Exa -> Linkup -> Firecrawl
  - Nguồn nào hết key/hết hạn mức/không ra kết quả thì tự sang nguồn sau. Wikipedia vẫn xen kẽ như cũ (do make_data.py quyết).
"""
import json, random, re, time, urllib.error, urllib.request

UA = "YuiDataBot/1.0 (educational dataset builder)"


class ModelGone(Exception): pass
class Transient(Exception): pass
class ModelBusy(Exception): pass   # model/provider đang quá tải hoặc phải chờ lâu: thử model khác, quay lại sau
class Empty(Exception): pass
class QuotaOut(Exception): pass
class SearchUnavailable(Exception): pass   # model/gói này không dùng được công cụ google_search


def http(method, url, headers=None, body=None, timeout=120):
    """(status, text). Lỗi mạng -> (599, lý do). Có User-Agent riêng vì vài API sau Cloudflare chặn 'Python-urllib'."""
    req = urllib.request.Request(url, method=method, data=None if body is None else json.dumps(body).encode("utf-8"),
                                 headers={"Content-Type": "application/json", "User-Agent": UA, "Accept": "application/json", **(headers or {})})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as e:
        return e.code, e.read().decode("utf-8", "replace")
    except (urllib.error.URLError, TimeoutError, ConnectionError, OSError) as e:
        return 599, str(e)


# ============================ LLM: lớp chung cho Cohere / OpenRouter / Mistral ============================
class ChatProvider:
    name = "?"
    cool429 = 30.0          # nghỉ bao lâu (giây) sau một lần 429 thường
    per_model_429 = False   # True: 429 chỉ tính cho model đó (OpenRouter free: giới hạn phía trên theo từng model); False: giới hạn theo KEY, nghỉ cả key
    def __init__(self, keys, base, models=None, rpm=10.0, max_wait=900.0, spill=15.0):
        # "mcool": hạn mức theo TỪNG model; "cool": nghỉ cả key (vd hết quota ngày chung)
        self.keys = [{"k": k, "cool": 0.0, "last": 0.0, "dead": False, "mcool": {}} for k in keys]
        self.base, self.forced = base.rstrip("/"), list(models or [])
        self.gap, self.max_wait, self.spill = 60.0 / max(rpm, 0.1), max_wait, spill
        self.no_system, self.busy = set(), {}

    # ---- phần mỗi provider tự định nghĩa ----
    def _call(self, key, model, system, prompt, temperature, max_tokens):   # -> (status, text); system có thể None
        raise NotImplementedError

    def _extract(self, txt):
        raise NotImplementedError

    def pool(self):
        """Danh sách model theo thứ tự ưu tiên."""
        return list(self.forced)

    def _exhausted(self, low):
        """429 mang nghĩa 'key này hết hạn mức dài hạn' (tháng/tín dụng) -> bỏ key."""
        return False

    def _cool_daily(self, key, model):
        key["mcool"][model] = time.time() + 6 * 3600

    # ---- phần chung ----
    @staticmethod
    def _daily(low):
        return "perday" in re.sub(r"[\s_\-]+", "", low)

    def _pick(self, model):
        alive = [k for k in self.keys if not k["dead"]]
        if not alive:
            raise QuotaOut(f"{self.name}: mọi key đều không dùng được")
        now = time.time()
        def ready(k):
            return max(k["cool"], k["mcool"].get(model, 0.0), k["last"] + self.gap)
        best = min(alive, key=ready)
        wait = max(0.0, ready(best) - now)
        if wait > self.max_wait:
            if all(k["mcool"].get(model, 0.0) > now + self.max_wait for k in alive):
                raise ModelGone(f"{self.name}: hết quota NGÀY của model này trên mọi key -> chuyển model kế")
            raise QuotaOut(f"{self.name}: mọi key phải chờ > {int(self.max_wait)}s (hết quota)")
        if wait > self.spill:   # chờ lâu: nhường cho provider/model khác thay vì đứng đợi
            self.busy[model] = now + wait
            raise ModelBusy(model)
        return best, wait

    def generate(self, model, system, prompt, temperature=1.0, max_tokens=4096):
        trans = 0
        if self.busy.get(model, 0.0) > time.time():
            raise ModelBusy(model)
        for _ in range(12 + 3 * len(self.keys)):
            key, wait = self._pick(model)
            if wait > 0:
                if wait >= 1:
                    print(f"  ⏳ {self.name}: chờ {wait:.0f}s (giới hạn tốc độ)...", flush=True)
                time.sleep(wait)
            key["last"] = time.time()
            if model in self.no_system:
                sysmsg, user = None, system + "\n\n" + prompt
            else:
                sysmsg, user = system, prompt
            st, txt = self._call(key["k"], model, sysmsg, user, temperature, min(max_tokens, 4096))
            low = txt.lower()
            if st == 200:
                try:
                    out = self._extract(txt)
                except (ValueError, KeyError, IndexError, TypeError, AttributeError) as e:
                    raise Empty(f"{self.name}: không đọc được phản hồi ({str(e)[:60]})")
                if not out.strip():
                    raise Empty(f"{self.name}: phản hồi rỗng")
                return out
            if st == 429:
                if self._exhausted(low):
                    key["dead"] = True
                    print(f"  ❌ {self.name}: key ...{key['k'][-4:]} hết hạn mức dài hạn (429), bỏ key này", flush=True)
                    continue
                if self._daily(low):
                    self._cool_daily(key, model)
                    print(f"  ↻ 429: {self.name} key ...{key['k'][-4:]} hết quota NGÀY ({model})", flush=True)
                else:
                    if self.per_model_429:
                        key["mcool"][model] = time.time() + self.cool429
                    else:
                        key["cool"] = time.time() + self.cool429
                    print(f"  ↻ 429: {self.name} key ...{key['k'][-4:]} nghỉ {self.cool429:.0f}s" + (f" với {model}" if self.per_model_429 else ""), flush=True)
                continue
            if st in (500, 502, 503, 504, 599):
                trans += 1
                key["mcool"][model] = time.time() + min(60.0, 6.0 * 2 ** trans) * random.uniform(0.7, 1.3)
                if trans >= 3:
                    self.busy[model] = time.time() + 90.0 * random.uniform(0.8, 1.5)
                    print(f"  ⏸ {self.name}/{model} đang quá tải (HTTP {st}) -> tạm đổi model ~90s", flush=True)
                    raise ModelBusy(model)
                continue
            if st == 400:
                if any(s in low for s in ("system instruction", "systeminstruction", "developer instruction", "system message", "system role")) and model not in self.no_system:
                    self.no_system.add(model)
                    continue
                raise ModelGone(f"{self.name} 400: {txt[:160]}")
            if st == 404:
                raise ModelGone(f"{self.name}: model không tồn tại")
            if st in (401, 402, 403):
                if st == 403 and "model" in low and "key" not in low:
                    raise ModelGone(f"{self.name}: không có quyền dùng model ({st})")
                key["dead"] = True
                print(f"  ❌ {self.name}: key ...{key['k'][-4:]} bị từ chối ({st}), bỏ key này", flush=True)
                continue
            raise Transient(f"{self.name} HTTP {st}: {txt[:120]}")
        raise Transient(f"{self.name}: thử quá nhiều lần")


def _msgs(system, prompt):
    return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": prompt}]


def _content_text(c):
    """content của chat/completions có thể là chuỗi hoặc danh sách khối {'type':'text','text':...}."""
    if isinstance(c, str):
        return c
    if isinstance(c, list):
        return "".join(x.get("text", "") for x in c if isinstance(x, dict) and x.get("type", "text") == "text")
    return ""


class Cohere(ChatProvider):
    name = "cohere"
    DEFAULT_MODELS = ["command-a-03-2025", "command-r-plus-08-2024", "command-r-08-2024"]
    cool429 = 30.0

    def __init__(self, keys, base="https://api.cohere.com", models=None, **kw):
        super().__init__(keys, base, models, **kw)

    def _call(self, key, model, system, prompt, temperature, max_tokens):
        body = {"model": model, "messages": _msgs(system, prompt), "temperature": min(max(temperature, 0.0), 1.0), "max_tokens": max_tokens}
        return http("POST", self.base + "/v2/chat", {"Authorization": "Bearer " + key}, body)

    def _extract(self, txt):
        j = json.loads(txt)
        return "".join(p.get("text", "") for p in (j.get("message") or {}).get("content") or [] if isinstance(p, dict) and p.get("type", "text") == "text")

    def _exhausted(self, low):
        return "month" in low and ("trial" in low or "limit" in low)   # "Trial key ... limited to 1000 API calls / month"

    def pool(self):
        if self.forced:
            return list(self.forced)
        have = set()
        for k in self.keys:
            st, txt = http("GET", self.base + "/v2/models?endpoint=chat&page_size=200", {"Authorization": "Bearer " + k["k"]}, timeout=30)
            if st == 200:
                try:
                    have |= {m.get("name") for m in json.loads(txt).get("models", [])}
                except (ValueError, AttributeError):
                    pass
                break
        ok = [m for m in self.DEFAULT_MODELS if m in have] if have else []
        return ok or list(self.DEFAULT_MODELS)   # không đọc được danh sách -> dùng mặc định (model sai sẽ tự bị loại khi gặp 404)


class Mistral(ChatProvider):
    name = "mistral"
    DEFAULT_MODELS = ["mistral-large-latest", "mistral-medium-latest", "mistral-small-latest", "open-mistral-nemo"]
    cool429 = 10.0   # gói miễn phí ~1 yêu cầu/giây

    def __init__(self, keys, base="https://api.mistral.ai/v1", models=None, **kw):
        super().__init__(keys, base, models, **kw)

    def _call(self, key, model, system, prompt, temperature, max_tokens):
        body = {"model": model, "messages": _msgs(system, prompt), "temperature": min(max(temperature, 0.0), 1.0), "max_tokens": max_tokens}
        return http("POST", self.base + "/chat/completions", {"Authorization": "Bearer " + key}, body)

    def _extract(self, txt):
        return _content_text(json.loads(txt)["choices"][0]["message"].get("content"))

    def pool(self):
        return list(self.forced or self.DEFAULT_MODELS)


class OpenRouter(ChatProvider):
    name = "openrouter"
    # model miễn phí đổi liên tục -> tự đọc danh sách ':free' rồi xếp theo họ model quen thuộc, context dài đứng trước
    DEFAULT_MODELS = ["meta-llama/llama-3.3-70b-instruct:free", "qwen/qwen3-235b-a22b:free", "openai/gpt-oss-120b:free", "google/gemma-3-27b-it:free"]
    PREFER = ("llama-3.3-70b", "qwen3-235b", "gpt-oss-120b", "deepseek-chat", "deepseek-v3", "qwen3-next", "gemma-3-27b", "mistral-small", "qwen", "llama", "deepseek", "gemma", "mistral")
    BAN = {"embedding", "embeddings", "guard", "moderation", "vl", "vision", "image", "audio", "tts", "whisper", "ocr", "safeguard"}
    cool429 = 30.0
    per_model_429 = True

    def __init__(self, keys, base="https://openrouter.ai/api/v1", models=None, max_models=6, referer="https://github.com/BachDaThan/Adaptive-MoE", **kw):
        super().__init__(keys, base, models, **kw)
        self.max_models, self.referer = max_models, referer

    def _call(self, key, model, system, prompt, temperature, max_tokens):
        body = {"model": model, "messages": _msgs(system, prompt), "temperature": temperature, "max_tokens": max_tokens}
        st, txt = http("POST", self.base + "/chat/completions", {"Authorization": "Bearer " + key, "HTTP-Referer": self.referer, "X-Title": "Yui data maker"}, body)
        if st == 200:   # OpenRouter đôi khi trả 200 kèm {"error": {...}} khi model phía trên lỗi
            try:
                j = json.loads(txt)
            except ValueError:
                return 502, txt
            if not j.get("choices") and j.get("error"):
                code = (j["error"] or {}).get("code")
                return (code if isinstance(code, int) and code >= 400 else 502), txt
        return st, txt

    def _extract(self, txt):
        return _content_text(json.loads(txt)["choices"][0]["message"].get("content"))

    def _cool_daily(self, key, model):
        key["cool"] = time.time() + 6 * 3600   # hạn mức ngày của model miễn phí tính chung cho cả key

    def pool(self):
        if self.forced:
            return list(self.forced)
        st, txt = http("GET", self.base + "/models", timeout=30)
        data = []
        if st == 200:
            try:
                data = json.loads(txt).get("data", [])
            except (ValueError, AttributeError):
                data = []
        scored = []
        for m in data:
            mid = str(m.get("id", ""))
            pr = m.get("pricing") or {}
            free = mid.endswith(":free") or (str(pr.get("prompt")) in ("0", "0.0") and str(pr.get("completion")) in ("0", "0.0"))
            outs = (m.get("architecture") or {}).get("output_modalities") or ["text"]
            low = mid.lower()
            if not free or outs != ["text"] or set(re.split(r"[-_./:]", low)) & self.BAN or low.startswith("openrouter/"):
                continue
            rank = next((i for i, t in enumerate(self.PREFER) if t in low), len(self.PREFER))
            scored.append(((rank, -int(m.get("context_length") or 0), mid), mid))
        ranked = [mid for _, mid in sorted(scored)][: self.max_models]
        return ranked or list(self.DEFAULT_MODELS)


# ============================ bộ định tuyến: ưu tiên + rơi xuống provider sau ============================
class LLMRouter:
    SEP = "::"

    def __init__(self, order, providers, gemini=None):
        """order: tên provider theo thứ tự ưu tiên; providers: {tên: ChatProvider}; gemini: đối tượng Gemini của make_data (hoặc None)."""
        self.order = [n for n in order if n in providers or (n == "gemini" and gemini is not None)]
        self.providers, self.gemini = providers, gemini
        self.busy, self.dead, self.empty = {}, set(), {}

    @classmethod
    def split(cls, full):
        prov, sep, model = full.partition(cls.SEP)
        return (prov, model) if sep else ("gemini", full)

    def can_search(self, full):
        prov, model = self.split(full)
        return prov == "gemini" and self.gemini is not None and "gemini" not in self.dead and model not in self.gemini.no_search

    def generate(self, full, system, prompt, temperature=1.0, max_tokens=8192, json_mode=True, search=False):
        prov, model = self.split(full)
        if prov in self.dead:
            raise ModelGone(f"{prov}: đã hết key/hạn mức trong lần chạy này")
        if self.busy.get(full, 0.0) > time.time():
            raise ModelBusy(full)
        if search and not self.can_search(full):
            raise SearchUnavailable(full)
        try:
            if prov == "gemini":
                try:
                    out = self.gemini.generate(model, system, prompt, temperature, max_tokens, json_mode, search)
                except ModelBusy:
                    self.busy[full] = self.gemini.busy.get(model, time.time() + 90.0)
                    raise
                return out
            p = self.providers[prov]
            try:
                out = p.generate(model, system, prompt, temperature, max_tokens)
            except ModelBusy:
                self.busy[full] = p.busy.get(model, time.time() + 60.0)
                raise
            except Empty:
                self.empty[full] = self.empty.get(full, 0) + 1
                if self.empty[full] >= 3:   # model cứ trả rỗng (vd model suy luận hết token) -> bỏ model này
                    raise ModelGone(f"{prov}/{model}: trả về rỗng liên tục")
                raise
            self.empty[full] = 0
            return out
        except QuotaOut as e:   # provider này hết sạch key/hạn mức: bỏ cả provider, make_data sẽ rơi xuống provider sau
            self.dead.add(prov)
            raise ModelGone(f"{prov}: {e}")


# ============================ cào web: Exa / Linkup / Firecrawl (Tavily nằm trong make_data.py) ============================
def join_results(base, items, label, chars=4000, need=300):
    """items: [(tiêu đề, nội dung)] -> {'title','text','source'} hoặc None nếu quá ngắn."""
    parts = [f"{str(t or '').strip()}: {str(c).strip()}" for t, c in items if str(c or "").strip()]
    text = "\n".join(parts)[:chars]
    return {"title": base, "text": text, "source": label} if len(text) >= need else None


class KeyedSearch:
    label, DEAD = "?", (401, 402, 403)

    def __init__(self, keys, base, delay=0.3):
        self.keys = [{"k": k, "cool": 0.0, "dead": False, "n429": 0} for k in keys]
        self.base, self.delay, self.fails, self.calls = base.rstrip("/"), delay, 0, 0

    @property
    def available(self):
        return any(not k["dead"] for k in self.keys) and self.fails < 5

    def _body(self, query):
        raise NotImplementedError

    def _request(self, key, body):   # -> (status, text)
        raise NotImplementedError

    def _parse(self, txt):           # -> [(tiêu đề, nội dung)]
        raise NotImplementedError

    def fetch(self, base, query):
        body = self._body(query)
        for _ in range(len(self.keys) * 2 + 1):
            alive = [k for k in self.keys if not k["dead"] and k["cool"] <= time.time()]
            if not alive:
                break
            key = random.choice(alive)
            time.sleep(self.delay)
            self.calls += 1
            st, txt = self._request(key["k"], body)
            if st == 200:
                key["n429"] = 0
                try:
                    items = self._parse(txt)
                except (ValueError, TypeError, AttributeError, KeyError):
                    items = []
                doc = join_results(base, items, self.label)
                if doc:
                    self.fails = 0
                return doc
            if st in self.DEAD:   # key sai / hết tín dụng -> bỏ key này
                key["dead"] = True
                print(f"  ⚠ {self.label}: key ...{key['k'][-4:]} bị từ chối/hết hạn mức ({st}), bỏ key này", flush=True)
                continue
            if st == 429:
                key["cool"] = time.time() + 20
                key["n429"] += 1
                if key["n429"] >= 3:   # 429 liên tiếp = rất có thể hết hạn mức gói
                    key["dead"] = True
                    print(f"  ⚠ {self.label}: key ...{key['k'][-4:]} bị 429 liên tục, bỏ key này", flush=True)
                continue
            self.fails += 1
            break
        return None


class Exa(KeyedSearch):
    label = "Exa"

    def __init__(self, keys, base="https://api.exa.ai", delay=0.3):
        super().__init__(keys, base, delay)

    def _body(self, query):
        return {"query": query, "numResults": 5, "type": "auto", "contents": {"text": {"maxCharacters": 1200}}}

    def _request(self, key, body):
        return http("POST", self.base + "/search", {"x-api-key": key}, body, timeout=40)

    def _parse(self, txt):
        return [(r.get("title"), r.get("text")) for r in json.loads(txt).get("results", [])]


class Linkup(KeyedSearch):
    label = "Linkup"

    def __init__(self, keys, base="https://api.linkup.so/v1", delay=0.3):
        super().__init__(keys, base, delay)

    def _body(self, query):
        return {"q": query, "depth": "standard", "outputType": "searchResults"}

    def _request(self, key, body):
        return http("POST", self.base + "/search", {"Authorization": "Bearer " + key}, body, timeout=40)

    def _parse(self, txt):
        return [(r.get("name") or r.get("title"), r.get("content")) for r in json.loads(txt).get("results", [])]


class Firecrawl(KeyedSearch):
    label = "Firecrawl"

    def __init__(self, keys, base="https://api.firecrawl.dev/v1", delay=0.3):
        super().__init__(keys, base, delay)

    def _body(self, query):
        return {"query": query, "limit": 3, "lang": "vi", "country": "vn", "scrapeOptions": {"formats": ["markdown"]}}

    def _request(self, key, body):
        st, txt = http("POST", self.base + "/search", {"Authorization": "Bearer " + key}, body, timeout=60)
        if st == 400 and ("lang" in body or "country" in body):   # bản API không nhận lang/country -> thử lại không có
            st, txt = http("POST", self.base + "/search", {"Authorization": "Bearer " + key}, {k: v for k, v in body.items() if k not in ("lang", "country")}, timeout=60)
        return st, txt

    @staticmethod
    def _clean(md):
        md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md or "")             # bỏ ảnh
        md = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", md)                # [chữ](link) -> chữ
        md = re.sub(r"[#*_>`|]+", " ", md)
        return re.sub(r"\s+", " ", md).strip()[:1200]

    def _parse(self, txt):
        data = json.loads(txt).get("data", [])
        if isinstance(data, dict):   # bản v2: {"web": [...]}
            data = data.get("web", [])
        return [(r.get("title"), self._clean(r.get("markdown")) or r.get("description")) for r in data]


class WebChain:
    """Chuỗi cào web theo thứ tự ưu tiên (mặc định Tavily -> Exa -> Linkup -> Firecrawl). Cùng một truy vấn được thử lần lượt qua từng nguồn."""

    def __init__(self, sources, make_query):
        self.sources, self.make_query = list(sources), make_query

    @property
    def available(self):
        return any(s.available for s in self.sources)

    @property
    def calls(self):
        return sum(s.calls for s in self.sources)

    def doc(self, topic):
        base, query = self.make_query(topic)
        for s in self.sources:
            if not s.available:
                continue
            d = s.fetch(base, query)
            if d:
                return d
        return None
