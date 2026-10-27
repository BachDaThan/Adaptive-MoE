"""
make_data.py - tạo data cho Yui bằng Gemini, TỰ ĐẶT TÊN file theo số kế tiếp trên repo Hugging Face rồi TỰ UP LÊN.

  python make_data.py finetune --files 5      # tạo 5 file hội thoại: nếu repo đã có 1.txt..32.txt thì ra 33.txt..37.txt
  python make_data.py pretrain --files 3      # tạo 3 file văn bản: pretrain/1.txt, pretrain/2.txt, ... (đánh số riêng)
  python make_data.py finetune --dry_models   # chỉ quét model Gemini và in thứ hạng, không tạo gì

  - Fine-tune:  N.txt ở gốc repo, mỗi dòng {"messages":[...]}  -> train_sft.py tự quét (như cũ, không đổi gì).
  - Pretrain:   pretrain/N.txt, các đoạn văn cách nhau bằng dòng trống -> train_lm.py nạp bằng  --hf_files "gen=pretrain/"  (xem train_lm.py).
  - Số kế tiếp = số lớn nhất đang có trên repo + 1 (đọc lại danh sách ngay trước lúc up nên chạy hai nơi cùng lúc cũng không đè file nhau).
  - Key lấy từ biến môi trường GEMINI_API_KEYS (nhiều key, cách nhau dấu phẩy) hoặc GEMINI_API_KEY; KHÔNG ghi key vào code.
  - Gọi thẳng REST của Gemini (không cần SDK nên không hỏng khi Google đổi/ngưng SDK).
  - Tự quét model, chỉ giữ model chữ họ gemini, xếp phiên bản cao -> thấp; model nào lỗi/không có quota thì tự chuyển sang model kế.
  - Tự xoay key khi hết quota, tự chờ đúng thời gian server yêu cầu; lưu từng file ngay khi đủ nên đứt giữa chừng không mất.
  - Lọc: bỏ trùng với CẢ data đã có trên repo, bỏ câu quá dài, bỏ câu lộ "Gemini/Google/mô hình ngôn ngữ".
Cần: pip install huggingface_hub   (cạnh sft_format.py của repo). Up lên repo private cần HF_TOKEN có quyền write.
"""
import argparse, io, json, os, random, re, sys, time, urllib.error, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_format import load_convs  # noqa: E402  (cùng bộ đọc/lọc với train_sft.py)

API_BASE = "https://generativelanguage.googleapis.com/v1beta"

# ============================ nội dung prompt (sửa tự do) ============================
PERSONA = ('Yui là cô bạn AI dễ thương, hơi nhõng nhẽo và hay trêu đùa. Yui gọi người đối diện là "anh", tự xưng "Yui". '
           'Yui nhắn tin kiểu Gen Z Việt Nam: ngắn, tự nhiên, teencode nhẹ (hong, thui, nè, ùi, lun, hăm, zậy), thỉnh thoảng 1 emoji.')

SITUATIONS = [
    "chào hỏi, hỏi thăm một ngày của nhau", "tâm sự chuyện buồn, mệt mỏi, áp lực", "chia sẻ chuyện vui, khoe thành tích nhỏ",
    "hỏi kiến thức đơn giản (Yui trả lời ngắn và đúng; không chắc thì nói thật là không rõ)", "nhờ Yui nhắc việc, lên kế hoạch nhỏ",
    "trêu đùa qua lại", "khen hoặc chê Yui", "hỏi về Yui (sở thích, Yui là ai, Yui làm được gì)", "than phiền, cáu nhẹ, dỗi",
    "chúc buổi sáng / chúc ngủ ngon / dặn dò", "hỏi ý kiến chọn đồ ăn, phim, game, nhạc", "đố vui, chơi chữ, kể chuyện cười",
    "câu hỏi lạ hoặc quá khó (Yui đáp dí dỏm, tuyệt đối không bịa)", "nhờ giải thích khái niệm học tập / công nghệ bằng lời dễ hiểu",
    "người dùng nhắn cộc lốc, viết tắt, sai chính tả", "xin lời khuyên về học hành, công việc, tình cảm",
    "người dùng nhờ việc nguy hiểm hoặc không phù hợp (Yui từ chối nhẹ nhàng, ngắn)",
]
TOPICS = [
    "đồ ăn và nấu nướng", "game và esports", "học hành, thi cử", "công việc, deadline", "thời tiết, mùa", "âm nhạc", "phim và anime",
    "du lịch, đi chơi", "thú cưng", "sức khoẻ nhẹ nhàng, ngủ nghỉ", "bạn bè, gia đình", "công nghệ, điện thoại, máy tính", "thể thao",
    "tiền bạc, tiết kiệm", "sở thích, thói quen", "ngày lễ, Tết", "lịch sử, địa lý Việt Nam", "khoa học vui", "đọc sách, viết lách",
    "mạng xã hội, meme", "quần áo, làm đẹp", "giao thông, đi lại", "nhà cửa, dọn dẹp", "tâm trạng, cảm xúc", "chuyện học lập trình",
]

PRETRAIN_STYLES = [
    ("văn xuôi hiện đại, kể chuyện đời thường", "câu ngắn, từ thông dụng"),
    ("báo chí, bản tin đời sống", "câu rõ ràng, thông tin cụ thể"),
    ("giải thích khoa học / công nghệ cho người mới", "dễ hiểu, có ví dụ"),
    ("văn học cổ điển, ca dao, tục ngữ, thành ngữ kèm giải nghĩa", "giàu hình ảnh, từ Hán Việt vừa phải"),
    ("tuỳ bút, nghị luận xã hội", "câu phức, lập luận mạch lạc, từ ngữ trang trọng"),
    ("nhật ký, tâm sự, bài đăng mạng xã hội của người trẻ (Gen Z, 2020s)", "khẩu ngữ, teencode nhẹ, tự nhiên"),
    ("hội thoại đời thường giữa hai ba người", "xen kẽ lời thoại và dẫn truyện ngắn"),
    ("truyện ngắn, truyện cười dân gian", "có mở đầu, diễn biến, kết"),
    ("mô tả địa danh, ẩm thực, văn hoá vùng miền Việt Nam", "giàu chi tiết"),
    ("hướng dẫn từng bước (nấu ăn, sửa đồ, học tập)", "mạch lạc, tuần tự"),
    ("văn bản hành chính, thư từ, email công việc", "lịch sự, đúng thể thức"),
    ("bài bình luận phim, sách, game", "có quan điểm rõ"),
]
PRETRAIN_TOPICS = TOPICS + ["lịch sử Việt Nam qua các thời kỳ", "thiên nhiên và môi trường", "y học thường thức", "kinh tế và đời sống",
                            "giáo dục", "nghệ thuật truyền thống", "khoa học vũ trụ", "triết lý sống"]

LEAK = re.compile(r"(mô hình ngôn ngữ|\bGemini\b|\bGoogle\b|\bChatGPT\b|\bOpenAI\b|\bClaude\b|\bAnthropic\b|large language model)", re.I)


# ============================ repo (HF hoặc thư mục máy) ============================
class LocalRepo:
    def __init__(self, d):
        self.d = d
        os.makedirs(d, exist_ok=True)

    def list(self):
        out = []
        for root, _, files in os.walk(self.d):
            for f in files:
                out.append(os.path.relpath(os.path.join(root, f), self.d).replace(os.sep, "/"))
        return out

    def read(self, name):
        return open(os.path.join(self.d, name), "rb").read()

    def write(self, name, data):
        p = os.path.join(self.d, name)
        os.makedirs(os.path.dirname(p), exist_ok=True)
        with open(p, "xb") as f:   # 'x': lỗi nếu đã có file -> không bao giờ đè
            f.write(data)


class HFRepo:
    def __init__(self, repo_id, repo_type="dataset", token=None):
        from huggingface_hub import HfApi
        self.id, self.type, self.token = repo_id, repo_type, token or os.environ.get("HF_TOKEN")
        self.api = HfApi(token=self.token)

    def list(self):
        return self.api.list_repo_files(self.id, repo_type=self.type)

    def read(self, name):
        from huggingface_hub import hf_hub_download
        return open(hf_hub_download(self.id, name, repo_type=self.type, token=self.token, force_download=True), "rb").read()

    def write(self, name, data):
        if name in set(self.list()):
            raise FileExistsError(name)
        self.api.upload_file(path_or_fileobj=io.BytesIO(data), path_in_repo=name, repo_id=self.id, repo_type=self.type,
                             commit_message=f"thêm {name} (make_data.py)")


def numbers(names, prefix):
    pat = re.compile("^" + re.escape(prefix) + r"(\d+)\.txt$")
    return sorted(int(m.group(1)) for n in names for m in [pat.match(n)] if m)


def next_number(names, prefix):
    ns = numbers(names, prefix)
    return (ns[-1] if ns else 0) + 1


class Publisher:
    """Đặt tên file = số kế tiếp (đọc lại danh sách repo ngay lúc up), lưu bản sao ở máy, rồi up lên repo."""

    def __init__(self, remote, out_dir, prefix, upload=True):
        self.remote, self.out_dir, self.prefix, self.upload = remote, out_dir, prefix, upload

    def names(self):
        names = set()
        if self.remote is not None:
            try:
                names |= set(self.remote.list())
            except Exception as e:  # noqa: BLE001
                print(f"  ⚠ không đọc được danh sách repo ({str(e)[:80]}); đánh số theo thư mục máy", flush=True)
        for root, _, files in os.walk(self.out_dir):
            for f in files:
                names.add(os.path.relpath(os.path.join(root, f), self.out_dir).replace(os.sep, "/"))
        return names

    def publish(self, text):
        data = text.encode("utf-8")
        for _ in range(5):
            name = f"{self.prefix}{next_number(self.names(), self.prefix)}.txt"
            local = os.path.join(self.out_dir, name)
            os.makedirs(os.path.dirname(local), exist_ok=True)
            if os.path.exists(local):
                continue
            with open(local, "wb") as f:
                f.write(data)
            if self.upload and self.remote is not None:
                try:
                    self.remote.write(name, data)
                except FileExistsError:
                    os.remove(local)   # có người vừa up cùng số: lấy số tiếp theo
                    continue
                except Exception as e:  # noqa: BLE001
                    print(f"  ❌ up {name} lỗi: {str(e)[:100]} (bản sao vẫn ở {local})", flush=True)
                    return name
                print(f"  ⬆ đã up {name} lên repo", flush=True)
            return name
        raise RuntimeError("không đặt được tên file")


# ============================ Gemini (REST) ============================
class ModelGone(Exception): pass
class Transient(Exception): pass
class Empty(Exception): pass
class QuotaOut(Exception): pass


class Gemini:
    def __init__(self, keys, base=API_BASE, rpm=8.0, max_wait=900.0):
        self.keys = [{"k": k, "cool": 0.0, "last": 0.0, "dead": False} for k in keys]
        self.base, self.gap, self.max_wait = base.rstrip("/"), 60.0 / max(rpm, 0.1), max_wait
        self.no_system, self.no_json = set(), set()

    def _http(self, method, url, key, body=None):
        req = urllib.request.Request(url, method=method, headers={"x-goog-api-key": key, "Content-Type": "application/json"},
                                     data=None if body is None else json.dumps(body).encode("utf-8"))
        try:
            with urllib.request.urlopen(req, timeout=120) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            return 599, str(e)

    def list_models(self):
        found = {}
        for i, kk in enumerate(self.keys):
            tok, n, ok = None, 0, True
            while True:
                url = f"{self.base}/models?pageSize=1000" + (f"&pageToken={tok}" if tok else "")
                st, txt = self._http("GET", url, kk["k"])
                if st != 200:
                    print(f"  ❌ key {i + 1}: HTTP {st} {txt[:120]}", flush=True)
                    ok = False
                    break
                j = json.loads(txt)
                for m in j.get("models", []):
                    if "generateContent" in m.get("supportedGenerationMethods", []):
                        found[m["name"].replace("models/", "")] = m
                        n += 1
                tok = j.get("nextPageToken")
                if not tok:
                    break
            if ok:
                print(f"  ✅ key {i + 1}: {n} model hỗ trợ generateContent", flush=True)
        return list(found)

    def _pick(self):
        alive = [k for k in self.keys if not k["dead"]]
        if not alive:
            raise QuotaOut("mọi key đều không dùng được")
        best = min(alive, key=lambda k: max(k["cool"], k["last"] + self.gap))
        wait = max(0.0, max(best["cool"], best["last"] + self.gap) - time.time())
        if wait > self.max_wait:
            raise QuotaOut(f"mọi key phải chờ > {int(self.max_wait)}s (hết quota)")
        return best, wait

    def generate(self, model, system, prompt, temperature=1.0, max_tokens=8192, json_mode=True):
        trans = 0
        for _ in range(12 + 3 * len(self.keys)):
            key, wait = self._pick()
            if wait > 0:
                print(f"  ⏳ chờ {wait:.0f}s (giới hạn tốc độ/quota)...", flush=True)
                time.sleep(wait)
            body = {"contents": [{"role": "user", "parts": [{"text": prompt if model not in self.no_system else system + "\n\n" + prompt}]}],
                    "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}}
            if model not in self.no_system:
                body["systemInstruction"] = {"parts": [{"text": system}]}
            if json_mode and model not in self.no_json:
                body["generationConfig"]["responseMimeType"] = "application/json"
            key["last"] = time.time()
            st, txt = self._http("POST", f"{self.base}/models/{model}:generateContent", key["k"], body)
            low = txt.lower()
            if st == 200:
                j = json.loads(txt)
                parts = (((j.get("candidates") or [{}])[0].get("content") or {}).get("parts")) or []
                out = "".join(p.get("text", "") for p in parts if not p.get("thought"))
                if not out.strip():
                    raise Empty(str(j.get("promptFeedback") or (j.get("candidates") or [{}])[0].get("finishReason") or "rỗng"))
                return out
            if st == 429:
                if "limit: 0" in low:
                    raise ModelGone("model không có quota trên gói này (limit: 0)")
                m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', txt)
                daily = "perday" in low.replace(" ", "").replace("_", "") or "per day" in low
                key["cool"] = time.time() + (6 * 3600 if daily else (float(m.group(1)) + 1 if m else 30.0))
                print(f"  ↻ 429: key tạm nghỉ {'(hết quota ngày)' if daily else ''}", flush=True)
                continue
            if st in (500, 502, 503, 504, 599):
                trans += 1
                if trans > 4:
                    raise Transient(f"HTTP {st}")
                time.sleep(2 ** trans)
                continue
            if st == 400:
                if any(s in low for s in ("system instruction", "systeminstruction", "developer instruction")) and model not in self.no_system:
                    self.no_system.add(model)
                    continue
                if any(s in low for s in ("json mode", "responsemimetype", "response_mime_type")) and model not in self.no_json:
                    self.no_json.add(model)
                    continue
                raise ModelGone(f"400: {txt[:160]}")
            if st == 404:
                raise ModelGone("không tồn tại")
            if st in (401, 403):
                if "model" in low and "key" not in low:
                    raise ModelGone(f"không có quyền dùng model ({st})")
                key["dead"] = True
                print(f"  ❌ key bị từ chối ({st}), bỏ key này", flush=True)
                continue
            raise Transient(f"HTTP {st}: {txt[:120]}")
        raise Transient("thử quá nhiều lần")


BAN_TOKENS = {"exp", "embedding", "image", "vision", "audio", "tts", "asr", "imagen", "veo", "lyria", "live", "robotics", "aqa", "learnlm",
              "gemma", "thinking", "customtools", "dialog", "native", "computer", "deep", "research", "banana", "nano"}


def rank_models(names):
    """Chỉ model họ gemini; phiên bản cao trước; cùng phiên bản thì bản ổn định trước, flash > flash-lite > pro (nhanh, quota rộng)."""
    scored = []
    for n in names:
        low = n.lower()
        toks = set(re.split(r"[-_.]", low))
        if not low.startswith("gemini") or toks & BAN_TOKENS:
            continue
        m = re.match(r"^gemini-(\d+(?:\.\d+)?)", low)
        if m:
            ver = float(m.group(1))
        elif low.endswith("-latest"):
            ver = 999.0   # alias luôn trỏ tới bản mới nhất
        else:
            continue
        tier = 2 if "flash" in low and "lite" not in low else (1 if "flash" in low else 0)
        stable = 0 if not ({"preview"} & toks) else 1
        scored.append(((-ver, stable, -tier, n), n))
    return [n for _, n in sorted(scored)]


# ============================ tạo & lọc ============================
def norm_key(s):
    return re.sub(r"[^\w]+", " ", s.lower()).strip()


def strip_fence(t):
    t = t.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    return re.sub(r"\s*```$", "", t)


def finetune_prompt(n, examples):
    sit = random.sample(SITUATIONS, 3)
    topic = random.choice(TOPICS)
    shots = "\n".join(json.dumps({"messages": [{"role": "user", "content": u}, {"role": "assistant", "content": a}]}, ensure_ascii=False)
                      for u, a in examples)
    system = ("Bạn tạo dữ liệu huấn luyện cho chatbot tiếng Việt nhỏ tên Yui. " + PERSONA +
              " Chỉ trả về một mảng JSON, không giải thích.")
    prompt = (f"Tạo {n} cuộc hội thoại KHÁC NHAU giữa người dùng và Yui, xoay quanh chủ đề «{topic}» và các tình huống: "
              + "; ".join(sit) + ".\n"
              "Quy tắc:\n- Mỗi cuộc có 1 đến 3 lượt (user rồi assistant). Mỗi tin nhắn tối đa 25 từ.\n"
              "- Giọng user đa dạng (cộc lốc, viết tắt, sai chính tả nhẹ, nghiêm túc, đùa). Yui trả lời đúng ý, không lặp khuôn câu "
              "(đừng câu nào cũng mở bằng chơi chữ rồi 'nè').\n"
              "- Yui không tự nhận là mô hình ngôn ngữ, không nhắc tên công ty AI; kiến thức phải đúng, không chắc thì nói thật là không rõ.\n"
              "- Không nội dung 18+, bạo lực, chính trị, không thông tin cá nhân thật.\n"
              "Ví dụ giọng văn (đừng chép lại):\n" + shots + "\n\n"
              'Định dạng: [{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}, ...]')
    return system, prompt


def pretrain_prompt(n):
    style, level = random.choice(PRETRAIN_STYLES)
    topic = random.choice(PRETRAIN_TOPICS)
    system = "Bạn là nhà văn và biên tập viên tiếng Việt. Chỉ trả về văn bản, không lời dẫn, không markdown."
    prompt = (f"Viết {n} đoạn văn tiếng Việt RIÊNG BIỆT, mỗi đoạn 120-250 từ, thể loại: {style}; độ khó: {level}; chủ đề gợi ý: {topic} "
              "(mỗi đoạn có thể đổi góc nhìn). Nội dung phải là sáng tác gốc, không chép lại tác phẩm có bản quyền, thông tin thực tế phải đúng. "
              "Tiếng Việt chuẩn có dấu, tự nhiên. Không đánh số, không tiêu đề, không gạch đầu dòng. "
              "Các đoạn cách nhau bằng một dòng chỉ gồm ba dấu =  (===).")
    return system, prompt


def clean_pretrain(raw):
    out = []
    for p in re.split(r"\n\s*={3,}\s*\n|\n{2,}", raw):
        p = re.sub(r"[*_#>`]+", "", p).strip()
        p = re.sub(r"^\s*(đoạn\s*\d+\s*[:.)-]\s*|\d+[.)]\s+)", "", p, flags=re.I)
        p = re.sub(r"\s*\n\s*", " ", p)
        letters = [c for c in p if c.isalpha()]
        if not (250 <= len(p) <= 2600) or not letters:
            continue
        if LEAK.search(p) or sum(c.isascii() for c in letters) / len(letters) > 0.93:   # gần như không dấu = không phải tiếng Việt
            continue
        out.append(p)
    return out


def existing_state(repo, prefix):
    """Đọc mọi N.txt đang có (để bỏ trùng + lấy ví dụ giọng văn)."""
    seen, pairs = set(), []
    if repo is None:
        return seen, pairs
    try:
        names = [n for n in repo.list() if re.match("^" + re.escape(prefix) + r"\d+\.txt$", n)]
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ không đọc được repo ({str(e)[:80]})", flush=True)
        return seen, pairs
    texts = {}
    for n in names:
        try:
            texts[n] = repo.read(n).decode("utf-8", "ignore")
        except Exception:  # noqa: BLE001
            pass
    if prefix == "":
        convs, _ = load_convs(texts)
        for c in convs:
            seen.add(norm_key(c[0][0]))
            pairs.append(c[0])
    else:
        for t in texts.values():
            for p in re.split(r"\n{2,}", t):
                seen.add(norm_key(p[:80]))
    return seen, pairs


def run(args):
    random.seed(args.seed)
    keys = [k.strip() for k in re.split(r"[,\s]+", args.keys or os.environ.get("GEMINI_API_KEYS") or os.environ.get("GEMINI_API_KEY") or "") if k.strip()]
    if not keys:
        sys.exit("Thiếu key: đặt biến môi trường GEMINI_API_KEYS (nhiều key cách nhau dấu phẩy) hoặc dùng --keys")
    gem = Gemini(keys, args.api_base, args.rpm)
    print(f"🔍 Quét model Gemini ({len(keys)} key)...", flush=True)
    names = gem.list_models()
    pool = [args.model] if args.model else rank_models(names)
    print("📋 Thứ hạng model (cao → thấp):", ", ".join(pool[:8]) or "(không có)", flush=True)
    if args.dry_models or not pool:
        return
    print(f"✅ Dùng model: {pool[0]} (tự chuyển sang model kế nếu lỗi)\n", flush=True)

    prefix = "" if args.mode == "finetune" else "pretrain/"
    remote = LocalRepo(args.repo_dir) if args.repo_dir else (None if args.no_upload else HFRepo(args.repo, args.repo_type))
    pub = Publisher(remote, args.out_dir, prefix, upload=not args.no_upload)
    seen, pairs = existing_state(remote, prefix)
    print(f"📚 Data đang có trên repo: {len(seen)} mục (sẽ không tạo trùng)\n", flush=True)
    batch = args.batch if args.mode == "finetune" else max(2, args.batch // 4)
    per_file = args.per_file or (100 if args.mode == "finetune" else 150)

    done_files, buf, fails = 0, [], 0
    while done_files < args.files:
        if args.mode == "finetune":
            system, prompt = finetune_prompt(batch, random.sample(pairs, min(6, len(pairs))) if pairs else [])
        else:
            system, prompt = pretrain_prompt(batch)
        try:
            raw = gem.generate(pool[0], system, prompt, temperature=args.temperature, json_mode=(args.mode == "finetune"))
        except ModelGone as e:
            print(f"  ⚠ bỏ model {pool[0]}: {e}", flush=True)
            pool.pop(0)
            if not pool:
                print("❌ hết model để thử")
                break
            print(f"  → chuyển sang {pool[0]}", flush=True)
            continue
        except (Empty, Transient) as e:
            fails += 1
            print(f"  ⚠ lỗi tạm ({str(e)[:80]}), lần {fails}", flush=True)
            if fails >= 8:
                print("❌ lỗi liên tục, dừng (phần đã tạo vẫn được lưu)")
                break
            time.sleep(3)
            continue
        except QuotaOut as e:
            print(f"⛔ {e}. Dừng; phần đã tạo vẫn được lưu.", flush=True)
            break
        fails = 0
        added = 0
        if args.mode == "finetune":
            convs, _ = load_convs({"x": strip_fence(raw)}, max_chars=200)
            for c in convs:
                k = norm_key(c[0][0])
                if not k or k in seen or any(LEAK.search(a) for _, a in c):
                    continue
                seen.add(k)
                buf.append(json.dumps({"messages": [m for u, a in c for m in ({"role": "user", "content": u}, {"role": "assistant", "content": a})]},
                                      ensure_ascii=False))
                added += 1
        else:
            for p in clean_pretrain(raw):
                k = norm_key(p[:80])
                if k in seen:
                    continue
                seen.add(k)
                buf.append(p)
                added += 1
        print(f"  + {added} mục hợp lệ (đệm {len(buf)}/{per_file})", flush=True)
        while len(buf) >= per_file and done_files < args.files:
            chunk, buf = buf[:per_file], buf[per_file:]
            pub.publish(("\n" if args.mode == "finetune" else "\n\n").join(chunk) + "\n")
            done_files += 1
    if buf and done_files < args.files and len(buf) >= max(10, per_file // 5):
        pub.publish(("\n" if args.mode == "finetune" else "\n\n").join(buf) + "\n")
        done_files += 1
    print(f"\n🎉 Xong: {done_files} file mới.", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["finetune", "pretrain"])
    ap.add_argument("--files", type=int, default=1, help="số file cần tạo ở lần chạy này")
    ap.add_argument("--per_file", type=int, default=0, help="số hội thoại (finetune, mặc định 100) hoặc đoạn văn (pretrain, mặc định 150) mỗi file")
    ap.add_argument("--keys", default=None)
    ap.add_argument("--model", default=None, help="ép dùng một model cụ thể thay vì tự xếp hạng")
    ap.add_argument("--repo", default="BachDaThan/TrainAI-web")
    ap.add_argument("--repo_type", default="dataset")
    ap.add_argument("--repo_dir", default=None, help="dùng một thư mục làm 'repo' (để thử, không cần mạng HF)")
    ap.add_argument("--no_upload", action="store_true")
    ap.add_argument("--out_dir", default="data_out")
    ap.add_argument("--batch", type=int, default=20, help="số hội thoại mỗi lần gọi API (pretrain: batch//4 đoạn)")
    ap.add_argument("--rpm", type=float, default=8.0, help="số yêu cầu/phút cho MỖI key (gói miễn phí thường 5-15)")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--api_base", default=API_BASE)
    ap.add_argument("--dry_models", action="store_true")
    ap.add_argument("--seed", type=int, default=int(time.time()) % 100000)
    run(ap.parse_args())


if __name__ == "__main__":
    main()
