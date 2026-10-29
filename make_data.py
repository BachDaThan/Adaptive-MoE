"""
make_data.py - tạo data cho Yui bằng Cohere -> OpenRouter -> Mistral -> Gemini (provider nào lỗi/hết quota thì tự rơi xuống provider sau), TỰ ĐẶT TÊN file theo số kế tiếp trên repo Hugging Face rồi TỰ UP LÊN.

  python make_data.py finetune --files 5      # tạo 5 file hội thoại: nếu repo đã có 1.txt..32.txt thì ra 33.txt..37.txt
  python make_data.py pretrain --files 3      # tạo 3 file văn bản: pretrain/1.txt, pretrain/2.txt, ... (đánh số riêng)
  python make_data.py finetune --dry_models   # chỉ quét model của mọi provider có key và in thứ hạng, không tạo gì
  python make_data.py finetune --tag 50M --files 5   # xếp theo cỡ model: Fine-tune/50M/N.txt (data train: data/50M/N.txt); thư mục chưa có thì tự tạo. Không có --tag = kiểu cũ (gốc repo / pretrain/)
  python make_data.py check --source yui_chat.jsonl        # soi data có sẵn (độ dài, câu mở đầu lặp, dấu hiệu lỗi chính tả) trước khi train
  python make_data.py restyle --source yui_chat.jsonl --files 3   # lấy CÂU HỎI THẬT trong file cũ, nhờ Gemini viết lại câu trả lời theo giọng Yui mới -> N.txt

  - Fine-tune:  N.txt ở gốc repo, mỗi dòng {"messages":[...]}  -> train_sft.py tự quét (như cũ, không đổi gì).
  - Pretrain:   pretrain/N.txt, các đoạn văn cách nhau bằng dòng trống -> train_lm.py nạp bằng  --hf_files "gen=pretrain/"  (xem train_lm.py).
  - Số kế tiếp = số lớn nhất đang có trên repo + 1 (đọc lại danh sách ngay trước lúc up nên chạy hai nơi cùng lúc cũng không đè file nhau).
  - Key lấy từ biến môi trường GEMINI_API_KEYS (nhiều key, cách nhau dấu phẩy) hoặc GEMINI_API_KEY; KHÔNG ghi key vào code.
  - Gọi thẳng REST của Gemini (không cần SDK nên không hỏng khi Google đổi/ngưng SDK).
  - Tự quét model, chỉ giữ model chữ họ gemini, xếp phiên bản cao -> thấp; model nào lỗi/không có quota thì tự chuyển sang model kế.
  - Tự xoay key khi hết quota, tự chờ đúng thời gian server yêu cầu; lưu từng file ngay khi đủ nên đứt giữa chừng không mất.
  - Lọc: bỏ trùng với CẢ data đã có trên repo, bỏ câu quá dài, bỏ câu lộ "Gemini/Google/mô hình ngôn ngữ".
Cần: pip install huggingface_hub   (cạnh sft_format.py của repo). Up lên repo private cần HF_TOKEN có quyền write.

TỰ TÌM TRÊN INTERNET (không chỉ chắt lọc data có sẵn):
  python make_data.py finetune --web both --files 3   # mỗi lần gọi: lấy bài Wikipedia tiếng Việt (hoặc nhờ Gemini tự tra Google) rồi chắt lọc thành hội thoại Yui
  python make_data.py pretrain --web wiki --files 3   # viết lại văn bản thật thành đoạn văn mới (không chép nguyên văn)
  --web off (mặc định, như cũ) | wiki (Wikipedia vi, không cần key) | search (Gemini + công cụ google_search) | tavily (chuỗi cào web Tavily->Exa->Linkup->Firecrawl) | both
  Gemini hết quota/không hỗ trợ google_search thì tự chuyển sang Wikipedia; mạng Wikipedia lỗi thì tự chuyển sang search; cả hai lỗi thì tạo như cũ.

NHIỀU NHÀ CUNG CẤP LLM (xem providers.py): thứ tự mặc định Cohere -> OpenRouter -> Mistral -> Gemini (Gemini hay quá tải nên để cuối)
  COHERE_API_KEY1.. / OPENROUTER_API_KEY1.. / MISTRAL_API_KEY1.. / GEMINI_API_KEY1..   (mỗi loại bao nhiêu key cũng được; hoặc NAME_KEYS cách nhau dấu phẩy; có ít nhất 1 loại là chạy)
  LLM_ORDER="cohere,openrouter,mistral,gemini"   đổi thứ tự / bỏ bớt provider (provider không có key tự bị bỏ qua)
  COHERE_MODELS, OPENROUTER_MODELS, MISTRAL_MODELS (cách nhau dấu phẩy) ép danh sách model; bỏ trống = tự chọn (OpenRouter tự quét model ':free'). GEMINI_MODEL ép model Gemini.
  COHERE_RPM / OPENROUTER_RPM / MISTRAL_RPM  số yêu cầu/phút MỖI key (mặc định 10/10/20; Gemini dùng GEMINI_RPM)
  LLM_MODEL=provider::model          ép đúng 1 model (tắt chuỗi dự phòng), vd cohere::command-a-03-2025
  SPILL_SECONDS                      provider phải chờ rate-limit lâu hơn ngần này giây (mặc định 15) thì nhường provider kế thay vì đứng đợi
CÀO WEB (cho --web tavily / both): thứ tự mặc định Tavily -> Exa -> Linkup -> Firecrawl, đổi bằng SCRAPE_ORDER
  TAVILY_API_KEY1.. / EXA_API_KEY1.. / LINKUP_API_KEY1.. / FIRECRAWL_API_KEY1..   (nguồn nào hết key/hết hạn mức/không ra kết quả thì tự sang nguồn sau)

CHẾ ĐỘ TỰ ĐỘNG - chỉnh hết bằng BIẾN MÔI TRƯỜNG (Secrets/Variables của Space, Colab, GitHub Actions...):
  python make_data.py auto            # đọc biến, so với data ĐANG CÓ trên repo, chỉ tạo phần CÒN THIẾU rồi up lên (chạy lại bao nhiêu lần cũng được)
  python make_data.py auto --plan     # chỉ in kế hoạch, không gọi Gemini
  HF_TOKEN, GEMINI_API_KEYS           khoá (nhiều key Gemini cách nhau dấu phẩy)
  HF_REPO                             repo dataset (mặc định BachDaThan/TrainAI-web); file xếp vào Fine-tune/<cỡ>/N.txt và data/<cỡ>/N.txt, <cỡ> tự tính từ số tham số (50M, 120M...) hoặc đặt MODEL_TAG
  DATA_KINDS                          finetune,pretrain (mặc định cả hai). Thêm ",tools" để tạo cả hội thoại DẠY DÙNG CÔNG CỤ (calc/time/search, không tốn quota; TOOL_PAIRS = số hội thoại, mặc định 10% mục tiêu fine-tune).
                                      CHỈ bật khi bot/app của bạn chạy qua yui_agent.py (thực thi công cụ), không thì model sẽ in thẻ <tool> thô ra cho người dùng.
  MODEL_PARAMS                        số tham số model, vd 50M / 1.2B / 50000000  -> hệ thống tự tính lượng data hợp với model
     hoặc MODEL_REPO (repo HF có config.json của model, vd BachDaThan/TrainAI)  hoặc MODEL_D, MODEL_LAYERS, MODEL_EXPERTS, MODEL_VOCAB
  FT_PAIRS                            TỔNG số cặp hội thoại fine-tune muốn có trên repo (bỏ trống = tự tính theo model); FT_NEW_PAIRS = tạo THÊM đúng ngần này
  FT_PER_FILE                         số cặp mỗi file N.txt (mặc định 100)
  PRETRAIN_TOKENS                     TỔNG token văn bản train muốn có (bỏ trống = tự tính); PRETRAIN_NEW_TOKENS = tạo thêm; KHÔNG cần đặt "cặp" cho data train
  WEB_SEARCH                          off | wiki | search | tavily | both (mặc định both: Tavily cào + Wikipedia xen kẽ, Gemini chỉ lọc/chắt lọc; google_search của Gemini chỉ là dự phòng cuối)
  WEB_TOPICS                          chủ đề riêng, cách nhau dấu ; (bỏ trống = dùng danh sách có sẵn + bài Wikipedia ngẫu nhiên)
  MAX_FILES_PER_RUN                   chặn trên số file tạo mỗi lần chạy (mặc định 40), GEMINI_RPM, GEMINI_MODEL
"""
import argparse, io, json, math, os, random, re, subprocess, sys, time, urllib.error, urllib.parse, urllib.request

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_format import load_convs, clean  # noqa: E402  (cùng bộ đọc/lọc với train_sft.py)
from data_layout import params_from_arch, size_tag, finetune_prefix, data_prefix  # noqa: E402
from providers import (ModelGone, Transient, ModelBusy, Empty, QuotaOut, SearchUnavailable, LLMRouter, Cohere, OpenRouter, Mistral,  # noqa: E402,F401
                       Exa, Linkup, Firecrawl, WebChain)

def collect_keys(name, extra=None):
    """Gom key từ: --keys, NAME_S (nhiều key cách nhau dấu phẩy/khoảng trắng), NAME, và NAME1, NAME2, NAME_3, ... (bao nhiêu cũng được). Bỏ trùng, giữ thứ tự."""
    out = []
    def add(v):
        for k in re.split(r"[,;\s]+", v or ""):
            if k.strip() and k.strip() not in out:
                out.append(k.strip())
    add(extra)
    add(os.environ.get(name + "S")); add(os.environ.get(name))
    numbered = []
    for k, v in os.environ.items():
        m = re.fullmatch(re.escape(name) + r"S?_?(\d+)", k)
        if m and v.strip():
            numbered.append((int(m.group(1)), v))
    for _, v in sorted(numbered):
        add(v)
    return out


KEY_BASES = ("COHERE_API_KEY", "OPENROUTER_API_KEY", "MISTRAL_API_KEY", "GEMINI_API_KEY", "TAVILY_API_KEY", "EXA_API_KEY", "LINKUP_API_KEY", "FIRECRAWL_API_KEY")


def load_kaggle_secrets(names=("HF_TOKEN",) + tuple(b + s for b in KEY_BASES for s in ("S", "")), numbered=KEY_BASES, limit=30):
    """Trên Kaggle: tự lấy secret cùng tên vào biến môi trường (không cần viết os.environ[...] = ... cho từng key). Ngoài Kaggle thì không làm gì."""
    try:
        from kaggle_secrets import UserSecretsClient
        cli = UserSecretsClient()
    except Exception:  # noqa: BLE001
        return 0
    got = 0
    def fetch(n):
        nonlocal got
        if os.environ.get(n):
            return True
        try:
            v = cli.get_secret(n)
        except Exception:  # noqa: BLE001
            return False
        if v:
            os.environ[n] = v; got += 1
            return True
        return False
    for n in names:
        fetch(n)
    for base in numbered:
        miss = 0
        for i in range(1, limit + 1):   # dừng khi thiếu 2 số liên tiếp
            if fetch(f"{base}{i}") or fetch(f"{base}_{i}"):
                miss = 0
            else:
                miss += 1
                if miss >= 2:
                    break
    return got


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

LEAK = re.compile(r"(mô hình ngôn ngữ|\bGemini\b|\bGoogle\b|\bChatGPT\b|\bOpenAI\b|\bClaude\b|\bAnthropic\b|\bCohere\b|\bMistral\b|\bOpenRouter\b|\bLlama\b|\bQwen\b|\bDeepSeek\b|large language model|wikipedia|theo tài liệu|tài liệu trên|nguồn tham khảo)", re.I)


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
# (ModelGone, Transient, ModelBusy, Empty, QuotaOut, SearchUnavailable: định nghĩa trong providers.py)


class Gemini:
    def __init__(self, keys, base=API_BASE, rpm=8.0, max_wait=900.0):
        # "mcool": hạn mức theo TỪNG model (quota Gemini tính riêng cho mỗi model) -> hết quota 1 model không làm cả key phải nghỉ
        self.keys = [{"k": k, "cool": 0.0, "last": 0.0, "dead": False, "mcool": {}} for k in keys]
        self.base, self.gap, self.max_wait = base.rstrip("/"), 60.0 / max(rpm, 0.1), max_wait
        self.no_system, self.no_json, self.no_search = set(), set(), set()
        self.busy = {}   # model -> thời điểm hết quá tải

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

    def _pick(self, model=None):
        alive = [k for k in self.keys if not k["dead"]]
        if not alive:
            raise QuotaOut("mọi key đều không dùng được")
        def ready(k):
            return max(k["cool"], k["mcool"].get(model, 0.0), k["last"] + self.gap)
        best = min(alive, key=ready)
        wait = max(0.0, ready(best) - time.time())
        if wait > self.max_wait:
            if model is not None and any(k["mcool"].get(model, 0.0) > time.time() + self.max_wait for k in alive) and \
                    all(k["mcool"].get(model, 0.0) > time.time() + self.max_wait for k in alive):
                raise ModelGone("hết quota NGÀY của model này trên mọi key -> chuyển model kế")
            raise QuotaOut(f"mọi key phải chờ > {int(self.max_wait)}s (hết quota)")
        return best, wait

    def generate(self, model, system, prompt, temperature=1.0, max_tokens=8192, json_mode=True, search=False):
        trans = 0
        if self.busy.get(model, 0.0) > time.time():
            raise ModelBusy(model)
        if search and model in self.no_search:
            raise SearchUnavailable(model)
        for _ in range(12 + 3 * len(self.keys)):
            key, wait = self._pick(model)
            if wait > 0:
                print(f"  ⏳ chờ {wait:.0f}s (giới hạn tốc độ/quota)...", flush=True)
                time.sleep(wait)
            body = {"contents": [{"role": "user", "parts": [{"text": prompt if model not in self.no_system else system + "\n\n" + prompt}]}],
                    "generationConfig": {"temperature": temperature, "maxOutputTokens": max_tokens}}
            if model not in self.no_system:
                body["systemInstruction"] = {"parts": [{"text": system}]}
            if search:   # Gemini tự tra Google (grounding). Không dùng chung được với chế độ JSON -> prompt tự đòi JSON, phía gọi tự tách
                body["tools"] = [{"google_search": {}}]
            elif json_mode and model not in self.no_json:
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
            if search and st in (400, 403, 404, 429) and any(w in low for w in ("google_search", "googlesearch", "grounding", "search tool")):
                self.no_search.add(model)
                print(f"  ⚠ {model} không dùng được google_search ({st}); chuyển sang nguồn Wikipedia", flush=True)
                raise SearchUnavailable(model)
            if st == 429:
                if "limit: 0" in low:
                    raise ModelGone("model không có quota trên gói này (limit: 0)")
                m = re.search(r'"retryDelay"\s*:\s*"(\d+(?:\.\d+)?)s"', txt)
                daily = "perday" in low.replace(" ", "").replace("_", "") or "per day" in low
                key["mcool"][model] = time.time() + (6 * 3600 if daily else (float(m.group(1)) + 1 if m else 30.0))
                if daily:
                    print(f"  ↻ 429: key ...{key['k'][-4:]} hết quota NGÀY của {model} (các model khác trên key này vẫn dùng được)", flush=True)
                else:
                    print(f"  ↻ 429: key ...{key['k'][-4:]} nghỉ {float(m.group(1)) + 1 if m else 30:.0f}s với {model}", flush=True)
                continue
            if st in (500, 502, 503, 504, 599):
                # 503 = model đang quá tải ở phía Google (không phải lỗi key): nghỉ có xáo trộn, thử key khác; 3 lần liền thì coi model đang bận -> đổi model
                trans += 1
                key["mcool"][model] = time.time() + min(60.0, 6.0 * 2 ** trans) * random.uniform(0.7, 1.3)
                if trans >= 3:
                    self.busy[model] = time.time() + 90.0 * random.uniform(0.8, 1.5)
                    print(f"  ⏸ {model} đang quá tải (HTTP {st}) -> tạm đổi model ~90s", flush=True)
                    raise ModelBusy(model)
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
    """Chỉ model họ gemini. Xếp phiên bản CAO -> THẤP (so từng số: 3.10 > 3.8); cùng phiên bản thì bản ổn định trước bản preview, flash > flash-lite > pro
    (nhanh, quota rộng). Các tên bí danh '-latest' chỉ để dự phòng nên xếp CUỐI (trước đây chúng bị xếp đầu nên thứ hạng trông lộn xộn)."""
    scored = []
    for n in names:
        low = n.lower()
        toks = set(re.split(r"[-_.]", low))
        if not low.startswith("gemini") or toks & BAN_TOKENS:
            continue
        m = re.match(r"^gemini-(\d+(?:\.\d+)*)", low)
        if m:
            ver, alias = tuple(int(x) for x in m.group(1).split(".")), 0
        elif low.endswith("-latest"):
            ver, alias = (0,), 1
        else:
            continue
        tier = 2 if "flash" in low and "lite" not in low else (1 if "flash" in low else 0)
        stable = 0 if "preview" not in toks else 1
        neg = tuple(-x for x in (ver + (0, 0, 0))[:3])
        scored.append(((alias, neg, stable, -tier, n), n))
    return [n for _, n in sorted(scored)]


# ============================ dựng bộ LLM + chuỗi cào web từ biến môi trường ============================
LLM_DEFAULT_ORDER = "cohere,openrouter,mistral,gemini"
SCRAPE_DEFAULT_ORDER = "tavily,exa,linkup,firecrawl"
LLM_KEY_NAMES = {"cohere": "COHERE_API_KEY", "openrouter": "OPENROUTER_API_KEY", "mistral": "MISTRAL_API_KEY", "gemini": "GEMINI_API_KEY"}
LLM_LABELS = {"cohere": "Cohere", "openrouter": "OpenRouter", "mistral": "Mistral", "gemini": "Gemini"}
LLM_RPM = {"cohere": 10.0, "openrouter": 10.0, "mistral": 20.0}   # số yêu cầu/phút MỖI key (an toàn hơn mức tối đa của gói miễn phí)


def _split_order(value, default, valid, what):
    out = []
    for n in re.split(r"[,;\s]+", (value or default).lower()):
        if not n or n in out:
            continue
        if n not in valid:
            print(f"  ⚠ {what}: bỏ qua tên lạ '{n}' (hợp lệ: {', '.join(valid)})", flush=True)
            continue
        out.append(n)
    return out


def _models_env(name):
    return [m.strip() for m in re.split(r"[,;\n]+", env(name, "") or "") if m.strip()]


def llm_key_total():
    return sum(len(collect_keys(n)) for n in LLM_KEY_NAMES.values())


def total_rpm(gemini_rpm=8.0):
    """Thông lượng gộp (yêu cầu/phút) của mọi provider đang có key - để ước lượng thời gian tạo data."""
    return sum(len(collect_keys(LLM_KEY_NAMES[n])) * (gemini_rpm if n == "gemini" else LLM_RPM[n]) for n in LLM_KEY_NAMES)


def build_llm(args):
    """-> (LLMRouter hoặc None nếu không có key nào, danh sách mô tả '2 key Gemini'...). Thứ tự ưu tiên theo LLM_ORDER / --llm_order."""
    order = _split_order(args.llm_order, LLM_DEFAULT_ORDER, list(LLM_KEY_NAMES), "LLM_ORDER")
    spill = float(env("SPILL_SECONDS", "15"))
    provs, gemini, parts = {}, None, []
    for name in order:
        keys = collect_keys(LLM_KEY_NAMES[name], args.keys if name == "gemini" else None)
        if not keys:
            continue
        parts.append(f"{len(keys)} key {LLM_LABELS[name]}")
        rpm = float(env(name.upper() + "_RPM", str(LLM_RPM.get(name, 8))))
        if name == "gemini":
            gemini = Gemini(keys, args.api_base, args.rpm)
        elif name == "cohere":
            provs[name] = Cohere(keys, env("COHERE_BASE", "https://api.cohere.com"), _models_env("COHERE_MODELS"), rpm=rpm, spill=spill)
        elif name == "openrouter":
            provs[name] = OpenRouter(keys, env("OPENROUTER_BASE", "https://openrouter.ai/api/v1"), _models_env("OPENROUTER_MODELS"), rpm=rpm, spill=spill)
        elif name == "mistral":
            provs[name] = Mistral(keys, env("MISTRAL_BASE", "https://api.mistral.ai/v1"), _models_env("MISTRAL_MODELS"), rpm=rpm, spill=spill)
    if not provs and gemini is None:
        return None, parts
    return LLMRouter(order, provs, gemini), parts


def build_pool(router, forced=None):
    """Danh sách model theo thứ tự ưu tiên: provider đứng trước, trong từng provider model tốt hơn đứng trước. Gemini tự quét + xếp hạng như cũ."""
    if forced:   # --model: "provider::model", hoặc tên trần = model Gemini (như trước đây)
        full = forced if LLMRouter.SEP in forced else "gemini" + LLMRouter.SEP + forced
        if LLMRouter.split(full)[0] not in router.order:
            sys.exit(f"--model/LLM_MODEL '{forced}' thuộc provider chưa có key (đang có: {', '.join(router.order)})")
        return [full]
    pool = []
    for name in router.order:
        if name == "gemini":
            print(f"🔍 Quét model Gemini ({len(router.gemini.keys)} key)...", flush=True)
            ms = _models_env("GEMINI_MODEL") or rank_models(router.gemini.list_models())
        else:
            ms = router.providers[name].pool()
        pool += [name + LLMRouter.SEP + m for m in ms]
    return pool


def build_scrapers(args):
    """-> (WebChain hoặc None, danh sách mô tả '2 key Tavily'...). Thứ tự theo SCRAPE_ORDER / --scrape_order."""
    order = _split_order(args.scrape_order, SCRAPE_DEFAULT_ORDER, ["tavily", "exa", "linkup", "firecrawl"], "SCRAPE_ORDER")
    sources, parts = [], []
    for name in order:
        keys = collect_keys(name.upper() + "_API_KEY", getattr(args, name + "_keys", None))
        if not keys:
            continue
        parts.append(f"{len(keys)} key {name.capitalize()}")
        if name == "tavily":
            sources.append(Tavily(keys, args.tavily_base))
        elif name == "exa":
            sources.append(Exa(keys, env("EXA_BASE", "https://api.exa.ai")))
        elif name == "linkup":
            sources.append(Linkup(keys, env("LINKUP_BASE", "https://api.linkup.so/v1")))
        elif name == "firecrawl":
            sources.append(Firecrawl(keys, env("FIRECRAWL_BASE", "https://api.firecrawl.dev/v1")))
    if "tavily" not in order or not collect_keys("TAVILY_API_KEY", args.tavily_keys):
        parts.insert(0, "không có key Tavily")
    return (WebChain(sources, make_query) if sources else None), parts


# ============================ tạo & lọc ============================
def norm_key(s):
    return re.sub(r"[^\w]+", " ", s.lower()).strip()


def strip_fence(t):
    t = t.strip()
    t = re.sub(r"^```(?:json)?\s*", "", t)
    return re.sub(r"\s*```$", "", t)



# ============================ internet: Wikipedia tiếng Việt + Gemini google_search ============================
class Wiki:
    """Lấy văn bản thật từ Wikipedia tiếng Việt (API công khai, không cần key). Dùng để chắt lọc thành data mới, KHÔNG up nguyên văn."""
    CUT = re.compile(r"\n=+\s*(Tham khảo|Liên kết ngoài|Chú thích|Xem thêm|Ghi chú|Thư mục)\s*=+.*", re.S | re.I)

    def __init__(self, base="https://vi.wikipedia.org", delay=0.6):
        self.base, self.delay, self.fails, self.warned = base.rstrip("/"), delay, 0, False

    def _get(self, params):
        url = f"{self.base}/w/api.php?" + urllib.parse.urlencode({**params, "format": "json", "formatversion": "2"})
        req = urllib.request.Request(url, headers={"User-Agent": "YuiDataBot/1.0 (educational dataset builder)", "Accept": "application/json"})
        time.sleep(self.delay)
        with urllib.request.urlopen(req, timeout=30) as r:
            return json.loads(r.read().decode("utf-8", "replace"))

    def titles(self, topic, n=6):
        if topic:
            j = self._get({"action": "query", "list": "search", "srsearch": topic, "srlimit": n, "srnamespace": 0})
            return [x["title"] for x in j.get("query", {}).get("search", [])]
        j = self._get({"action": "query", "list": "random", "rnlimit": n, "rnnamespace": 0})
        return [x["title"] for x in j.get("query", {}).get("random", [])]

    def text(self, title, chars=4000):
        j = self._get({"action": "query", "prop": "extracts", "explaintext": 1, "exchars": chars, "redirects": 1, "titles": title})
        pages = j.get("query", {}).get("pages", [])
        t = (pages[0].get("extract") if pages else "") or ""
        t = self.CUT.sub("", t)
        t = re.sub(r"\n=+[^\n]*=+\s*\n", "\n", t)          # bỏ dòng tiêu đề mục
        t = re.sub(r"\[\d+\]|\[cần dẫn nguồn\]", "", t)
        t = re.sub(r"\n{2,}", "\n", t).strip()
        return t

    def doc(self, topic):
        """Trả {'title','text'} hoặc None nếu lỗi/không có bài đủ dài."""
        try:
            ts = self.titles(topic) or self.titles(None)
            random.shuffle(ts)
            for title in ts[:4]:
                t = self.text(title)
                letters = sum(c.isalpha() for c in t)
                if len(t) >= 500 and letters > 0.6 * len(t) * 0.8:
                    self.fails = 0
                    return {"title": title, "text": t, "source": "Wikipedia"}
        except Exception as e:  # noqa: BLE001
            self.fails += 1
            if not self.warned:
                self.warned = True
                print(f"  ⚠ không lấy được Wikipedia ({str(e)[:80]}); sẽ thử nguồn khác", flush=True)
        return None


QUERY_ANGLES = ["là gì", "cách làm", "mẹo hay", "ở Việt Nam", "giải thích đơn giản", "lịch sử", "ví dụ", "so sánh", "lưu ý", "câu hỏi thường gặp",
                "nguyên nhân", "lợi ích", "sai lầm thường gặp", "cho người mới", "ngày nay", "ý nghĩa"]


def make_query(topic):
    """(chủ đề gốc, truy vấn) - thêm "góc nhìn" ngẫu nhiên để mỗi lần tìm ra trang khác nhau, không lặp một truy vấn chung chung."""
    base = topic or random.choice(["kiến thức đời sống", "mẹo vặt hữu ích", "sự thật thú vị", "kiến thức khoa học cơ bản", "văn hoá Việt Nam", "sức khoẻ đời thường"])
    return base, f"{base} {random.choice(QUERY_ANGLES)}".strip()


class Tavily:
    """Tìm web bằng Tavily (tuỳ chọn). Nhiều key xoay vòng, không giới hạn số key; hết key/hết hạn mức thì tự bỏ nguồn này, các nguồn khác vẫn chạy."""

    def __init__(self, keys, base="https://api.tavily.com", delay=0.3):
        self.keys = [{"k": k, "cool": 0.0, "dead": False} for k in keys]
        self.base, self.delay, self.fails, self.calls = base.rstrip("/"), delay, 0, 0

    @property
    def available(self):
        return any(not k["dead"] for k in self.keys) and self.fails < 5

    def _post(self, key, body):
        req = urllib.request.Request(self.base + "/search", method="POST", data=json.dumps(body).encode("utf-8"),
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                return r.status, r.read().decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            return e.code, e.read().decode("utf-8", "replace")
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            return 599, str(e)

    label = "Tavily"

    def doc(self, topic):
        """Trả {'title','text'} từ các kết quả tìm kiếm của chủ đề, hoặc None."""
        return self.fetch(*make_query(topic))

    def fetch(self, base, query):
        body = {"query": query, "search_depth": "basic", "max_results": 5, "include_answer": False}
        for _ in range(len(self.keys) * 2 + 1):
            alive = [k for k in self.keys if not k["dead"] and k["cool"] <= time.time()]
            if not alive:
                break
            key = random.choice(alive)
            time.sleep(self.delay)
            self.calls += 1
            st, txt = self._post(key["k"], body)
            if st == 200:
                try:
                    res = json.loads(txt).get("results", [])
                except ValueError:
                    res = []
                parts = [f"{r.get('title', '').strip()}: {str(r.get('content', '')).strip()}" for r in res if str(r.get("content", "")).strip()]
                text = "\n".join(parts)[:4000]
                if len(text) >= 300:
                    self.fails = 0
                    return {"title": base, "text": text, "source": "Tavily"}
                return None
            if st in (401, 403, 432, 433):    # key sai / hết hạn mức gói -> bỏ key này
                key["dead"] = True
                print(f"  ⚠ Tavily: key ...{key['k'][-4:]} bị từ chối/hết hạn mức ({st}), bỏ key này", flush=True)
                continue
            if st == 429:
                key["cool"] = time.time() + 20
                continue
            self.fails += 1
            break
        return None


class WebPicker:
    """Mỗi lần gọi LLM chọn nguồn thông tin: chuỗi cào web (Tavily -> Exa -> Linkup -> Firecrawl), bài Wikipedia thật ('wiki'),
    hay để Gemini tự tra Google ('search'), hay thuần sáng tác."""

    def __init__(self, mode, wiki, topics, scraper=None):
        self.mode, self.wiki, self.topics, self.scraper = mode, wiki, topics, scraper

    def pick(self, model, gem):
        """Cào web là nguồn chính khi có key; Wikipedia xen ~25% cho đa dạng (hoặc làm nguồn chính khi không có key cào).
        Gemini google_search CHỈ là dự phòng cuối (hoặc khi chọn --web search): LLM để lọc/chắt lọc, không tốn quota vào việc tra cứu.
        google_search chỉ có ở Gemini: đang dùng model của provider khác thì chế độ 'search' tự chuyển sang Wikipedia/cào web."""
        topic = random.choice(self.topics) if self.topics and random.random() < 0.8 else None   # 20% bài ngẫu nhiên cho đa dạng
        can_s = self.mode in ("search", "both") and gem.can_search(model)
        mode = "both" if (self.mode == "search" and not can_s) else self.mode
        can_t = mode in ("tavily", "both") and self.scraper is not None and self.scraper.available
        can_w = mode in ("wiki", "both") and self.wiki.fails < 5
        order = []
        if can_t:
            order += ["scrape"] + (["wiki"] if can_w and random.random() < 0.25 else [])
            order += (["wiki"] if can_w and "wiki" not in order else [])
        elif can_w:
            order.append("wiki")
        if can_s:
            order.append("search")
        if mode == "search":
            order = [o for o in order if o == "search"] or order
        for o in order:
            if o == "search":
                return None, topic or random.choice(self.topics or ["một chủ đề đời sống bất kỳ"]), True
            d = (self.scraper if o == "scrape" else self.wiki).doc(topic)
            if d:
                return d, topic, False
        return None, topic, False   # không nguồn nào dùng được -> thuần sáng tác như cũ


def json_slice(t):
    """Gemini dùng google_search (và các provider khác, không có JSON mode kiểu mảng) hay kèm lời dẫn/```json: lấy mảng [...] ra khỏi câu trả lời."""
    t = strip_fence(t)
    if t.lstrip().startswith(("[", "{")):   # đã là mảng JSON (hoặc JSONL): giữ nguyên
        return t
    a, b = t.find("["), t.rfind("]")
    return t[a:b + 1] if 0 <= a < b else t


def too_close(par, doc, n=12):
    """True nếu đoạn văn chép nguyên >= n từ liên tiếp từ tài liệu gốc (tránh up lại nguyên văn Wikipedia)."""
    if not doc:
        return False
    src = " ".join(norm_key(doc["text"]).split())
    w = norm_key(par).split()
    return any(" ".join(w[i:i + n]) in src for i in range(0, max(1, len(w) - n + 1)))


def finetune_prompt(n, examples, doc=None, topic=None, search=False):
    sit = random.sample(SITUATIONS, 3)
    topic = topic or random.choice(TOPICS)
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
    if doc:
        prompt = (f"TÀI LIỆU thật lấy từ internet (chủ đề «{doc['title']}»):\n\"\"\"\n{doc['text']}\n\"\"\"\n\n"
                  f"Hãy tạo {n} cuộc hội thoại KHÁC NHAU mà người dùng hỏi những điều CÓ TRONG tài liệu trên (hỏi tự nhiên, đời thường, không nhắc tới tài liệu), "
                  "Yui trả lời ĐÚNG theo tài liệu, ngắn gọn, giọng Yui. Chỉ dùng thông tin có trong tài liệu, không thêm số liệu ngoài tài liệu; "
                  "Yui không nói 'theo tài liệu/Wikipedia'. Mỗi cuộc có 1 đến 2 lượt, mỗi tin tối đa 25 từ. "
                  "Giọng user đa dạng (cộc lốc, viết tắt, sai chính tả nhẹ, tò mò). Không nội dung 18+, bạo lực, chính trị, không thông tin cá nhân thật.\n"
                  "Ví dụ giọng văn Yui (đừng chép lại):\n" + shots + "\n\n"
                  'Chỉ trả về mảng JSON: [{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}, ...]')
    elif search:
        prompt = (f"Hãy TRA CỨU trên Google thông tin chính xác, hữu ích về chủ đề «{topic}», rồi tạo {n} cuộc hội thoại KHÁC NHAU dựa trên thông tin tra được: "
                  "người dùng hỏi tự nhiên, đời thường (không nhắc việc tra cứu), Yui trả lời ĐÚNG sự thật, ngắn gọn, giọng Yui. "
                  "Không biết chắc thì Yui nói thật là không rõ. Mỗi cuộc có 1 đến 2 lượt, mỗi tin tối đa 25 từ. Yui không nhắc tên nguồn/Google/Wikipedia. "
                  "Không nội dung 18+, bạo lực, chính trị, không thông tin cá nhân thật.\n"
                  "Ví dụ giọng văn Yui (đừng chép lại):\n" + shots + "\n\n"
                  'Chỉ trả về mảng JSON (không giải thích, không markdown): [{"messages":[{"role":"user","content":"..."},{"role":"assistant","content":"..."}]}, ...]')
    return system, prompt


def pretrain_prompt(n, doc=None, topic=None, search=False):
    style, level = random.choice(PRETRAIN_STYLES)
    topic = topic or random.choice(PRETRAIN_TOPICS)
    system = "Bạn là nhà văn và biên tập viên tiếng Việt. Chỉ trả về văn bản, không lời dẫn, không markdown."
    prompt = (f"Viết {n} đoạn văn tiếng Việt RIÊNG BIỆT, mỗi đoạn 120-250 từ, thể loại: {style}; độ khó: {level}; chủ đề gợi ý: {topic} "
              "(mỗi đoạn có thể đổi góc nhìn). Nội dung phải là sáng tác gốc, không chép lại tác phẩm có bản quyền, thông tin thực tế phải đúng. "
              "Tiếng Việt chuẩn có dấu, tự nhiên. Không đánh số, không tiêu đề, không gạch đầu dòng. "
              "Các đoạn cách nhau bằng một dòng chỉ gồm ba dấu =  (===).")
    if doc:
        prompt = (f"TÀI LIỆU thật lấy từ internet (chủ đề «{doc['title']}»):\n\"\"\"\n{doc['text']}\n\"\"\"\n\n"
                  f"Dựa trên thông tin trong tài liệu, viết {n} đoạn văn tiếng Việt RIÊNG BIỆT, mỗi đoạn 120-250 từ, thể loại: {style}; độ khó: {level}. "
                  "PHẢI viết lại bằng lời văn của bạn (không chép nguyên câu, không chép quá 8 từ liên tiếp), thông tin đúng như tài liệu, không thêm số liệu ngoài tài liệu. "
                  "Tiếng Việt chuẩn có dấu, tự nhiên. Không đánh số, không tiêu đề, không gạch đầu dòng, không nhắc tới 'tài liệu' hay Wikipedia. "
                  "Các đoạn cách nhau bằng một dòng chỉ gồm ba dấu =  (===).")
    elif search:
        prompt = (f"Hãy TRA CỨU trên Google thông tin chính xác về chủ đề «{topic}», rồi viết {n} đoạn văn tiếng Việt RIÊNG BIỆT, mỗi đoạn 120-250 từ, "
                  f"thể loại: {style}; độ khó: {level}, dựa trên thông tin tra được. Viết bằng lời văn của bạn, thông tin phải đúng. "
                  "Tiếng Việt chuẩn có dấu, tự nhiên. Không đánh số, không tiêu đề, không gạch đầu dòng, không nhắc tên nguồn. "
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


def parse_source(text):
    """Đọc file hội thoại cũ: JSONL {"user","assistant"} / {"messages"} hoặc dạng văn bản 'Người dùng: ...\nYui: ...'. Trả [(hỏi, đáp)]."""
    t = text.lstrip("\ufeff")
    if re.match(r"\s*[\[{]", t):
        convs, _ = load_convs({"s": t}, max_chars=10 ** 6)
        return [(c[0][0], c[0][1]) for c in convs]
    pairs = []
    for blk in re.split(r"\n\s*\n(?=Người dùng:)", t):
        m = re.match(r"\s*Người dùng:\s*(.*?)\n\s*Yui:\s*(.*)\Z", blk, re.S)
        if m:
            pairs.append((clean(m.group(1)), clean(m.group(2))))
    return pairs


def read_source(args, remote):
    if args.source_file:
        return open(args.source_file, encoding="utf-8", errors="ignore").read()
    if not args.source:
        sys.exit("Thiếu --source TÊN_FILE_TRÊN_REPO hoặc --source_file ĐƯỜNG_DẪN")
    if remote is None:
        sys.exit("Không có repo để đọc --source (dùng --source_file, hoặc bỏ --no_upload)")
    return remote.read(args.source).decode("utf-8", "ignore")


def pct(xs, q):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(len(xs) * q))] if xs else 0


def check(args, remote):
    text = read_source(args, remote)
    pairs = parse_source(text)
    broken = len(re.findall(r"(?<![a-zA-ZÀ-ỹ])(?:đy|dy)(?![a-zA-ZÀ-ỹ])", text.lower()))
    if pairs:
        us, an = [len(u) for u, _ in pairs], [len(a) for _, a in pairs]
        print(f"📄 {len(pairs)} cặp hỏi-đáp ({len(set(norm_key(u) for u, _ in pairs))} câu hỏi khác nhau)")
        print(f"   câu hỏi:   trung vị {pct(us, .5)} ký tự, 90% ≤ {pct(us, .9)}, dài nhất {max(us)}")
        print(f"   câu đáp:   trung vị {pct(an, .5)} ký tự, 90% ≤ {pct(an, .9)}, dài nhất {max(an)}")
        print(f"   đáp > 200 ký tự: {100 * sum(a > 200 for a in an) / len(an):.0f}%  (model chỉ nhìn được 128 token ~ 400 ký tự cho CẢ hỏi lẫn đáp)")
        ops = {}
        for _, a in pairs:
            k = " ".join(a.split()[:4]).lower()
            ops[k] = ops.get(k, 0) + 1
        print("   mở đầu lặp nhiều nhất:", "; ".join(f"“{k}” ×{v}" for k, v in sorted(ops.items(), key=lambda x: -x[1])[:5]))
        boiler = sum("siêu trợ lý" in a.lower() for _, a in pairs)
        print(f"   chứa “siêu trợ lý”: {boiler} cặp ({100 * boiler / len(pairs):.0f}%)  -> giọng cũ, khác giọng Yui dễ thương trong 1.txt..N.txt")
    else:
        blocks = [b for b in re.split(r"\n\s*\n(?=Chủ đề:)", text) if b.strip()]
        ln = [len(b) for b in blocks]
        print(f"📄 dạng văn bản/kiến thức: {len(blocks)} khối, trung vị {pct(ln, .5)} ký tự")
        print(f"   khối có dấu '[...]' (bị cắt cụt): {sum('[...]' in b for b in blocks)}; có '##' (còn markdown): {sum('##' in b for b in blocks)}")
    if broken:
        print(f"⚠ {broken} chỗ nghi mất chữ “â” (vd “đy”/“dy” thay cho “đây”/“dây”): file này có thể bị hỏng chính tả, KHÔNG nên dùng để train trước khi sửa nguồn thu thập")
    else:
        print("✅ không thấy dấu hiệu mất chữ “â” (đy/dy)")


def restyle_prompt(items):
    system = ("Bạn tạo dữ liệu huấn luyện cho chatbot tiếng Việt nhỏ tên Yui. " + PERSONA + " Chỉ trả về một mảng JSON, không giải thích.")
    lines = "\n".join(json.dumps({"i": i, "question": q, "old_answer": a[:220]}, ensure_ascii=False) for i, q, a in items)
    prompt = ("Dưới đây là các câu người dùng thật đã nhắn cho Yui, kèm câu trả lời cũ (giọng cũ, dài dòng, chỉ để tham khảo ý chính, có thể sai).\n"
              "Với MỖI câu, viết lại câu trả lời của Yui theo đúng giọng Yui: tối đa 25 từ, đúng ý câu hỏi, tự nhiên, không lặp khuôn.\n"
              "Nếu câu hỏi cần thông tin thời gian thực (tin tức, thời tiết, giá cả) hoặc việc Yui không làm được thì Yui nói thật là không xem được/không làm được "
              "theo kiểu dễ thương, tuyệt đối không bịa. Không tự nhận là mô hình ngôn ngữ, không nhắc tên công ty AI. Không nội dung 18+, bạo lực, chính trị.\n"
              "Các câu:\n" + lines + "\n\nTrả về: [{\"i\":0,\"answer\":\"...\"}, ...] đủ mọi i, đúng thứ tự.")
    return system, prompt


def restyle(args, gem, pool, pub, remote, seen):
    pairs = parse_source(read_source(args, remote))
    qs, mine = [], set()
    for q, a in pairs:
        q = clean(q)
        k = norm_key(q)
        if not (2 <= len(q) <= 200) or not k or k in seen or k in mine or k == "test":
            continue
        mine.add(k)
        qs.append((q, a))
    random.shuffle(qs)
    per_file = args.per_file or 100
    print(f"🔁 restyle: {len(pairs)} cặp trong file nguồn -> {len(qs)} câu hỏi dùng được (đã bỏ trùng với data trên repo)\n", flush=True)
    buf, done_files, fails, pos, idle_since = [], 0, 0, 0, None
    while done_files < args.files and pos < len(qs) and time.time() <= DEADLINE:
        now = time.time()
        cur = next((m for m in pool if gem.busy.get(m, 0.0) <= now), None)   # model xếp hạng cao nhất đang rảnh
        if cur is None:
            idle_since = idle_since or now
            if now - idle_since > 20 * 60:
                print("❌ mọi model đều bận/quá tải hơn 20 phút, dừng (phần đã tạo vẫn được lưu)", flush=True)
                break
            time.sleep(max(5.0, min(60.0, min(gem.busy.get(m, 0.0) for m in pool) - now)))
            continue
        idle_since = None
        chunk = qs[pos:pos + args.batch]
        system, prompt = restyle_prompt([(i, q, a) for i, (q, a) in enumerate(chunk)])
        try:
            raw = gem.generate(cur, system, prompt, temperature=0.9, json_mode=True)
            ans = {int(x["i"]): clean(str(x["answer"])) for x in json.loads(json_slice(raw)) if isinstance(x, dict) and "i" in x and "answer" in x}
        except ModelBusy:
            continue   # đã ghi nhận bận: vòng sau tự chọn model/provider kế
        except ModelGone as e:
            print(f"  ⚠ bỏ model {cur}: {e}", flush=True)
            pool.remove(cur)
            if not pool:
                print("❌ hết model để thử")
                break
            continue
        except (Empty, Transient, ValueError, KeyError, TypeError) as e:
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
        pos += len(chunk)
        added = 0
        for i, (q, _) in enumerate(chunk):
            a = ans.get(i, "")
            if not a or len(a) > 200 or LEAK.search(a):
                continue
            buf.append(json.dumps({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}, ensure_ascii=False))
            seen.add(norm_key(q))
            added += 1
        print(f"  + {added}/{len(chunk)} cặp hợp lệ (đệm {len(buf)}/{per_file}, còn {len(qs) - pos} câu hỏi)", flush=True)
        while len(buf) >= per_file and done_files < args.files:
            part, buf = buf[:per_file], buf[per_file:]
            pub.publish("\n".join(part) + "\n")
            done_files += 1
    if buf and done_files < args.files and len(buf) >= max(10, per_file // 5):
        pub.publish("\n".join(buf) + "\n")
        done_files += 1
    print(f"\n🎉 Xong: {done_files} file mới.", flush=True)


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
    if not prefix.startswith(("pretrain/", "data/")):
        convs, _ = load_convs(texts)
        for c in convs:
            seen.add(norm_key(c[0][0]))
            pairs.append(c[0])
    else:
        for t in texts.values():
            for p in re.split(r"\n{2,}", t):
                seen.add(norm_key(p[:80]))
    return seen, pairs



# ============================ chế độ tự động: đọc BIẾN MÔI TRƯỜNG, tạo phần còn thiếu ============================
def env(name, default=None):
    v = os.environ.get(name)
    return default if v is None or not v.strip() else v.strip()


def parse_count(s):
    """'50M' -> 50_000_000, '1.2B' -> 1.2e9, '300k', '50tr' (triệu), '50000000'."""
    m = re.fullmatch(r"\s*([\d.,]+)\s*([kKmMbBgG]|tr|tỷ|ty)?\s*", str(s))
    if not m:
        raise ValueError(f"không hiểu số '{s}' (vd 50M, 1.2B, 50000000)")
    v = float(m.group(1).replace(",", "."))
    mult = {"k": 1e3, "m": 1e6, "tr": 1e6, "b": 1e9, "g": 1e9, "tỷ": 1e9, "ty": 1e9}.get((m.group(2) or "").lower(), 1)
    return int(v * mult)


def model_params(token=None, remote=None):
    """(số tham số, mô tả nguồn). Thứ tự: MODEL_PARAMS > config.json của MODEL_REPO > config.json nằm ngay trong repo dataset (train_lm.py --hf_repo tự đẩy lên)
    > MODEL_D/LAYERS/EXPERTS/VOCAB > mặc định 50M. Nghĩa là: đã train/đẩy checkpoint lên repo thì KHÔNG cần nhập tham số, hệ thống tự nhận."""
    def from_cfg(cfg):
        return params_from_arch(cfg["vocab"], cfg["d"], cfg["layers"], cfg["experts"])
    if env("MODEL_PARAMS"):
        return parse_count(env("MODEL_PARAMS")), "MODEL_PARAMS"
    if env("MODEL_REPO"):
        try:
            from huggingface_hub import hf_hub_download
            cfg = json.load(open(hf_hub_download(env("MODEL_REPO"), "config.json", repo_type=env("MODEL_REPO_TYPE", "dataset"), token=token), encoding="utf-8"))
            return from_cfg(cfg), f"config.json của {env('MODEL_REPO')}"
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠ không đọc được config.json của MODEL_REPO ({str(e)[:80]}); thử cách khác", flush=True)
    if remote is not None:
        try:
            return from_cfg(json.loads(remote.read("config.json").decode("utf-8"))), "config.json trong repo dataset (tự nhận)"
        except Exception:  # noqa: BLE001
            pass
    if env("MODEL_D") and env("MODEL_LAYERS") and env("MODEL_EXPERTS"):
        return params_from_arch(int(env("MODEL_VOCAB", "16000")), int(env("MODEL_D")), int(env("MODEL_LAYERS")), int(env("MODEL_EXPERTS"))), "MODEL_D/LAYERS/EXPERTS/VOCAB"
    return 50_000_000, "mặc định 50M (chưa thấy config.json trong repo; đặt MODEL_PARAMS để đổi)"


def fmt_n(n):
    return f"{n / 1e9:.2f}B" if n >= 1e9 else (f"{n / 1e6:.1f}M" if n >= 1e6 else (f"{n / 1e3:.1f}k" if n >= 1e3 else str(int(n))))


def make_plan(params, have_pairs, have_tokens, tokens_per_file):
    """Từ tham số model + data đang có -> cần tạo thêm bao nhiêu. Công thức (đều ghi đè được bằng biến):
       cặp fine-tune mục tiêu = params / 2000 (kẹp 2.000 .. 100.000);   token train mục tiêu = params x 0,2 (PRETRAIN_TOKENS_PER_PARAM; phần lớn data train nên lấy
       từ web có sẵn qua train_lm.py --hf_dataset, phần tạo bằng Gemini là phần tinh chỉnh chất lượng)."""
    ft_per_file = int(env("FT_PER_FILE", "100"))
    pre_per_file = int(env("PRETRAIN_PER_FILE", "150"))
    if env("FT_NEW_PAIRS"):
        ft_need = int(parse_count(env("FT_NEW_PAIRS"))); ft_target = have_pairs + ft_need
    else:
        ft_target = int(parse_count(env("FT_PAIRS"))) if env("FT_PAIRS") else int(min(100_000, max(2_000, params / 2000)))
        ft_need = max(0, ft_target - have_pairs)
    if env("PRETRAIN_NEW_TOKENS"):
        pre_need = parse_count(env("PRETRAIN_NEW_TOKENS")); pre_target = have_tokens + pre_need
    else:
        pre_target = parse_count(env("PRETRAIN_TOKENS")) if env("PRETRAIN_TOKENS") else int(params * float(env("PRETRAIN_TOKENS_PER_PARAM", "0.2")))
        pre_need = max(0, pre_target - have_tokens)
    return dict(ft_per_file=ft_per_file, ft_target=ft_target, ft_need=ft_need, ft_files=math.ceil(ft_need / ft_per_file),
                pre_per_file=pre_per_file, pre_target=pre_target, pre_need=pre_need, pre_files=math.ceil(pre_need / max(1, tokens_per_file)))


_TOOL_HAVE = 0   # số hội thoại công cụ đã có (count_existing điền)


def count_existing(remote, tag):
    """Đếm data ĐÃ CÓ cho cỡ model này: Fine-tune/<tag>/N.txt (số cặp) và data/<tag>/N.txt (số ký tự)."""
    names = remote.list() if remote is not None else []
    fp, dp = finetune_prefix(tag), data_prefix(tag)
    ft = [n for n in names if re.fullmatch(re.escape(fp) + r"\d+\.txt", n)]
    pre = [n for n in names if re.fullmatch(re.escape(dp) + r"\d+\.txt", n)]
    pairs, chars, pre_files = 0, 0, 0
    global _TOOL_HAVE
    _TOOL_HAVE = 0
    for n in ft:
        try:
            for ln in remote.read(n).decode("utf-8", "ignore").splitlines():
                if ln.strip().startswith(("{", "[")):
                    if "<tool>" in ln:
                        _TOOL_HAVE += 1
                    else:
                        pairs += 1
        except Exception:  # noqa: BLE001
            pass
    for n in pre:
        try:
            chars += len(remote.read(n).decode("utf-8", "ignore")); pre_files += 1
        except Exception:  # noqa: BLE001
            pass
    return pairs, chars, pre_files


def auto(args, build_parser):
    load_kaggle_secrets()
    token = env("HF_TOKEN")
    cpt = float(env("CHARS_PER_TOKEN", "3.0"))
    kinds = [k.strip() for k in (env("DATA_KINDS", "finetune,pretrain")).split(",") if k.strip() in ("finetune", "pretrain", "tools")]
    if not kinds:
        sys.exit("DATA_KINDS phải chứa finetune, pretrain và/hoặc tools")
    repo_dir = env("DATA_REPO_DIR") or args.repo_dir
    remote = LocalRepo(repo_dir) if repo_dir else HFRepo(env("HF_REPO", args.repo), env("HF_REPO_TYPE", args.repo_type), token)
    params, src = model_params(token, remote)
    tag = env("MODEL_TAG") or size_tag(params)
    try:
        have_pairs, have_chars, pre_files = count_existing(remote, tag)
    except Exception as e:  # noqa: BLE001
        sys.exit(f"Không đọc được repo dữ liệu ({str(e)[:120]}): kiểm tra HF_TOKEN / HF_REPO")
    have_tokens = int(have_chars / cpt)
    est_file_chars = (have_chars / pre_files) if pre_files else int(env("PRETRAIN_PER_FILE", "150")) * 800
    plan = make_plan(params, have_pairs, have_tokens, int(est_file_chars / cpt))
    cap = int(env("MAX_FILES_PER_RUN", "100000" if DEADLINE != float("inf") else "40"))   # có giới hạn thời gian thì thời gian quyết định, không phải số file
    print(f"🧭 Model: {fmt_n(params)} tham số (từ {src}) -> thư mục cỡ '{tag}': Fine-tune/{tag}/ và data/{tag}/")
    if "finetune" in kinds:
        print(f"   Fine-tune: đang có {have_pairs} cặp -> mục tiêu {plan['ft_target']} -> cần tạo thêm {plan['ft_need']} cặp "
              f"= {plan['ft_files']} file ({plan['ft_per_file']} cặp/file; tối đa {cap} file mỗi lần chạy)")
    if "pretrain" in kinds:
        print(f"   Data train: đang có ~{fmt_n(have_tokens)} token ({pre_files} file) -> mục tiêu {fmt_n(plan['pre_target'])} -> cần thêm ~{fmt_n(plan['pre_need'])} token "
              f"= {plan['pre_files']} file (không cần đặt 'cặp'; mỗi file ~{plan['pre_per_file']} đoạn văn)")
    if args.plan:
        return
    web = env("WEB_SEARCH", "both")
    if web not in ("off", "wiki", "search", "tavily", "both"):
        sys.exit("WEB_SEARCH phải là off | wiki | search | tavily | both")
    for kind in kinds:
        if kind == "tools":
            want = int(parse_count(env("TOOL_PAIRS"))) if env("TOOL_PAIRS") else max(500, plan["ft_target"] // 10)
            need = max(0, want - _TOOL_HAVE)
            if need <= 0:
                print(f"\n✅ tools: đã có {_TOOL_HAVE} hội thoại dùng công cụ, đủ mục tiêu {want}.")
                continue
            print(f"\n▶ tools: tạo {need} hội thoại dạy dùng công cụ (calc/time/search, không gọi Gemini)...", flush=True)
            import tool_data
            convs = tool_data.make(need, random.randrange(10**6), tool_data.load_facts(env("TOOL_FACTS")), [], int(env("TOOL_RESULT_CHARS", "300")), {"calc": 3, "time": 1, "search": 3})
            pub = Publisher(remote, args.out_dir, finetune_prefix(tag), upload=True)
            per = plan["ft_per_file"]
            for i in range(0, len(convs), per):
                pub.publish("\n".join(json.dumps({"messages": [{"role": r, "content": t} for r, t in c]}, ensure_ascii=False) for c in convs[i:i + per]) + "\n")
            continue
        files = min(cap, plan["ft_files"] if kind == "finetune" else plan["pre_files"])
        if files <= 0:
            print(f"\n✅ {kind}: đã đủ so với mục tiêu, không tạo thêm.")
            continue
        print(f"\n▶ {kind}: tạo {files} file mới...\n", flush=True)
        argv = [kind, "--files", str(files), "--per_file", str(plan["ft_per_file"] if kind == "finetune" else plan["pre_per_file"]), "--web", web, "--tag", tag,
                "--repo", env("HF_REPO", args.repo), "--repo_type", env("HF_REPO_TYPE", args.repo_type), "--rpm", env("GEMINI_RPM", str(args.rpm)),
                "--out_dir", args.out_dir]
        if repo_dir:
            argv += ["--repo_dir", repo_dir]
        if env("LLM_MODEL"):   # ép đúng 1 model "provider::model" (tắt chuỗi dự phòng); GEMINI_MODEL chỉ ép phần Gemini, xem build_pool
            argv += ["--model", env("LLM_MODEL")]
        if env("WEB_TOPICS"):
            argv += ["--topics", env("WEB_TOPICS")]
        argv += ["--api_base", args.api_base]
        argv += ["--wiki_base", args.wiki_base, "--tavily_base", args.tavily_base, "--max_minutes", str(max(0.0, (DEADLINE - time.time()) / 60)) if DEADLINE != float("inf") else "0"]
        run(build_parser().parse_args(argv))


class RunLock:
    """Khoá tệp: hai tiến trình make_data cùng thư mục/chế độ không chạy chồng nhau (tránh đặt số file trùng và đốt quota gấp đôi)."""

    def __init__(self, path):
        self.path, self.f = path, None

    def acquire(self):
        try:
            import fcntl
        except ImportError:   # Windows: bỏ qua khoá
            return True
        os.makedirs(os.path.dirname(os.path.abspath(self.path)), exist_ok=True)
        self.f = open(self.path, "w")
        try:
            fcntl.flock(self.f, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except OSError:
            self.f.close(); self.f = None
            return False

    def release(self):
        if self.f:
            self.f.close(); self.f = None


def run(args):
    if args.mode in ("finetune", "pretrain", "restyle"):
        lock = RunLock(os.path.join(args.out_dir, f".lock_{args.mode}_{(args.tag or 'x').replace('/', '_')}"))
        if not lock.acquire():
            print(f"⚠ đang có một tiến trình make_data {args.mode} khác chạy trong '{args.out_dir}'; bỏ qua lần chạy này để không chạy chồng.", flush=True)
            return
        try:
            return _run(args)
        finally:
            lock.release()
    return _run(args)


def _run(args):
    random.seed(args.seed)
    if args.mode == "check":
        check(args, LocalRepo(args.repo_dir) if args.repo_dir else (None if args.source_file else HFRepo(args.repo, args.repo_type)))
        return
    load_kaggle_secrets()   # trên Kaggle: tự lấy secret; ngoài Kaggle không làm gì
    gem, llm_parts = build_llm(args)   # gem = LLMRouter: Cohere -> OpenRouter -> Mistral -> Gemini (theo LLM_ORDER), provider không có key bị bỏ qua
    if gem is None:
        sys.exit("Thiếu key LLM: đặt ít nhất một loại COHERE_API_KEY1 / OPENROUTER_API_KEY1 / MISTRAL_API_KEY1 / GEMINI_API_KEY1 (bao nhiêu key cũng được; hoặc NAME_KEYS cách nhau dấu phẩy; Gemini còn có --keys)")
    scraper, scrape_parts = build_scrapers(args)
    print("🔑 " + ", ".join(llm_parts + scrape_parts) + ("" if scraper else " (dùng LLM + Wikipedia)"), flush=True)
    print("🔀 Thứ tự LLM: " + " -> ".join(LLM_LABELS[n] for n in gem.order) + (" | cào web: " + " -> ".join(s_.label for s_ in scraper.sources) if scraper else ""), flush=True)
    pool = build_pool(gem, args.model)
    print("📋 Thứ hạng model (cao → thấp):", ", ".join(pool[:12]) or "(không có)", flush=True)
    if args.dry_models or not pool:
        return
    print(f"✅ Dùng model: {pool[0]} (tự chuyển sang model kế / provider kế nếu lỗi)\n", flush=True)

    if args.tag:   # xếp theo cỡ model: data/<tag>/ và Fine-tune/<tag>/ (xem data_layout.py)
        prefix = data_prefix(args.tag) if args.mode == "pretrain" else finetune_prefix(args.tag)
        print(f"📁 Thư mục đích trên repo: {prefix} (tự tạo nếu chưa có)", flush=True)
    else:
        prefix = "pretrain/" if args.mode == "pretrain" else ""
    remote = LocalRepo(args.repo_dir) if args.repo_dir else (None if args.no_upload else HFRepo(args.repo, args.repo_type))
    pub = Publisher(remote, args.out_dir, prefix, upload=not args.no_upload)
    seen, pairs = existing_state(remote, prefix)
    print(f"📚 Data đang có trên repo: {len(seen)} mục (sẽ không tạo trùng)\n", flush=True)
    if args.mode == "restyle":
        return restyle(args, gem, pool, pub, remote, seen)
    batch = args.batch if args.mode == "finetune" else max(2, args.batch // 4)
    per_file = args.per_file or (100 if args.mode == "finetune" else 150)

    webber = None
    if args.web != "off":
        topics = [t.strip() for t in re.split(r"[;\n]", args.topics or "") if t.strip()] or (TOPICS if args.mode == "finetune" else PRETRAIN_TOPICS)
        mode = args.web
        if mode == "tavily" and scraper is None:
            print("  ⚠ --web tavily nhưng không có key cào web nào (Tavily/Exa/Linkup/Firecrawl): dùng các nguồn còn lại (Wikipedia + Gemini google_search nếu có)", flush=True)
            mode = "both"
        webber = WebPicker(mode, Wiki(args.wiki_base), topics, scraper)
        print(f"🌐 Tìm thông tin trên internet: nguồn '{args.web}' ({len(topics)} chủ đề, xen bài ngẫu nhiên)\n", flush=True)

    done_files, buf, fails, busy_since, pending = 0, [], 0, None, None
    while done_files < args.files:
        if time.time() > DEADLINE:
            print(f"⏱ Hết thời gian cho phép ({args.max_minutes:g} phút); phần đã tạo vẫn được lưu.", flush=True)
            break
        now = time.time()
        cur = next((m for m in pool if gem.busy.get(m, 0.0) <= now), None)   # model xếp hạng cao nhất đang không bận
        if cur is None:
            busy_since = busy_since or now
            if now - busy_since > 20 * 60:
                print("❌ mọi model đều quá tải hơn 20 phút, dừng (phần đã tạo vẫn được lưu)", flush=True)
                break
            wait = max(5.0, min(60.0, min(gem.busy.get(m, 0.0) for m in pool) - now))
            print(f"  ⏳ mọi model đang quá tải, chờ {wait:.0f}s rồi thử lại...", flush=True)
            time.sleep(wait)
            continue
        busy_since = None
        fresh = pending is None
        if fresh:   # chỉ lấy tài liệu MỚI khi lần trước đã dùng xong (thử lại sau lỗi thì dùng lại tài liệu cũ, không tốn lượt Tavily/Wikipedia)
            pending = webber.pick(cur, gem) if webber else (None, None, False)
        doc, topic, use_search = pending
        if use_search and not gem.can_search(cur):   # google_search chỉ có ở Gemini: model hiện tại không dùng được thì chọn lại nguồn
            pending = None
            continue
        shots = random.sample(pairs, min(6, len(pairs))) if pairs else []
        if args.mode == "finetune":
            system, prompt = finetune_prompt(batch, shots, doc=doc, topic=topic, search=use_search)
        else:
            system, prompt = pretrain_prompt(batch, doc=doc, topic=topic, search=use_search)
        if doc and fresh:
            print(f"  📖 nguồn: {doc.get('source', 'web')} «{doc['title']}» -> {cur}", flush=True)
        elif doc:
            print(f"  ↻ thử lại cùng tài liệu «{doc['title']}» -> {cur}", flush=True)
        elif use_search and fresh:
            print(f"  🔎 Gemini tự tra Google: «{topic}»", flush=True)
        try:
            raw = gem.generate(cur, system, prompt, temperature=args.temperature, json_mode=(args.mode == "finetune" and not use_search), search=use_search)
        except SearchUnavailable:
            pending = None
            continue   # model này không có google_search: lần sau WebPicker tự bỏ 'search' cho model đó
        except ModelBusy:
            continue   # đã ghi nhận bận: vòng sau tự chọn model kế (giữ nguyên tài liệu)
        except ModelGone as e:
            print(f"  ⚠ bỏ model {cur}: {e}", flush=True)
            pool.remove(cur)
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
            time.sleep(min(30, 3 * fails))
            continue
        except QuotaOut as e:
            print(f"⛔ {e}. Dừng; phần đã tạo vẫn được lưu.", flush=True)
            break
        fails = 0
        pending = None   # dùng xong tài liệu này
        added = 0
        if args.mode == "finetune":
            convs, _ = load_convs({"x": json_slice(raw)}, max_chars=200)
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
                if k in seen or too_close(p, doc):
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


WIKI_BASE = "https://vi.wikipedia.org"
DEADLINE = float("inf")   # đặt trong main() từ --max_minutes


def build_parser():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["finetune", "pretrain", "restyle", "check", "auto"])
    ap.add_argument("--web", choices=["off", "wiki", "search", "tavily", "both"], default="off",
                    help="tìm thông tin trên internet: wiki = Wikipedia vi, search = Gemini tự tra Google, tavily = chuỗi cào web Tavily->Exa->Linkup->Firecrawl (cần ít nhất 1 key), both = tất cả nguồn đang dùng được")
    ap.add_argument("--tavily_keys", default=None, help="key Tavily (hoặc biến TAVILY_API_KEY1, TAVILY_API_KEY2, ... / TAVILY_API_KEYS); không có key vẫn chạy bằng LLM + Wikipedia")
    ap.add_argument("--exa_keys", default=None, help="key Exa (hoặc EXA_API_KEY1.. / EXA_API_KEYS); đứng sau Tavily trong chuỗi cào web")
    ap.add_argument("--linkup_keys", default=None, help="key Linkup (hoặc LINKUP_API_KEY1.. / LINKUP_API_KEYS); đứng sau Exa")
    ap.add_argument("--firecrawl_keys", default=None, help="key Firecrawl (hoặc FIRECRAWL_API_KEY1.. / FIRECRAWL_API_KEYS); đứng sau Linkup")
    ap.add_argument("--llm_order", default=env("LLM_ORDER"), help="thứ tự provider LLM, mặc định cohere,openrouter,mistral,gemini (hoặc biến LLM_ORDER)")
    ap.add_argument("--scrape_order", default=env("SCRAPE_ORDER"), help="thứ tự nguồn cào web, mặc định tavily,exa,linkup,firecrawl (hoặc biến SCRAPE_ORDER)")
    ap.add_argument("--tavily_base", default=env("TAVILY_BASE", "https://api.tavily.com"))
    ap.add_argument("--max_minutes", type=float, default=0, help="dừng tạo data sau ngần này phút (0 = không giới hạn); phần đã tạo vẫn được lưu")
    ap.add_argument("--topics", default=None, help="chủ đề riêng cho --web, cách nhau dấu ; (mặc định: danh sách có sẵn + bài ngẫu nhiên)")
    ap.add_argument("--wiki_base", default=env("WIKI_BASE", WIKI_BASE))
    ap.add_argument("--tag", default=None, help="cỡ model, vd 50M: file vào Fine-tune/<tag>/N.txt hoặc data/<tag>/N.txt (tự tạo thư mục). Bỏ trống = kiểu cũ")
    ap.add_argument("--plan", action="store_true", help="mode auto: chỉ in kế hoạch, không tạo gì")
    ap.add_argument("--source", default=None, help="restyle/check: tên file nguồn trên repo (vd yui_chat.jsonl)")
    ap.add_argument("--source_file", default=None, help="restyle/check: đường dẫn file nguồn trên máy")
    ap.add_argument("--files", type=int, default=1, help="số file cần tạo ở lần chạy này")
    ap.add_argument("--per_file", type=int, default=0, help="số hội thoại (finetune, mặc định 100) hoặc đoạn văn (pretrain, mặc định 150) mỗi file")
    ap.add_argument("--keys", default=None, help="key Gemini (ngoài biến môi trường GEMINI_API_KEY*)")
    ap.add_argument("--model", default=None, help="ép dùng đúng một model, không dự phòng: 'provider::model' (vd cohere::command-a-03-2025) hoặc tên trần = model Gemini")
    ap.add_argument("--repo", default="BachDaThan/TrainAI-web")
    ap.add_argument("--repo_type", default="dataset")
    ap.add_argument("--repo_dir", default=None, help="dùng một thư mục làm 'repo' (để thử, không cần mạng HF)")
    ap.add_argument("--no_upload", action="store_true")
    ap.add_argument("--out_dir", default="data_out")
    ap.add_argument("--batch", type=int, default=20, help="số hội thoại mỗi lần gọi API (pretrain: batch//4 đoạn)")
    ap.add_argument("--rpm", type=float, default=8.0, help="số yêu cầu/phút cho MỖI key Gemini (gói miễn phí thường 5-15); provider khác: COHERE_RPM/OPENROUTER_RPM/MISTRAL_RPM")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--api_base", default=env("GEMINI_API_BASE", API_BASE))
    ap.add_argument("--dry_models", action="store_true")
    ap.add_argument("--seed", type=int, default=int(time.time()) % 100000)
    return ap


def main():
    global DEADLINE
    args = build_parser().parse_args()
    if args.max_minutes > 0:
        DEADLINE = time.time() + args.max_minutes * 60
    if args.mode == "auto":
        random.seed(args.seed)
        return auto(args, build_parser)
    run(args)


def estimate_minutes(params, n_keys, rpm=8.0, per_call=20, lo=15, hi=240):
    """Số phút cần để tạo xong phần fine-tune theo cỡ model: mục tiêu cặp = params/2000 (kẹp 2k..100k), mỗi lượt gọi LLM ra ~per_call cặp,
    thông lượng ~ n_keys x rpm x 0,7 lượt/phút (hao do 429/503), cộng 30% dự phòng. Kẹp [lo, hi] phút; thiếu thì lần chạy sau tự bù tiếp."""
    pairs = min(100_000, max(2_000, params / 2000))
    throughput = max(1, n_keys) * rpm * 0.7
    return int(min(hi, max(lo, math.ceil(pairs / per_call / throughput * 1.3))))


def make_for_training(params, tag, repo, repo_type="dataset", kinds="finetune,pretrain", minutes="auto", extra_env=None):
    """Gọi từ train_lm.py: tạo (phần còn thiếu của) data cho model `params` tham số TRƯỚC khi train. Không có key LLM nào (Cohere/OpenRouter/Mistral/Gemini) thì bỏ qua, train vẫn chạy.
    Chạy bằng tiến trình con (`make_data.py auto`) nên lỗi ở đây không làm hỏng lần train."""
    load_kaggle_secrets()
    if llm_key_total() == 0:
        print("[data] chưa có key LLM nào (COHERE_API_KEY1, OPENROUTER_API_KEY1, MISTRAL_API_KEY1, GEMINI_API_KEY1, ...) nên bỏ qua bước tự tạo data; train bằng data có sẵn.", flush=True)
        return False
    if minutes in (None, "", "auto"):
        rpm = total_rpm(float(env("GEMINI_RPM", "8")))
        minutes = estimate_minutes(params, 1, rpm)   # rpm = tổng thông lượng mọi key của mọi provider
        print(f"[data] tự tính thời gian tạo data theo cỡ model {fmt_n(params)} và {llm_key_total()} key (~{rpm:.0f} yêu cầu/phút): {minutes} phút", flush=True)
    minutes = float(minutes)
    e = dict(os.environ, MODEL_PARAMS=str(int(params)), MODEL_TAG=tag, HF_REPO=repo, HF_REPO_TYPE=repo_type, DATA_KINDS=kinds, **(extra_env or {}))
    print(f"[data] tạo data cho model {fmt_n(params)} tham số (thư mục cỡ {tag}) trên {repo}, tối đa {minutes:g} phút...", flush=True)
    rc = subprocess.run([sys.executable, "-u", os.path.abspath(__file__), "auto", "--max_minutes", str(minutes)], env=e).returncode
    if rc:
        print(f"[data] bước tạo data kết thúc với mã {rc}; vẫn tiếp tục train.", flush=True)
    return rc == 0


if __name__ == "__main__":
    main()
