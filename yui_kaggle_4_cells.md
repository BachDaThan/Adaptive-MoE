# Yui pipeline v6.5 — 4 ô Kaggle (có BitNet ternary + XEM ẢNH)

Thêm vào repo (cạnh `train_lm.py`): `bitnet_yui.py`, `bitnet_pack.py`, `train_lm.py`, `export_yui.py` (bản mới). Đặt `BITNET = 0` ở Ô1 để tắt BitNet và chạy như cũ.

**v6.5 — xem ảnh (tuỳ chọn, tắt mặc định):** thêm vào repo `vision_yui.py`, `train_vision.py`, `export_vision.py`, `make_vision_data.py`, `elastic_ctx.py` (bản mới), `yui.cpp` (bản mới). Bật bằng `VISION = True` ở đầu Ô4. Model MỚI nên đặt `CTX = "512"` ở Ô1 (256 vẫn chạy được nhưng sát; ảnh chiếm 49 token). Cần `GEMINI_API_KEY` trong môi trường lúc chạy Ô4 để tạo data ảnh (nếu đã có data ảnh trên HF từ phiên trước thì không cần).

## Ô 1: Env + Secrets (Chống Rate Limit)

```python
# ============================================================
# YUI PIPELINE v6.4 (chạy 12 tiếng) — Ô 1 (copy từng ô vào Kaggle). Khớp đúng repo Adaptive-MoE.
#   Ô1  Env + nạp Secrets CHỐNG RATE LIMIT (chạy ở MỌI phiên; MODE="cpu" hoặc "gpu")
# ============================================================

import os, sys, json, time

RUN_HOURS = 12          # ⏱ TỔNG thời gian phiên Kaggle bạn muốn dùng (Kaggle cho tối đa 12h). Các ô tự tính phút còn lại và dừng êm TRƯỚC giới hạn.
RESERVE_CPU = 20        # phút chừa cuối phiên CPU (tự kiểm tra + báo cáo)
RESERVE_GPU = 75        # phút chừa cuối phiên GPU cho Ô4 (SFT + export int8/q4/t2 + đóng gói zip). Chỉ chạy Ô3 thôi thì có thể hạ xuống 15.
KEYS_REPO = "BachDaThan/yui-keys"   # repo HF RIÊNG TƯ tự tạo để cất key (dã mã hoá): quét Kaggle Secrets 1 lần duy nhất, các phiên sau lấy từ đây (chỉ tốn 1 lượt Kaggle cho HF_TOKEN)
REFRESH_KEYS = False    # True = quét lại Secrets rồi ghi đè repo (dùng khi bạn thêm/đổi key mới); xong nhớ đặt lại False
MODE = "cpu"            # "cpu" = Ô2 (nạp HF_TOKEN + mọi key)  |  "gpu" = Ô3/Ô4 (chỉ cần HF_TOKEN -> chỉ 1 lần gọi Kaggle)
MODEL_SIZE = "1B"
BITNET = 64             # v6.4: BitNet b1.58 (ternary). Số = cỡ nhóm scale (bội của 32, khuyên 64). 0 = tắt. Đã bật cho một model thì GIỮ MÃI (không quay lại float). qkv/o/expert thành {-1,0,+1}; router/head/embedding giữ float. Xuất file bằng export_yui.py --quant t2 (Ô4). Cần yui.cpp bản mới để chạy.
BIT_WARMUP = 2000       # v6.4: số bước trộn dần float -> ternary (để tiếp tục được checkpoint cũ). 0 = ternary ngay từ đầu (chỉ nên dùng với model MỚI)
VI_TOKENIZER = True     # v6.3: tokenizer tiếng Việt sâu (NFC + teencode/không dấu + từ khoá Boss/Yui, vocab 32000). CHỈ có tác dụng khi tạo model MỚI (NEW_MODEL = True): đổi tokenizer = phải pre-train lại từ đầu, model đang train dở giữ nguyên tokenizer cũ.
PERSONA_MIGRATE = False # v6.3: đổi data SFT CŨ trên HF sang xưng hô mới (gọi người dùng là "Boss"). Bản gốc được sao lưu ở backup/persona_anh/ trên cùng repo. False = chỉ chạy thử (in số dòng sẽ đổi + ví dụ, không ghi gì)
TEEN_AUG = "0.02"       # v6.3: thêm 2% đoạn teencode/không dấu vào data pre-train (đặt "0" để tắt)
DATA_REPO = "BachDaThan/TrainAI-web"
MODEL_REPO = f"BachDaThan/TrainAI-{MODEL_SIZE}"
CTX = "256"             # ngữ cảnh --L. Model MỚI: 256 (đủ cho tool + chuỗi 2 bước), 384 nếu muốn chuỗi dài hơn. Tiếp tục model đã train với L=128: đặt "128" (khi đó data nhiều bước tự bị lọc bớt)
# Tỉ lệ các loại hội thoại dạy tool (0 = tắt). py/recall/note/skill/mkskill/chain là MỚI ở v6. Model nhỏ (<=100M) nên giảm chain/skill, model 1B+ cứ để mặc định.
TOOL_MIX = "calc=3,time=1,search=3,py=4,recall=3,note=1,skill=1,mkskill=1,chain=2"
NAME = None             # None = repo tự đặt tên yui-<cỡ>-vN và tự TIẾP TỤC bản mới nhất; hoặc "ten-model"
NEW_MODEL = False       # True = tạo bản mới thay vì tiếp tục

# Key cần nạp (bớt dòng nào bạn không dùng để đỡ tốn lượt gọi). Mistral: bỏ dấu # nếu có key.
BASES = ["COHERE_API_KEY", "OPENROUTER_API_KEY", "TAVILY_API_KEY", "EXA_API_KEY",
         "LINKUP_API_KEY", "FIRECRAWL_API_KEY", "GEMINI_API_KEY"]  # , "MISTRAL_API_KEY"
PACE = 0.35             # giây nghỉ giữa các lần gọi Kaggle (tăng lên 0.6 nếu vẫn bị chặn)

os.environ.setdefault("SESSION_T0", str(time.time()))   # mốc bắt đầu phiên (chạy Ô1 ngay khi mở phiên)
os.environ.update({
    "RUN_HOURS": str(RUN_HOURS), "RESERVE_CPU": str(RESERVE_CPU), "RESERVE_GPU": str(RESERVE_GPU),
    "HF_REPO": DATA_REPO, "MODEL_PARAMS": MODEL_SIZE, "PARAMS": MODEL_SIZE, "MODEL_REPO": MODEL_REPO,
    "BITNET": str(BITNET), "BIT_WARMUP": str(BIT_WARMUP),
    "CTX": CTX, "FT_PAIRS": "40000",
    "DATA_KINDS": "tools,finetune",   # tools (không tốn LLM) chạy TRƯỚC; bỏ "pretrain": pre-train đã có FineWeb-2 tiếng Việt 1GB (Ô3), tạo 200M token bằng LLM là bất khả thi
    "MAX_FILES_PER_RUN": "100000",   # bỏ trần 40 file/lần: thời gian (DATA_MINUTES) mới là giới hạn thật              # tools = hội thoại dạy dùng calc/time/search (không tốn quota LLM)
    "TOOL_RESULT_CHARS": "120" if int(CTX) <= 128 else "300",   # độ dài kết quả tool phải vừa ngữ cảnh
    "TOOL_MIX": TOOL_MIX,                                        # make_data -> tool_data.py/tool_data_x.py; CTX ở trên cũng dùng để LỌC hội thoại dài quá ngữ cảnh
    "LLM_ORDER": "cohere,openrouter", "SCRAPE_ORDER": "tavily,exa,linkup,firecrawl",
    "FORCE_SFT": "0",
    "VI_TOKENIZER": "1" if VI_TOKENIZER else "0", "PERSONA_MIGRATE": "1" if PERSONA_MIGRATE else "0", "TEEN_AUG": TEEN_AUG,
    # ---- v6.2: đa dạng data ----
    "TEACHER_ROTATE": "3",            # xoay vòng giữa N model/teacher hàng đầu (1 = như cũ: luôn model xếp hạng nhất) -> bớt 'một giọng' của một teacher
    "PRETRAIN_NEW_TOKENS": "3M",      # v6.3: = 1T token = KHÔNG CÓ TRẦN THỰC TẾ -> make_pretrain chạy tới hết thời gian (PRETRAIN_SYNTH_SHARE) rồi dừng êm, không dừng sớm vì "đủ" 3M. "" = tắt
    "PRETRAIN_SYNTH_SHARE": "0.1",    # v6.3: phần ngân sách thời gian Ô2 dành cho make_pretrain.py (cũ 0.25 = 174 phút; 0.7 ≈ 490 phút của phiên 12h). Phần còn lại cho tools + finetune
    # v6.3: ngưỡng lọc (đã nới trong make_pretrain.py; chỉnh ở đây nếu vẫn loại nhiều/ít quá)
    "REP4_MIN": "10", "REP4_DIV": "60", "REP8_MAX": "3",   # lặp cụm: loại khi 1 cụm 4 từ lặp > max(REP4_MIN, số_từ/REP4_DIV) lần, hoặc 1 cụm 8 từ lặp > REP8_MAX lần
    "COPY_N": "10", "COPY_MAX_FRAC": "0.15",               # chép nguồn: chỉ GỌT các câu chép >=10 từ liên tiếp; loại cả bài nếu >15% câu bị chép
})
if NAME: os.environ["NAME"] = NAME
if NEW_MODEL: os.environ["NEW_MODEL"] = "1"

CACHE = "/tmp/.yui_secrets.json"      # /tmp: KHÔNG nằm trong /kaggle/working nên không bị lưu vào output notebook (tránh lộ key)
SHIM = "/tmp/yui_shim"
cache = {"vals": {}, "miss": []}
if os.path.exists(CACHE) and os.environ.get("RELOAD_SECRETS") != "1":
    try: cache = json.load(open(CACHE))
    except Exception: pass

if REFRESH_KEYS: cache["miss"] = []      # quét lại thật sự: quên các nhãn từng bị coi là "không có"
last_err = {}
_cli, _calls, _blocked = None, 0, False
def fetch(label):
    """Trả giá trị secret hoặc None. Nhớ cả giá trị lẫn nhãn KHÔNG tồn tại -> chạy lại ô này tốn 0 lượt gọi."""
    global _cli, _calls, _blocked
    if label in cache["vals"]: return cache["vals"][label]
    if label in cache["miss"] or _blocked: return None
    if _cli is None:
        from kaggle_secrets import UserSecretsClient
        _cli = UserSecretsClient()
    delay = 3
    for _ in range(6):
        try:
            time.sleep(PACE); _calls += 1
            v = _cli.get_secret(label)
            if v: cache["vals"][label] = v; return v
            cache["miss"].append(label); return None
        except Exception as e:
            m = str(e)
            low = m.lower()
            if "no user secrets exist" in low or "not found" in low or "does not exist" in low:   # CHẮC CHẮN không có / chưa bật -> không phải lỗi mạng
                last_err[label] = m[-170:]
                cache["miss"].append(label); return None
            # mọi lỗi khác (giới hạn tốc độ, "Connection error", timeout, 5xx...) là TẠM THỜI: chờ rồi thử lại, KHÔNG coi là "không có key"
            print(f"⏳ Kaggle tạm lỗi/giới hạn, nghỉ {delay}s rồi thử lại ({label}): {m[:60]}"); time.sleep(delay); delay = min(delay * 2, 60)
            continue
    _blocked = True
    print("⛔ Vẫn bị giới hạn. Đã lưu tiến độ: đợi 3-5 phút rồi chạy lại Ô 1, nó làm tiếp từ chỗ dừng (không gọi lại những gì đã có).")
    return None

def save_cache():
    json.dump(cache, open(CACHE, "w")); os.chmod(CACHE, 0o600)

# ---- GÓI KEY: 1 secret duy nhất chứa TẤT CẢ key (kể cả HF_TOKEN) -> chỉ tốn 1 lượt gọi Kaggle (2 nếu có YUI_KEYS2) ----
import re
def load_bundle():
    n = 0
    for lab in ("YUI_KEYS", "YUI_KEYS2"):
        if lab in cache["miss"]: cache["miss"].remove(lab)      # chạy trước khi tạo secret thì không bị nhớ nhầm là "không có"
        raw = fetch(lab)
        if not raw: break
        for line in re.split(r"[\r\n;]+", raw):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line: continue
            k, v = line.split("=", 1); k, v = k.strip(), v.strip().strip('"').strip("'")
            if k and v: os.environ[k] = v; cache["vals"][k] = v; n += 1
    return n
tok = os.environ.get("HF_TOKEN") or fetch("HF_TOKEN")
if tok: os.environ["HF_TOKEN"] = tok
else: print("⚠️ Không có HF_TOKEN trong Secrets (cần quyền write).")

def _fernet():
    """Khoá mã hoá suy ra từ HF_TOKEN của bạn: người chỉ đọc được repo (không có token) thì chỉ thấy dữ liệu mã hoá."""
    import base64
    from cryptography.fernet import Fernet
    from cryptography.hazmat.primitives import hashes
    from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
    kdf = PBKDF2HMAC(algorithm=hashes.SHA256(), length=32, salt=b"yui-keys-v1", iterations=200_000)
    return Fernet(base64.urlsafe_b64encode(kdf.derive(tok.encode())))

def _read_store(quiet=False):
    """Đọc + giải mã keys.enc trên HF -> dict ({} nếu chưa có / giải mã lỗi)."""
    try:
        from huggingface_hub import hf_hub_download
        raw = open(hf_hub_download(KEYS_REPO, "keys.enc", repo_type="model", token=tok), "rb").read()
    except Exception as e:
        if not quiet: print(f"(chưa có kho key mã hoá trên HF: {str(e)[:60]})")
        return {}
    try: return json.loads(_fernet().decrypt(raw))
    except Exception:
        print("⚠️ Có keys.enc nhưng giải mã không được (HF_TOKEN đã đổi?) -> bỏ qua bản cũ, quét lại từ Secrets."); return {}

def load_from_hf():
    if REFRESH_KEYS or not tok: return 0
    d = _read_store()
    for k, v in d.items():
        if v: os.environ[k] = v; cache["vals"][k] = v
    return len(d)

def push_to_hf():
    d = {k: v for k, v in cache["vals"].items() if k != "HF_TOKEN" and v}
    if d and not _blocked and tok:
        old = _read_store(quiet=True)        # GỘP với kho cũ: key đang bị tắt/không bật trong phiên này vẫn được giữ lại, không bị mất
        for k, v in old.items():
            if v and k not in d: d[k] = v; os.environ.setdefault(k, v); cache["vals"][k] = v
    if _blocked: print("⛔ Chưa quét xong (bị giới hạn) nên CHƯA lưu lên HF. Đợi 3-5 phút rồi chạy lại Ô1, nó làm tiếp."); return
    if not d or not tok: return
    try:
        blob = _fernet().encrypt(json.dumps(d).encode())
    except Exception as e:
        print(f"⚠️ Không mã hoá được ({str(e)[:60]}) nên KHÔNG đẩy key lên HF (không bao giờ đẩy bản thô)."); return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=tok)
        api.create_repo(KEYS_REPO, repo_type="model", private=True, exist_ok=True)
        try: api.update_repo_settings(KEYS_REPO, private=True, repo_type="model")   # ép riêng tư kể cả khi repo đã tồn tại
        except Exception: pass
        api.upload_file(path_or_fileobj=blob, path_in_repo="keys.enc", repo_id=KEYS_REPO, repo_type="model", commit_message="keys (encrypted)")
        try: api.delete_file("keys.json", repo_id=KEYS_REPO, repo_type="model")      # xoá bản thô cũ nếu từng có
        except Exception: pass
        print(f"🔐 Đã lưu {len(d)} biến key (MÃ HOÁ) vào repo riêng tư {KEYS_REPO}. Từ phiên sau Ô1 chỉ cần 1 lượt gọi Kaggle.")
    except Exception as e:
        print(f"⚠️ Không lưu được key lên HF ({str(e)[:80]}); phiên sau sẽ phải quét lại.")

n_hf = load_from_hf()
if n_hf: print(f"🔐 Nạp {n_hf} biến key từ repo riêng tư {KEYS_REPO} (tổng {_calls} lượt gọi Kaggle).")
n_bundle = 0 if n_hf else load_bundle()          # secret gộp YUI_KEYS (tuỳ chọn) chỉ thử khi kho HF chưa có
if n_bundle: print(f"📦 Gói key: nạp {n_bundle} biến từ secret YUI_KEYS (tổng {_calls} lượt gọi Kaggle).")

if MODE == "cpu" and not n_bundle and not n_hf:
    print("ℹ️ Lần đầu: quét từng key trong Secrets rồi tự lưu vào repo riêng tư trên HF (có thể mất vài phút; nếu bị chặn thì chạy lại Ô1, nó làm tiếp).")
    for base in BASES:
        v = fetch(base + "S")           # cách gọn nhất: 1 secret "COHERE_API_KEYS" chứa nhiều key, ngăn bằng dấu phẩy/xuống dòng (chỉ 1 lượt gọi)
        if v: os.environ[base + "S"] = v
        v = fetch(base)
        if v: os.environ[base] = v
        style, miss = None, 0           # key đánh số: COHERE_API_KEY1.. hoặc COHERE_API_KEY_1.., dừng khi thiếu 2 số liên tiếp
        for i in range(1, 31):
            got = None
            for lab in ([f"{base}{i}", f"{base}_{i}"] if style is None else [f"{base}{'_' if style else ''}{i}"]):
                got = fetch(lab)
                if got:
                    style = lab.startswith(base + "_"); os.environ[lab] = got; break
            miss = 0 if got else miss + 1
            if miss >= 2 or _blocked: break
        if not _blocked and not any(k.startswith(base) and v for k, v in os.environ.items()):
            e1 = last_err.get(base + "1") or last_err.get(base + "_1") or last_err.get(base + "S") or "?"
            print(f"⚠️ {base}: không đọc được key nào. Lỗi Kaggle: {e1}. Kiểm tra: đúng tên secret (vd {base}1) và đã BẬT (tick) cho notebook này trong Add-ons > Secrets.")
        save_cache()
    push_to_hf()
save_cache()

os.makedirs(SHIM, exist_ok=True)
open(f"{SHIM}/kaggle_secrets.py", "w").write(
    'import json, os\n'
    'class UserSecretsClient:\n'
    '    def get_secret(self, label):\n'
    '        v = os.environ.get(label)\n'
    '        if v: return v\n'
    '        try: v = json.load(open("/tmp/.yui_secrets.json"))["vals"].get(label)\n'
    '        except Exception: v = None\n'
    '        if v: return v\n'
    '        raise Exception("secret khong co trong cache (shim, khong goi API Kaggle)")\n')
os.environ["PYTHONPATH"] = SHIM + os.pathsep + os.environ.get("PYTHONPATH", "")

def _nkeys(b):
    return sum(len([x for x in v.replace("\n", ",").split(",") if x.strip()]) if k == b + "S" else 1
               for k, v in os.environ.items() if k.startswith(b) and v)
cnt = {b: _nkeys(b) for b in BASES}
print(f"✅ Xong Ô1 | MODE={MODE} | BitNet={'BẬT nhóm ' + str(BITNET) if BITNET else 'tắt'} | số lượt gọi Kaggle lần này: {_calls}")
if n_hf and MODE == "cpu":
    _zero = [b.replace("_API_KEY", "") for b, n in cnt.items() if n == 0]
    if _zero: print(f"ℹ️ Kho key trên HF chưa có: {', '.join(_zero)}. Nếu bạn ĐÃ có key các hãng này trong Secrets (và đã bật cho notebook): đặt REFRESH_KEYS = True ở đầu Ô1, chạy lại một lần, rồi đặt lại False.")
if MODE == "cpu":
    print("   Key đã nạp:", ", ".join(f"{b.replace('_API_KEY','')} {n}" for b, n in cnt.items()))
```

## Ô 2: Data (CPU) — Dùng lại data cũ, chỉ bù phần thiếu

```python
# %% ====== Ô 2: DATA (CPU) — dùng lại data cũ, chỉ bù phần thiếu ======
import os, sys, time, subprocess

REPO = os.environ.get("HF_REPO", "BachDaThan/TrainAI-web")
_left = float(os.environ.get("RUN_HOURS", "12")) * 60 - (time.time() - float(os.environ.get("SESSION_T0", time.time()))) / 60 - float(os.environ.get("RESERVE_CPU", "20"))
MINUTES = os.environ.get("DATA_MINUTES") or str(max(10, int(_left)))
_pre_on = bool(os.environ.get("PRETRAIN_NEW_TOKENS")) and os.path.exists("make_pretrain.py")
MIN_PRE = int(int(MINUTES) * float(os.environ.get("PRETRAIN_SYNTH_SHARE", "0.25"))) if _pre_on else 0
if MIN_PRE < 10: MIN_PRE = 0
MIN_FT = str(int(MINUTES) - MIN_PRE)
print(f"⏱ Ô2 được chạy tối đa {MINUTES} phút (make_data {MIN_FT} phút" + (f" + make_pretrain {MIN_PRE} phút)" if MIN_PRE else ")"))

def sh(cmd, retries=0, wait=45):
    for i in range(retries + 1):
        print("\n$", " ".join(cmd), flush=True)
        if subprocess.run(cmd).returncode == 0: return True
        if i < retries:
            w = wait * (i + 1); print(f"⚠️ Lỗi, nghỉ {w}s rồi thử lại ({i + 1}/{retries})"); time.sleep(w)
    return False

print("🧭 Kế hoạch (chưa tạo gì):")
sh([sys.executable, "make_data.py", "auto", "--repo", REPO, "--plan", "--max_minutes", MIN_FT])
print("\n🚀 Chạy thật:")
ok = sh([sys.executable, "make_data.py", "auto", "--repo", REPO, "--max_minutes", MIN_FT], retries=2)
if os.path.exists("persona_fix.py"):
    sh([sys.executable, "persona_fix.py", "--repo", REPO, "--tag", os.environ.get("MODEL_PARAMS", "1B")] + (["--apply"] if os.environ.get("PERSONA_MIGRATE") == "1" else []))
print("🎉 Data đã sẵn sàng trên HF!" if ok else "⚠️ make_data báo lỗi — xem log ở trên (phần đã tạo vẫn được giữ, chạy lại sẽ làm tiếp).")

if MIN_PRE:
    sh([sys.executable, "make_pretrain.py", "--plan"])
    ok_p = sh([sys.executable, "make_pretrain.py", "--repo", REPO, "--max_minutes", str(MIN_PRE)], retries=1)
    print("📗 Pre-train tổng hợp xong." if ok_p else "⚠️ make_pretrain báo lỗi — phần đã tạo vẫn được giữ.")
if os.path.exists("diversity_audit.py"):
    sh([sys.executable, "diversity_audit.py"])

if os.environ.get("RUN_SELFTEST", "1") == "1":
    try: import pytest
    except ImportError: sh([sys.executable, "-m", "pip", "install", "-q", "pytest"])
    ok_t = sh([sys.executable, "-m", "pytest", "-q", "-x", "test_yui_sandbox.py", "test_yui_graph.py", "test_yui_mind.py"])
    print("✅ Tự kiểm tra qua" if ok_t else "⚠️ Tự kiểm tra có lỗi — data dạy tool có thể lệch với runtime, xem log trên trước khi train.")
    sh([sys.executable, "tool_data_x.py", "--demo", "6"])

try:
    import re
    from huggingface_hub import HfApi, hf_hub_download
    tag, tok = os.environ.get("MODEL_PARAMS", "1B"), os.environ.get("HF_TOKEN")
    for rt in ("dataset", "model"):
        try: fl = HfApi(token=tok).list_repo_files(REPO, repo_type=rt); break
        except Exception: fl = []
    fl = [f for f in fl if re.match(rf"^Fine-tune/{re.escape(tag)}/\d+\.txt$", f)]
    n_tool = n_all = 0
    per = {k: 0 for k in ("calc", "time", "search", "py", "recall", "note", "skill", "mkskill")}
    for f in fl:
        t = open(hf_hub_download(REPO, f, repo_type=rt, token=tok), encoding="utf-8", errors="ignore").read()
        n_tool += t.count("<tool>"); n_all += t.count('"messages"')
        for k in per: per[k] += len(re.findall(rf"<tool>{k}[ <]", t))
    print(f"📊 Fine-tune/{tag}: {len(fl)} file, ~{n_all} hội thoại, trong đó ~{n_tool} lệnh gọi <tool>")
    print("   theo loại:", ", ".join(f"{k} {v}" for k, v in per.items()))
    if n_tool and per["py"] + per["recall"] == 0:
        print("⚠️ Chưa có data py/recall: data cũ tạo trước v6. Chạy lại Ô2: make_data v6 tự bù riêng các loại mới (py/recall/note/skill/chain) dù đã đủ calc/time/search, rồi SFT lại (FORCE_SFT=\"1\" ở Ô1).")
except Exception as e:
    print(f"(bỏ qua bước kiểm tra: {str(e)[:80]})")
```

## Ô 3: Pre-train (GPU T4 — DDP 2 GPU + BitNet)

```python
# %% ================= Ô 3: PRE-TRAIN (GPU T4) =================
import os, sys, time, subprocess
import torch

PARAMS, MODEL_REPO = os.environ["PARAMS"], os.environ["MODEL_REPO"]
DATA_REPO, CTX = os.environ["HF_REPO"], os.environ.get("CTX", "256")
PRE_OUT = "/kaggle/working/run_pre"
_left = float(os.environ.get("RUN_HOURS", "12")) * 60 - (time.time() - float(os.environ.get("SESSION_T0", time.time()))) / 60 - float(os.environ.get("RESERVE_GPU", "75"))
MAX_MIN = os.environ.get("MAX_MINUTES") or str(max(10, int(_left)))
print(f"⏱ Pre-train chạy tối đa {MAX_MIN} phút (~{int(MAX_MIN)/60:.1f}h)")
name_args = (["--name", os.environ["NAME"]] if os.environ.get("NAME") else []) + (["--new_model"] if os.environ.get("NEW_MODEL") == "1" else [])

cmd = [sys.executable, "-u", "train_lm.py", "--out", PRE_OUT, "--langs", "vi,web",
       "--hf_dataset", "HuggingFaceFW/fineweb-2:vie_Latn:text", "--hf_mb", "3000", "--mb", "30",
       "--size", PARAMS, "--amp", "--fast_moe", "--steps", "200000", "--max_minutes", MAX_MIN,
       "--hf_repo", MODEL_REPO, "--data_repo", DATA_REPO, "--make_data", "off",
       "--hf_every", "1000", "--hf_squash", "3", "--L", CTX, *name_args,
       "--quality_filter", "on", "--teen_aug", os.environ.get("TEEN_AUG", "0.02"),
       "--opt", "adamw8bit", "--B", "8", "--accum", "4"]
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "bitsandbytes"], check=False)
if os.environ.get("NEW_MODEL") == "1" and os.environ.get("VI_TOKENIZER", "1") == "1":
    cmd += ["--vi_tokenizer", "--vocab", "32000"]

# v6.4: BitNet b1.58 (ternary). --fast_moe phải tắt (bmm bỏ qua forward của BitLinear); train_lm.py cũng tự tắt, ở đây bỏ luôn cho rõ.
if os.environ.get("BITNET", "0") not in ("", "0"):
    cmd += ["--bitnet", os.environ["BITNET"], "--bit_warmup", os.environ.get("BIT_WARMUP", "2000")]
    if "--fast_moe" in cmd: cmd.remove("--fast_moe")
    print(f"🧊 BitNet BẬT: nhóm {os.environ['BITNET']}, fade-in {os.environ.get('BIT_WARMUP', '2000')} bước (fast_moe tắt)")

# Cấu hình DDP cho 2 GPU
n_gpu = torch.cuda.device_count()
cmd[cmd.index("--B") + 1] = "2"
cmd[cmd.index("--accum") + 1] = "8" if n_gpu >= 2 else "16"   # B*accum*số GPU = 32
launcher = [sys.executable, "-u", "train_lm.py"]
if n_gpu >= 2:
    launcher = [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nproc_per_node=2", "train_lm.py"]
cmd = launcher + cmd[3:]      # bỏ [python, -u, train_lm.py] cũ

print("🔥", " ".join(cmd))
rc = subprocess.run(cmd).returncode
print("✅ Pre-train xong." if rc == 0 else f"ℹ️ Pre-train dừng (mã {rc}); checkpoint đã được đẩy lên HF, chạy lại ô này để tiếp tục.")
```

## Ô 4: SFT + Export .yui (GPU T4)

```python
# %% ======== Ô 4: SFT + EXPORT .yui (GPU T4) ========
import os, sys, subprocess

PARAMS, MODEL_REPO = os.environ["PARAMS"], os.environ["MODEL_REPO"]
DATA_REPO = os.environ["HF_REPO"]
SFT_OUT = "/kaggle/working/run_sft"
os.makedirs(SFT_OUT, exist_ok=True)

cmd = [sys.executable, "train_sft.py", "--out", SFT_OUT, "--hf_repo", MODEL_REPO, "--data_repo", DATA_REPO]
if os.environ.get("NAME"): cmd += ["--src", os.environ["NAME"]]
if os.environ.get("FORCE_SFT") == "1": cmd.append("--force")
print("🎯", " ".join(cmd))
rc = subprocess.run(cmd).returncode
if rc != 0:
    raise SystemExit(f"❌ train_sft.py lỗi (mã {rc}) — xem log ở trên. Không export để tránh ra file rỗng.")

# v6.4: model train với BitNet -> xuất t2 (ternary) + int8 dự phòng; model thường -> int8 + q4 như cũ.
# t2 cần bitnet_pack.py cạnh export_yui.py; nhóm scale lấy từ config.json ("bitnet") nên khớp lúc train.
_bit = os.environ.get("BITNET", "0") not in ("", "0")
for q in (("t2", "int8") if _bit else ("int8", "q4")):
    subprocess.run([sys.executable, "export_yui.py", "--out", SFT_OUT, "--quant", q], check=False)

# ======================= v6.5: XEM ẢNH (tuỳ chọn) =======================
# Chuỗi: tạo data ảnh tiếng Việt (Gemini TỰ CHỌN model mới nhất còn dùng được, hết quota thì tụt xuống model thấp hơn)
#        -> giai đoạn 1: chỉ train projector -> giai đoạn 2: mở khoá LM -> xuất yui-vision.onnx + file .yui biết xem ảnh.
# Không đụng tới file chữ đã xuất ở trên: file xem ảnh có tên riêng <tên>-vision_<quant>.yui. Phải dùng CẶP: .yui "-vision" + yui-vision.onnx.
VISION = False          # True = bật
VISION_IMAGES = 1500    # số ảnh tối đa tạo data (chạy lại phiên sau sẽ làm tiếp và cộng dồn, data được cất trên HF)
VISION_SRC, VISION_SPLIT = "nlphuji/flickr30k", "test"   # nguồn ảnh công khai trên HF (tự kiểm tra giấy phép); đổi tên nếu bạn dùng dataset khác
VISION_MAX_MIN = 120    # tổng phút tối đa cho train ảnh (chia đôi cho 2 giai đoạn). Nhớ chừa đủ RESERVE_GPU ở Ô1.

if VISION:
    import json, shutil
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "transformers", "pillow", "datasets", "onnx", "onnxruntime"], check=False)
    VD = "/kaggle/working/vdata"
    try:   # lấy data ảnh đã tạo ở phiên trước
        from huggingface_hub import snapshot_download
        snapshot_download(repo_id=DATA_REPO, repo_type="dataset", allow_patterns=["vision/*", "vision/**/*"], local_dir="/kaggle/working/_vd")
        if os.path.isdir("/kaggle/working/_vd/vision"):
            shutil.copytree("/kaggle/working/_vd/vision", VD, dirs_exist_ok=True)
    except Exception as e:
        print(f"(chưa có data ảnh cũ trên HF: {str(e)[:80]})")
    done = os.path.join(VD, "done.txt")
    have = sum(1 for _ in open(done)) if os.path.exists(done) else 0
    if have < VISION_IMAGES and os.environ.get("GEMINI_API_KEY"):
        subprocess.run([sys.executable, "make_vision_data.py", "--hf_dataset", VISION_SRC, "--split", VISION_SPLIT, "--n", str(VISION_IMAGES), "--out", VD], check=False)
        try:
            from huggingface_hub import upload_folder
            upload_folder(folder_path=VD, path_in_repo="vision", repo_id=DATA_REPO, repo_type="dataset", commit_message="data ảnh tiếng Việt")
        except Exception as e:
            print(f"(không đẩy được data ảnh lên HF: {str(e)[:80]})")
    elif have < VISION_IMAGES:
        print("⚠ Không có GEMINI_API_KEY trong môi trường -> bỏ qua bước tạo thêm data ảnh (dùng phần đã có, nếu có).")
    jl = os.path.join(VD, "data.jsonl")
    if os.path.exists(jl) and os.path.getsize(jl) > 0:
        half = str(max(5, VISION_MAX_MIN // 2))
        base = [sys.executable, "train_vision.py", "--out", SFT_OUT, "--data", jl, "--images_dir", VD, "--max_minutes", half, "--push_repo", MODEL_REPO]
        ok = subprocess.run(base + ["--stage", "1", "--epochs", "2"]).returncode == 0
        ok = ok and subprocess.run(base + ["--stage", "2", "--epochs", "1"]).returncode == 0
        if ok:
            subprocess.run([sys.executable, "export_vision.py", "--out", SFT_OUT, "--dst", "/kaggle/working/yui-vision.onnx"], check=False)
            _name = json.load(open(os.path.join(SFT_OUT, "config.json"), encoding="utf-8")).get("name") or "yui"
            for q in (("t2", "int8") if _bit else ("int8", "q4")):
                subprocess.run([sys.executable, "export_yui.py", "--out", SFT_OUT, "--ckpt", "vision_best.pt", "--quant", q,
                                "--dst", f"/kaggle/working/{_name}-vision_{q}.yui"], check=False)
            print("🖼 Xem ảnh: tải yui-vision.onnx + <tên>-vision_*.yui về điện thoại. Xem dòng 'chênh' của giai đoạn 1: gần 0 nghĩa là model vẫn bỏ qua ảnh -> thêm data/ảnh rồi chạy lại.")
        else:
            print("❌ train ảnh lỗi (xem log trên). File chữ vẫn xuất bình thường.")
    else:
        print("⚠ Chưa có data ảnh (vdata/data.jsonl) nên bỏ qua train ảnh.")


import zipfile, glob
RUNTIME = ["yui_agent.py", "yui_sandbox.py", "yui_vault.py", "yui_rag.py", "yui_graph.py", "yui_skills.py", "yui_orch.py", "yui_web.py", "yui_condense.py", "yui_kb.py", "yui_task.py",
           "agent.example.json", "graphify_bench.py", "LICENSE", "yui.cpp"]
try:
    with zipfile.ZipFile("/kaggle/working/yui_runtime.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in RUNTIME:
            if os.path.exists(f): z.write(f)
        for f in glob.glob("skills_example/*.md"): z.write(f)
        z.writestr("CHAY_THU.txt", "python yui_agent.py status\npython yui_sandbox.py \"print(2**100)\"\npython graphify_bench.py --seeds 5\n"
                   "Cấu hình: chép agent.example.json thành agent.json, đặt memory.dir = thư mục lưu chat; mã hoá: export YUI_KEY=...\n"
                   "File *_t2.yui (BitNet) cần yui.cpp bản mới: g++ -O3 -std=c++17 -pthread yui.cpp -o yui && ./yui --model ten_t2.yui\n")
    print("📦 /kaggle/working/yui_runtime.zip (bộ chạy: tool + sandbox + RAG/graphify + skill mẫu + yui.cpp)")
except Exception as e:
    print(f"(bỏ qua đóng gói bộ chạy: {str(e)[:80]})")
print("🎉 Xong. File .yui nằm ở thư mục hiện tại; chat thử bản int8 trước khi dùng " + ("t2" if _bit else "q4") + ". Nhớ tải kèm yui_runtime.zip.")
```
