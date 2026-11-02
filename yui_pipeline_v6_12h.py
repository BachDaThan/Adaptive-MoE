# ============================================================
# YUI PIPELINE v6.2 (chạy 12 tiếng) — 4 Ô (copy từng ô vào Kaggle). Khớp đúng repo Adaptive-MoE (đã có yui_sandbox / yui_rag / yui_graph / yui_skills / yui_orch / yui_web / yui_kb).
#   Ô1  Env + nạp Secrets CHỐNG RATE LIMIT (chạy ở MỌI phiên; MODE="cpu" hoặc "gpu")
#   Ô2  CPU  — make_data auto (dùng lại data cũ, chỉ bù phần thiếu) + TỰ KIỂM TRA data dạy tool + báo cáo theo từng loại tool
#   Ô3  GPU  — Pre-train
#   Ô4  GPU  — SFT + Export .yui + đóng gói bộ chạy (yui_runtime.zip)
# Công cụ model được DẠY (cú pháp một dòng <tool>tên đối-số</tool> / <result>kết-quả</result>):
#   calc, time, search            như v5
#   py        mini-sandbox tự chế: tính CHÍNH XÁC (số lớn, phân số, số nguyên tố, ngày tháng, đếm chữ...). Kết quả trong data do chính sandbox tính.
#   recall, note   tìm lại lịch sử chat / ghi chú (graphify: câu gọn, sạch cũ/trùng) và ghi nhớ
#   skill, mkskill dùng / tự tạo skill
#   chuỗi nhiều bước (recall -> calc -> trả lời; skill -> py -> trả lời; dạng "[tiến độ: ...]" và "[tóm tắt: ...]" đúng khuôn bộ điều phối yui_orch)
# Lưu ý ngữ cảnh: các ví dụ nhiều bước dài hơn; CTX=256 đủ cho hầu hết, muốn chuỗi 3-5 bước thoải mái thì đặt CTX="384" (model MỚI; đừng đổi giữa chừng với model đã train).
# ============================================================


# %% ============ Ô 1: ENV + SECRETS (CHỐNG RATE LIMIT) ============
import os, sys, json, time

RUN_HOURS = 12          # ⏱ TỔNG thời gian phiên Kaggle bạn muốn dùng (Kaggle cho tối đa 12h). Các ô tự tính phút còn lại và dừng êm TRƯỚC giới hạn.
RESERVE_CPU = 20        # phút chừa cuối phiên CPU (tự kiểm tra + báo cáo)
RESERVE_GPU = 75        # phút chừa cuối phiên GPU cho Ô4 (SFT + export int8/q4 + đóng gói zip). Chỉ chạy Ô3 thôi thì có thể hạ xuống 15.
KEYS_REPO = "BachDaThan/yui-keys"   # repo HF RIÊNG TƯ tự tạo để cất key (đã mã hoá): quét Kaggle Secrets 1 lần duy nhất, các phiên sau lấy từ đây (chỉ tốn 1 lượt Kaggle cho HF_TOKEN)
REFRESH_KEYS = False    # True = quét lại Secrets rồi ghi đè repo (dùng khi bạn thêm/đổi key mới); xong nhớ đặt lại False
MODE = "cpu"            # "cpu" = Ô2 (nạp HF_TOKEN + mọi key)  |  "gpu" = Ô3/Ô4 (chỉ cần HF_TOKEN -> chỉ 1 lần gọi Kaggle)
MODEL_SIZE = "1B"
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
    "CTX": CTX, "FT_PAIRS": "40000",
    "DATA_KINDS": "tools,finetune",   # tools (không tốn LLM) chạy TRƯỚC; bỏ "pretrain": pre-train đã có FineWeb-2 tiếng Việt 1GB (Ô3), tạo 200M token bằng LLM là bất khả thi
    "MAX_FILES_PER_RUN": "100000",   # bỏ trần 40 file/lần: thời gian (DATA_MINUTES) mới là giới hạn thật              # tools = hội thoại dạy dùng calc/time/search (không tốn quota LLM)
    "TOOL_RESULT_CHARS": "120" if int(CTX) <= 128 else "300",   # độ dài kết quả tool phải vừa ngữ cảnh
    "TOOL_MIX": TOOL_MIX,                                        # make_data -> tool_data.py/tool_data_x.py; CTX ở trên cũng dùng để LỌC hội thoại dài quá ngữ cảnh
    "LLM_ORDER": "cohere,openrouter", "SCRAPE_ORDER": "tavily,exa,linkup,firecrawl",
    "FORCE_SFT": "0",
    # ---- v6.2: đa dạng data ----
    "TEACHER_ROTATE": "3",            # xoay vòng giữa N model/teacher hàng đầu (1 = như cũ: luôn model xếp hạng nhất) -> bớt 'một giọng' của một teacher
    "PRETRAIN_NEW_TOKENS": "3M",      # make_pretrain.py: thêm ~3M token văn bản DÀI kiểu sách giáo khoa/báo/truyện (đa lĩnh vực, MinHash bỏ trùng) cạnh FineWeb-2. "" = tắt
    "PRETRAIN_SYNTH_SHARE": "0.25",   # phần ngân sách thời gian Ô2 dành cho make_pretrain.py (phần còn lại cho tools + finetune)
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
# Tạo secret tên YUI_KEYS (Add-ons > Secrets), giá trị dạng (mỗi dòng 1 hãng, nhiều key ngăn bằng dấu phẩy; hoặc ngăn các dòng bằng dấu ; nếu ô nhập chỉ cho 1 dòng):
#   HF_TOKEN=hf_xxx
#   COHERE_API_KEYS=key1,key2,key3
#   OPENROUTER_API_KEYS=key1,key2
#   TAVILY_API_KEYS=...    EXA_API_KEYS=...    LINKUP_API_KEYS=...    FIRECRAWL_API_KEYS=...    GEMINI_API_KEYS=...
# (viết kiểu đánh số COHERE_API_KEY1=xxx mỗi dòng cũng được). Nếu Kaggle chê quá dài: chia phần sau sang secret YUI_KEYS2.
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

# Shim: ép MỌI tiến trình con (make_data.py, train_*.py) đọc secret từ cache/env thay vì gọi Kaggle lần nữa.
# (make_data.py tự quét tới ~50 nhãn mỗi lần chạy; Ô2 cũ chạy nó nhiều lần -> đó là nguyên nhân bị "Too many requests".)
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

def _nkeys(b):   # đếm key thật: biến NAME_KEYS tính theo số key ngăn bằng dấu phẩy, biến đánh số tính 1
    return sum(len([x for x in v.replace("\n", ",").split(",") if x.strip()]) if k == b + "S" else 1
               for k, v in os.environ.items() if k.startswith(b) and v)
cnt = {b: _nkeys(b) for b in BASES}
print(f"✅ Xong Ô1 | MODE={MODE} | số lượt gọi Kaggle lần này: {_calls}")
if n_hf and MODE == "cpu":
    _zero = [b.replace("_API_KEY", "") for b, n in cnt.items() if n == 0]
    if _zero: print(f"ℹ️ Kho key trên HF chưa có: {', '.join(_zero)}. Nếu bạn ĐÃ có key các hãng này trong Secrets (và đã bật cho notebook): đặt REFRESH_KEYS = True ở đầu Ô1, chạy lại một lần, rồi đặt lại False.")
if MODE == "cpu":
    print("   Key đã nạp:", ", ".join(f"{b.replace('_API_KEY','')} {n}" for b, n in cnt.items()))


# %% ====== Ô 2: DATA (CPU) — dùng lại data cũ, chỉ bù phần thiếu ======
# make_data.py auto tự: đếm data đang có trong Fine-tune/<cỡ>/ và data/<cỡ>/, chỉ tạo phần còn thiếu, tạo từ đầu nếu cỡ mới,
# và (khi DATA_KINDS có 'tools') bù hội thoại dạy dùng tool cho đủ ~10% mục tiêu. Tự upload, KHÔNG cần lệnh push riêng.
import os, sys, time, subprocess

REPO = os.environ.get("HF_REPO", "BachDaThan/TrainAI-web")
_left = float(os.environ.get("RUN_HOURS", "12")) * 60 - (time.time() - float(os.environ.get("SESSION_T0", time.time()))) / 60 - float(os.environ.get("RESERVE_CPU", "20"))
MINUTES = os.environ.get("DATA_MINUTES") or str(max(10, int(_left)))   # mặc định: dùng hết ngân sách RUN_HOURS (đặt DATA_MINUTES để ép số khác)
_pre_on = bool(os.environ.get("PRETRAIN_NEW_TOKENS")) and os.path.exists("make_pretrain.py")
MIN_PRE = int(int(MINUTES) * float(os.environ.get("PRETRAIN_SYNTH_SHARE", "0.25"))) if _pre_on else 0
if MIN_PRE < 10: MIN_PRE = 0
MIN_FT = str(int(MINUTES) - MIN_PRE)      # phần còn lại cho make_data (tools + finetune)
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
print("🎉 Data đã sẵn sàng trên HF!" if ok else "⚠️ make_data báo lỗi — xem log ở trên (phần đã tạo vẫn được giữ, chạy lại sẽ làm tiếp).")

# Văn bản pre-train tổng hợp (dài, đa lĩnh vực) — bổ sung cho FineWeb-2 ở Ô3. Ghi vào data/<cỡ>/N.txt, đánh số tiếp, không đụng file cũ.
if MIN_PRE:
    sh([sys.executable, "make_pretrain.py", "--plan"])
    ok_p = sh([sys.executable, "make_pretrain.py", "--repo", REPO, "--max_minutes", str(MIN_PRE)], retries=1)
    print("📗 Pre-train tổng hợp xong." if ok_p else "⚠️ make_pretrain báo lỗi — phần đã tạo vẫn được giữ.")
if os.path.exists("diversity_audit.py"):      # đo độ đa dạng thật (trùng, distinct-n, rập khuôn mở đầu, khuôn tool...)
    sh([sys.executable, "diversity_audit.py"])

# Tự kiểm tra (nhanh, không cần GPU): sandbox + graphify + data dạy tool khớp nhau; kết quả py trong data phải bằng kết quả sandbox thật.
if os.environ.get("RUN_SELFTEST", "1") == "1":
    try: import pytest  # noqa: F401
    except ImportError: sh([sys.executable, "-m", "pip", "install", "-q", "pytest"])
    ok_t = sh([sys.executable, "-m", "pytest", "-q", "-x", "test_yui_sandbox.py", "test_yui_graph.py", "test_yui_mind.py"])
    print("✅ Tự kiểm tra qua" if ok_t else "⚠️ Tự kiểm tra có lỗi — data dạy tool có thể lệch với runtime, xem log trên trước khi train.")
    sh([sys.executable, "tool_data_x.py", "--demo", "6"])      # xem thử vài hội thoại mẫu (py/recall/note/skill/chain)

# Kiểm tra: bao nhiêu hội thoại trong Fine-tune/<cỡ>/ đã có dạng dùng tool, theo từng loại?
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


# %% ================= Ô 3: PRE-TRAIN (GPU T4) =================
# Chạy Ô1 với MODE="gpu" trước. --make_data off: KHÔNG tự tạo data trong phiên GPU (đỡ đốt giờ GPU và lượt gọi secret).
import os, sys, time, subprocess

PARAMS, MODEL_REPO = os.environ["PARAMS"], os.environ["MODEL_REPO"]
DATA_REPO, CTX = os.environ["HF_REPO"], os.environ.get("CTX", "256")
PRE_OUT = "/kaggle/working/run_pre"
_left = float(os.environ.get("RUN_HOURS", "12")) * 60 - (time.time() - float(os.environ.get("SESSION_T0", time.time()))) / 60 - float(os.environ.get("RESERVE_GPU", "75"))
MAX_MIN = os.environ.get("MAX_MINUTES") or str(max(10, int(_left)))   # 12h -> ~645 phút pre-train, chừa ~75 phút cho Ô4
print(f"⏱ Pre-train chạy tối đa {MAX_MIN} phút (~{int(MAX_MIN)/60:.1f}h)")
name_args = (["--name", os.environ["NAME"]] if os.environ.get("NAME") else []) + (["--new_model"] if os.environ.get("NEW_MODEL") == "1" else [])

cmd = [sys.executable, "-u", "train_lm.py", "--out", PRE_OUT, "--langs", "vi,web",
       "--hf_dataset", "HuggingFaceFW/fineweb-2:vie_Latn:text", "--hf_mb", "1000", "--mb", "30",
       "--size", PARAMS, "--amp", "--fast_moe", "--steps", "200000", "--max_minutes", MAX_MIN,
       "--hf_repo", MODEL_REPO, "--data_repo", DATA_REPO, "--make_data", "off",
       "--hf_every", "1000", "--hf_squash", "3", "--L", CTX, *name_args]
print("🔥", " ".join(cmd))
rc = subprocess.run(cmd).returncode
print("✅ Pre-train xong." if rc == 0 else f"ℹ️ Pre-train dừng (mã {rc}); checkpoint đã được đẩy lên HF, chạy lại ô này để tiếp tục.")


# %% ======== Ô 4: SFT + EXPORT .yui (GPU T4) ========
# train_sft.py tự quét Fine-tune/<cỡ>/N.txt (gồm cả hội thoại dùng tool), tự kéo model gốc từ HF, chỉ train khi có data/model mới.
import os, sys, subprocess

PARAMS, MODEL_REPO = os.environ["PARAMS"], os.environ["MODEL_REPO"]
DATA_REPO = os.environ["HF_REPO"]
SFT_OUT = "/kaggle/working/run_sft"
os.makedirs(SFT_OUT, exist_ok=True)

cmd = [sys.executable, "train_sft.py", "--out", SFT_OUT, "--hf_repo", MODEL_REPO, "--data_repo", DATA_REPO]
if os.environ.get("NAME"): cmd += ["--src", os.environ["NAME"]]
if os.environ.get("FORCE_SFT") == "1": cmd.append("--force")     # đặt FORCE_SFT="1" ở Ô1 nếu muốn train lại dù không có gì mới
print("🎯", " ".join(cmd))
rc = subprocess.run(cmd).returncode
if rc != 0:
    raise SystemExit(f"❌ train_sft.py lỗi (mã {rc}) — xem log ở trên. Không export để tránh ra file rỗng.")

for q in ("int8", "q4"):   # tên file tự lấy từ tên model: <tên>_<quant>.yui
    subprocess.run([sys.executable, "export_yui.py", "--out", SFT_OUT, "--quant", q], check=False)
# Đóng gói BỘ CHẠY để mang về máy/điện thoại: agent + sandbox + kho (RAG/graphify/mã hoá) + skill mẫu + cấu hình mẫu. Model .yui tải riêng (file lớn).
import zipfile, glob
RUNTIME = ["yui_agent.py", "yui_sandbox.py", "yui_vault.py", "yui_rag.py", "yui_graph.py", "yui_skills.py", "yui_orch.py", "yui_web.py", "yui_condense.py", "yui_kb.py", "yui_task.py",
           "agent.example.json", "graphify_bench.py", "LICENSE"]
try:
    with zipfile.ZipFile("/kaggle/working/yui_runtime.zip", "w", zipfile.ZIP_DEFLATED) as z:
        for f in RUNTIME:
            if os.path.exists(f): z.write(f)
        for f in glob.glob("skills_example/*.md"): z.write(f)
        z.writestr("CHAY_THU.txt", "python yui_agent.py status\npython yui_sandbox.py \"print(2**100)\"\npython graphify_bench.py --seeds 5\n"
                   "Cấu hình: chép agent.example.json thành agent.json, đặt memory.dir = thư mục lưu chat; mã hoá: export YUI_KEY=...\n")
    print("📦 /kaggle/working/yui_runtime.zip (bộ chạy: tool + sandbox + RAG/graphify + skill mẫu)")
except Exception as e:
    print(f"(bỏ qua đóng gói bộ chạy: {str(e)[:80]})")
print("🎉 Xong. File .yui nằm ở thư mục hiện tại; chat thử bản int8 trước khi dùng q4. Nhớ tải kèm yui_runtime.zip.")
