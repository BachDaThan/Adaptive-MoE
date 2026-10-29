"""
kaggle_pipeline.py - chạy trọn quy trình trên Kaggle: tạo data -> pretrain -> fine-tune -> xuất .yui, chỉ cần chỉnh vài biến ở đầu file.

    %run kaggle_pipeline.py                       # trong ô notebook (hoặc  !python kaggle_pipeline.py)
    STAGES=data python kaggle_pipeline.py         # CHỈ tạo data: chạy ở phiên CPU/máy khác để KHÔNG đốt giờ GPU của Kaggle (data chỉ cần mạng + CPU)
    STAGES=pretrain,sft,export ...                # phiên GPU: chỉ train (data đã có trên repo)

Secrets cần có (Kaggle > Add-ons > Secrets): HF_TOKEN + key LLM: COHERE_API_KEY1..N, OPENROUTER_API_KEY1..N, MISTRAL_API_KEY1..N, GEMINI_API_KEY1..N (bao nhiêu cũng được, có ít nhất 1 loại;\nthứ tự dùng Cohere -> OpenRouter -> Mistral -> Gemini). Cào web (tuỳ chọn): TAVILY_API_KEY1..N, EXA_API_KEY1..N, LINKUP_API_KEY1..N, FIRECRAWL_API_KEY1..N (thứ tự Tavily -> Exa -> Linkup -> Firecrawl).
Tên model: tự đặt yui-<cỡ>-v<N> (không trùng repo/tài khoản HF) và lưu ở <MODEL_REPO>/models/<tên>/; đặt NAME = "tên-của-bạn" để tự chọn.
Chạy lại cùng lệnh trên phiên mới = TIẾP TỤC bản mới nhất; NEW_MODEL=True để tạo bản mới.
"""
import os
import subprocess
import sys
import time

# ===================== CHỈNH Ở ĐÂY =====================
PARAMS = os.environ.get("PARAMS", "1B")                       # số tham số: 50M, 250M, 1B, 3B, 7B...
MODEL_REPO = os.environ.get("MODEL_REPO", "BachDaThan/TrainAI-1B")   # nơi lưu model (models/<tên>/)
DATA_REPO = os.environ.get("HF_REPO", "BachDaThan/TrainAI-web")      # nơi lưu data (data/<cỡ>/ và Fine-tune/<cỡ>/)
NAME = os.environ.get("NAME") or None                          # None = tự đặt tên
NEW_MODEL = os.environ.get("NEW_MODEL") == "1"
CTX = os.environ.get("CTX", "256")                             # độ dài ngữ cảnh (--L); dùng công cụ cần >= 256
STEPS = os.environ.get("STEPS", "200000")
MAX_MIN = os.environ.get("MAX_MINUTES", "720")
OUT = os.environ.get("OUT", "/kaggle/working")
STAGES = os.environ.get("STAGES", "data,pretrain,sft,export").split(",")
DATA_KINDS = os.environ.get("DATA_KINDS", "finetune,pretrain")        # thêm ,tools nếu bot của bạn chạy qua yui_agent.py
# ========================================================

os.environ["HF_REPO"], os.environ["MODEL_PARAMS"], os.environ["DATA_KINDS"] = DATA_REPO, PARAMS, DATA_KINDS
import make_data
make_data.load_kaggle_secrets()   # trên Kaggle: tự lấy HF_TOKEN + key của mọi provider; ngoài Kaggle: tự export biến môi trường
print("Key LLM:", ", ".join(f"{lbl} {len(make_data.collect_keys(make_data.LLM_KEY_NAMES[n]))}" for n, lbl in make_data.LLM_LABELS.items()),
      "| Cào web:", ", ".join(f"{n.capitalize()} {len(make_data.collect_keys(n.upper() + '_API_KEY'))}" for n in ("tavily", "exa", "linkup", "firecrawl")))


def sh(cmd, retries=0, wait=30):
    for i in range(retries + 1):
        print("\n$", " ".join(cmd), flush=True)
        if subprocess.run(cmd).returncode == 0:
            return True
        if i < retries:
            print(f"⚠ lỗi, nghỉ {wait}s rồi thử lại ({i + 1}/{retries})", flush=True)
            time.sleep(wait)
    return False


name_args = (["--name", NAME] if NAME else []) + (["--new_model"] if NEW_MODEL else [])
PRE_OUT, SFT_OUT = f"{OUT}/run_pre", f"{OUT}/run_sft"

if "data" in STAGES:   # make_data.py tự lo 503/429/đổi model/nghỉ; khoá file chống chạy chồng
    sh([sys.executable, "make_data.py", "auto", "--max_minutes", os.environ.get("DATA_MINUTES", "120")], retries=3)

if "pretrain" in STAGES:
    sh([sys.executable, "-u", "train_lm.py", "--out", PRE_OUT, "--langs", "vi,web", "--hf_dataset", "HuggingFaceFW/fineweb-2:vie_Latn:text",
        "--hf_mb", "1000", "--mb", "30", "--size", PARAMS, "--amp", "--fast_moe", "--steps", STEPS, "--max_minutes", MAX_MIN, "--hf_repo", MODEL_REPO,
        "--data_repo", DATA_REPO, "--make_data", "off", "--hf_every", "2000", "--hf_squash", "3", "--L", CTX, *name_args])

if "sft" in STAGES:
    sh([sys.executable, "train_sft.py", "--out", SFT_OUT, "--hf_repo", MODEL_REPO, "--data_repo", DATA_REPO, "--force"] + (["--src", NAME] if NAME else []))   # train_sft.py không có --L: ngữ cảnh lấy từ config.json lúc pretrain

if "export" in STAGES:
    for q in ("int8", "q4"):
        sh([sys.executable, "export_yui.py", "--out", SFT_OUT, "--quant", q])   # tên file tự lấy từ tên model: <tên>_<quant>.yui
    print("\n✅ Xong. File .yui nằm ở thư mục hiện tại; kiểm: python verify_cpp.py ... và chat thử trước khi dùng bản q4.")
