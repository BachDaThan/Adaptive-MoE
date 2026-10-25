# Dùng cho Hugging Face Space (Gradio, CPU miễn phí): tải best.pt, config.json, tokenizer.json từ repo dataset của bạn rồi mở web.
# Cần đặt secret HF_TOKEN trong Settings của Space (repo dataset đang để riêng tư). Sửa HF_REPO nếu repo của bạn tên khác.
import sys
HF_REPO = "BachDaThan/TrainAI"
sys.argv = [sys.argv[0], "--web", "--out", "model", "--hf_repo", HF_REPO, "--k", "2"]
from chat import main
main()
