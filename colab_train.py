# Dán vào MỘT ô Colab (đã ở trong thư mục repo có train_lm.py). Hiện log trực tiếp, tự lưu và đẩy checkpoint, Ctrl+C an toàn.
import subprocess, signal, sys
from huggingface_hub import notebook_login
notebook_login()

OUT_DIR = "/content/drive/MyDrive/adaptive_moe"
cmd = [sys.executable, "-u", "train_lm.py", "--out", OUT_DIR,
       "--langs", "vi", "--mb", "100", "--size", "small", "--amp",
       "--steps", "20000",            # muốn train tiếp sau khi xong: tăng lên 40000
       "--max_minutes", "120",
       "--hf_repo", "BachDaThan/TrainAI", "--hf_every", "2000"]
print("Chạy:", " ".join(cmd))
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
try:
    for line in p.stdout:
        print(line, end="")
    p.wait()
except KeyboardInterrupt:           # Stop trong Colab: chuyển tín hiệu cho train_lm.py để nó kịp lưu và đẩy lên HF
    p.send_signal(signal.SIGINT)
    for line in p.stdout:
        print(line, end="")
    p.wait()
print("mã thoát:", p.returncode)
