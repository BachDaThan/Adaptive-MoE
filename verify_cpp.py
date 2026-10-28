"""
verify_cpp.py - đối chiếu yui (C++) với PyTorch trên model THẬT của bạn. Chạy trong thư mục repo Adaptive-MoE.

    make
    python export_yui.py --out thu_muc_model --dst model.yui
    python verify_cpp.py --out thu_muc_model --model model.yui --yui ./yui

1) Tokenizer: mã hoá nhiều chuỗi bằng thư viện tokenizers và bằng ./yui --encode, đếm số chuỗi lệch (mong đợi 0).
2) Logits: chạy CtxLM của repo (float32, eval, k_cap = 1 và 2) trên một chuỗi id, so với ./yui --ids ... --logits.
   Mong đợi file float32: max|diff| < 1e-3 và argmax khớp ~100%. File int8: argmax khớp cao (thường > 90%), diff lớn hơn là bình thường.
   Với file int8 script chạy 2 lần: chế độ NHANH (kích hoạt lượng tử int8 theo khối 32, mặc định của yui) và chế độ --exact
   (kích hoạt float như PyTorch). --exact chỉ sai khác do trọng số int8; nếu chế độ nhanh lệch hẳn so với --exact thì báo lại.
"""
import argparse, json, os, subprocess, sys, tempfile
import numpy as np

SAMPLES = ["xin chào", "Hà Nội là thủ đô của Việt Nam.", "  hai  dấu   cách  ", "User: Yui ơi, nay trời đẹp ghê á!\nYui: Đẹp thía mà hong có ai dắt Yui đi dạo nè 🥺",
           "Hello world, it's 2024! Don't stop... 123,456.78", "line1\nline2\r\n\r\n  indented\ttab\n\n", "日本語 中文 한국어 🎉", "hôm nay 12/3 lúc 7:30 — ok?"]


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=300)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="file .yui đã export")
    ap.add_argument("--yui", default="./yui")
    ap.add_argument("--ckpt", default="auto")
    ap.add_argument("--skip_logits", action="store_true", help="chỉ kiểm tra tokenizer (không cần torch)")
    a = ap.parse_args()
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.out, "tokenizer.json"))
    texts = list(SAMPLES)
    extra = os.path.join(a.out, "sft_data", "1.txt")
    if os.path.exists(extra):
        texts += [l for l in open(extra, encoding="utf-8").read().split("\n") if l.strip()][:200]
    bad = 0
    for t in texts:
        got = [int(x) for x in run([a.yui, "--model", a.model, "--encode", t]).stdout.split()]
        if got != tok.encode(t).ids:
            bad += 1
            if bad <= 3:
                print("  LỆCH tokenizer:", repr(t[:60]))
    print(f"[tokenizer] {len(texts)} chuỗi, lệch {bad}")
    if a.skip_logits:
        return
    import torch
    sys.path.insert(0, os.getcwd())
    from train_lm import build_model
    cfg = json.load(open(os.path.join(a.out, "config.json"), encoding="utf-8"))
    ck = a.ckpt
    if ck == "auto":
        ck = "sft_best.pt" if os.path.exists(os.path.join(a.out, "sft_best.pt")) else "best.pt"
    model = build_model(cfg, "cpu")
    model.load_state_dict(torch.load(os.path.join(a.out, ck), map_location="cpu")["model"])
    model.eval()
    ids = tok.encode("User: Yui ơi, nay trời đẹp ghê á! Hôm nay mình đi dạo nhé\nYui: Dạ anh, đi liền nè").ids[:cfg["L"]]
    tmp = os.path.join(tempfile.gettempdir(), "yui_logits.bin")
    for k in (1, 2):
        for b in model.blocks:
            b.moe.k_cap = k
        with torch.no_grad():
            ref = model(torch.tensor([ids]), None)[0][0].float().numpy()
        for tag, extra in (("nhanh", []), ("exact", ["--exact"])):
            r = run([a.yui, "--model", a.model, "--ids", ",".join(map(str, ids)), "--k", str(k), "--logits", tmp] + extra)
            if r.returncode:
                if extra:   # bản yui cũ chưa có --exact
                    continue
                print(r.stderr)
                sys.exit(1)
            got = np.fromfile(tmp, dtype=np.float32).reshape(len(ids), -1)
            print(f"[logits] {ck} k={k} ({tag}): max|diff| = {np.abs(got - ref).max():.2e}, argmax khớp {100 * (got.argmax(-1) == ref.argmax(-1)).mean():.0f}% ({len(ids)} vị trí)")

if __name__ == "__main__":
    main()
