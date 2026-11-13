"""
verify_cpp.py - đối chiếu yui (C++) với PyTorch trên model THẬT của bạn. Chạy trong thư mục repo Adaptive-MoE. Hỗ trợ .yui v1/v2/v3.

    make
    python export_yui.py --out thu_muc_model --dst model.yui [--quant q4 --fmt 3 ...]
    python verify_cpp.py --out thu_muc_model --model model.yui --yui ./yui
    python verify_cpp.py --out thu_muc_model --model model_v3.yui --compare model_v2.yui --skip_logits   # v3 và v2 cùng checkpoint/quant: logits phải GIỐNG HỆT

0) File: đọc và kiểm toàn bộ bố cục bằng yui_format.py (v3: cả META, mục lục, chân file), đối chiếu cấu hình với config.json.
1) Tokenizer: mã hoá nhiều chuỗi bằng thư viện tokenizers và bằng ./yui --encode, đếm số chuỗi lệch (mong đợi 0).
2) Logits: chạy CtxLM của repo (float32, eval, k_cap = 1 và 2) trên một chuỗi id, so với ./yui --ids ... --logits.
   Mong đợi file float32: max|diff| < 1e-3 và argmax khớp 100%. File lượng tử (int8/q4/t2): argmax khớp cao (thường > 90%), diff lớn hơn là bình thường.
   Với file lượng tử script chạy 2 lần: chế độ NHANH (kích hoạt lượng tử int8 theo khối 32, mặc định của yui) và chế độ --exact
   (kích hoạt float như PyTorch). --exact chỉ sai khác do trọng số lượng tử; nếu chế độ nhanh lệch hẳn so với --exact thì báo lại.
3) --compare FILE.yui (không cần torch): so logits của engine giữa 2 file. Cùng quant + cùng trọng số (vd v2 và v3 xuất từ một checkpoint)
   thì phải giống HỆT (max|diff| = 0) và cùng tokenizer_sha256.
Kết thúc in ĐẠT/KHÔNG ĐẠT; mã thoát 1 nếu có lỗi cứng (cảnh báo mềm "CẢNH BÁO" không làm hỏng kết quả).
"""
import argparse, json, os, subprocess, sys, tempfile
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import yui_format  # noqa: E402

SAMPLES = ["xin chào", "Hà Nội là thủ đô của Việt Nam.", "  hai  dấu   cách  ", "User: Yui ơi, nay trời đẹp ghê á!\nYui: Đẹp thía mà hong có ai dắt Yui đi dạo nè 🥺",
           "Hello world, it's 2024! Don't stop... 123,456.78", "line1\nline2\r\n\r\n  indented\ttab\n\n", "日本語 中文 한국어 🎉", "hôm nay 12/3 lúc 7:30 — ok?"]
PROMPT = "User: Yui ơi, nay trời đẹp ghê á! Hôm nay mình đi dạo nhé\nYui: Dạ anh, đi liền nè"
# cấu hình config.json (khoá train_lm.py) <-> header .yui
CFG_KEYS = (("vocab", "V"), ("d", "D"), ("layers", "layers"), ("experts", "E"), ("L", "L"))
F32_MAXDIFF = 1e-3
QUANT_ARGMAX_WARN = 90.0


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, stdin=subprocess.DEVNULL, timeout=300)


def engine_logits(yui, model, ids, k, exact):
    """Logits [len(ids), V] float32 từ ./yui. exact=True mà bản yui cũ chưa có --exact thì trả None. Lỗi khác thì dừng."""
    fd, tmp = tempfile.mkstemp(suffix=".bin", prefix="yui_logits_")
    os.close(fd)
    try:
        r = run([yui, "--model", model, "--ids", ",".join(map(str, ids)), "--k", str(k), "--logits", tmp] + (["--exact"] if exact else []))
        if r.returncode:
            if exact:
                return None
            print(r.stderr)
            print("KHÔNG ĐẠT: yui không chạy được với file này")
            sys.exit(1)
        return np.fromfile(tmp, dtype=np.float32).reshape(len(ids), -1)
    finally:
        if os.path.exists(tmp):
            os.remove(tmp)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--model", required=True, help="file .yui đã export (v1/v2/v3)")
    ap.add_argument("--yui", default="./yui")
    ap.add_argument("--ckpt", default="auto")
    ap.add_argument("--compare", default=None, help="file .yui thứ hai để so logits engine với --model (không cần torch)")
    ap.add_argument("--skip_logits", action="store_true", help="bỏ so với PyTorch (không cần torch); vẫn chạy kiểm file, tokenizer và --compare")
    a = ap.parse_args()
    fails, warns = [], []

    # 0) file: bố cục + META + cấu hình
    try:
        info = yui_format.read_yui(a.model, with_tensors=False)
    except yui_format.YuiFormatError as e:
        print(f"[file] {a.model}: LỖI - {e}")
        print("KHÔNG ĐẠT: file .yui hỏng")
        sys.exit(1)
    print("[file]", yui_format.summary(info))
    cfgj = {}
    cpath = os.path.join(a.out, "config.json")
    if os.path.exists(cpath):
        cfgj = json.load(open(cpath, encoding="utf-8"))
        for kj, kh in CFG_KEYS:
            if kj in cfgj and cfgj[kj] != info["cfg"][kh]:
                fails.append(f"config.json {kj}={cfgj[kj]} khác header .yui {kh}={info['cfg'][kh]}")
    else:
        warns.append(f"không thấy {cpath}, bỏ đối chiếu cấu hình")

    # 1) tokenizer
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
    if bad:
        fails.append(f"tokenizer lệch {bad}/{len(texts)} chuỗi")
    L = info["cfg"]["L"]
    ids = tok.encode(PROMPT).ids[:L]

    # 3) --compare: hai file .yui, engine phải cho cùng logits khi cùng quant + trọng số
    if a.compare:
        try:
            info2 = yui_format.read_yui(a.compare, with_tensors=False)
        except yui_format.YuiFormatError as e:
            print(f"[compare] {a.compare}: LỖI - {e}")
            print("KHÔNG ĐẠT: file --compare hỏng")
            sys.exit(1)
        print("[compare]", yui_format.summary(info2))
        if info2["tokenizer_sha256"] != info["tokenizer_sha256"]:
            fails.append("hai file có tokenizer khác nhau (sha256 lệch)")
        same_arch = info["cfg"] == info2["cfg"] and info["quant"] == info2["quant"] and info["params"] == info2["params"]
        if not same_arch:
            warns.append("hai file khác cấu hình/quant nên KHÔNG kỳ vọng logits giống hệt (chỉ in độ lệch)")
        for k in (1, 2):
            for tag, ex in (("nhanh", False), ("exact", True)):
                x = engine_logits(a.yui, a.model, ids, k, ex)
                y = engine_logits(a.yui, a.compare, ids, k, ex)
                if x is None or y is None:
                    continue
                if x.shape != y.shape:
                    fails.append(f"compare k={k} ({tag}): kích thước logits khác nhau {x.shape} vs {y.shape}")
                    continue
                d = float(np.abs(x - y).max())
                agree = 100 * float((x.argmax(-1) == y.argmax(-1)).mean())
                print(f"[compare] k={k} ({tag}): max|diff| = {d:.2e}, argmax khớp {agree:.0f}% ({len(ids)} vị trí)")
                if same_arch and d != 0.0:
                    fails.append(f"compare k={k} ({tag}): cùng quant/trọng số mà logits lệch {d:.2e} (mong đợi giống hệt)")

    # 4) logits so với PyTorch
    if not a.skip_logits:
        import torch
        sys.path.insert(0, os.getcwd())
        from train_lm import build_model
        cfg = cfgj or json.load(open(cpath, encoding="utf-8"))
        ck = a.ckpt
        if ck == "auto":
            ck = "sft_best.pt" if os.path.exists(os.path.join(a.out, "sft_best.pt")) else "best.pt"
        model = build_model(cfg, "cpu")
        model.load_state_dict(torch.load(os.path.join(a.out, ck), map_location="cpu")["model"])
        model.eval()
        for k in (1, 2):
            for b in model.blocks:
                b.moe.k_cap = k
            with torch.no_grad():
                ref = model(torch.tensor([ids]), None)[0][0].float().numpy()
            got_by = {}
            for tag, ex in (("nhanh", False), ("exact", True)):
                got = engine_logits(a.yui, a.model, ids, k, ex)
                if got is None:   # bản yui cũ chưa có --exact
                    continue
                got_by[tag] = got
                d = float(np.abs(got - ref).max())
                agree = 100 * float((got.argmax(-1) == ref.argmax(-1)).mean())
                print(f"[logits] {ck} k={k} ({tag}): max|diff| = {d:.2e}, argmax khớp {agree:.0f}% ({len(ids)} vị trí)")
                if info["quant"] == 0:   # float32: phải trùng PyTorch
                    if d >= F32_MAXDIFF or agree < 100:
                        fails.append(f"logits f32 k={k} ({tag}): max|diff| {d:.2e} (cần < {F32_MAXDIFF:g}), argmax {agree:.0f}% (cần 100%)")
                elif agree < QUANT_ARGMAX_WARN:
                    warns.append(f"logits {info['quant_name']} k={k} ({tag}): argmax khớp {agree:.0f}% < {QUANT_ARGMAX_WARN:.0f}%")
            if "nhanh" in got_by and "exact" in got_by:
                fe = 100 * float((got_by["nhanh"].argmax(-1) == got_by["exact"].argmax(-1)).mean())
                if fe < QUANT_ARGMAX_WARN:
                    warns.append(f"k={k}: chế độ nhanh lệch hẳn so với --exact (argmax khớp {fe:.0f}%), hãy báo lại")

    for w in warns:
        print("CẢNH BÁO:", w)
    for f in fails:
        print("LỖI:", f)
    print("KẾT QUẢ: " + ("KHÔNG ĐẠT" if fails else "ĐẠT"))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
