"""
export_yui.py - đóng gói model Adaptive-MoE (checkpoint .pt + config.json + tokenizer.json) thành MỘT file .yui cho yui.cpp.

    python export_yui.py --out thu_muc_model --ckpt auto --dst model.yui            # float32 (~200 MB)
    python export_yui.py --out thu_muc_model --ckpt auto --dst model_q8.yui --quant int8   # int8 theo hàng (~50 MB)
    python export_yui.py --out thu_muc_model --ckpt auto --dst model_q4.yui --quant q4     # expert 4-bit theo khối 32, phần còn lại int8 (định dạng v2)
    python export_yui.py --out thu_muc_model --ckpt auto --dst model_q4all.yui --quant q4all   # TẤT CẢ ma trận 4-bit (nhỏ nhất, chất lượng giảm nhiều nhất)

File .yui gồm: cấu hình, tokenizer (vocab + merges + bảng chữ cái/số Unicode cho bước tách từ) và toàn bộ trọng số.
Router (gate, strength) và LayerNorm luôn giữ float32 để việc chọn expert giống hệt PyTorch; int8 chỉ áp cho ma trận lớn
(embedding, qkv, o, expert, head), mỗi hàng một hệ số scale.
Định dạng (little-endian): xem hàm write_yui; yui.cpp đọc đúng thứ tự này.

v1 (f32, int8): như cũ, engine nào cũng đọc được. v2 (q4, q4all): sau tiêu đề, MỖI ma trận có 1 byte kiểu: 0 = float32, 1 = int8 theo hàng (hệ số/hàng rồi int8),
2 = q4: mỗi hàng chia khối 32 trọng số (hàng ngắn hơn thì đệm 0), mỗi khối 1 hệ số fp16 + 16 byte (byte j = trọng số j ở 4 bit thấp, j+16 ở 4 bit cao);
giá trị = (nibble - 8) * hệ số. Ghi: toàn bộ hệ số fp16 của ma trận, rồi toàn bộ byte 4-bit. Router/LayerNorm luôn float32.
q4 chỉ lượng tử expert (chiếm ~85% tham số) và giữ attention + embedding + head ở int8: nhỏ hơn int8 khoảng 1/3, chất lượng gần như int8.
"""
import re
import argparse, json, os, struct, unicodedata
import numpy as np

MAGIC = b"YUI1"


def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, [chr(c) for c in cs]))


def unicode_ranges():
    """Các khoảng code point thuộc \\p{L} (cls 1) và \\p{N} (cls 2), để C++ tách từ giống regex GPT-2 của tokenizers."""
    out, cur = [], None
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            cls = 0
        else:
            cat = unicodedata.category(chr(cp))
            cls = 1 if cat[0] == "L" else (2 if cat[0] == "N" else 0)
        if cur and cur[2] == cls and cur[1] == cp - 1:
            cur[1] = cp
        else:
            if cur and cur[2]:
                out.append(tuple(cur))
            cur = [cp, cp, cls]
    if cur and cur[2]:
        out.append(tuple(cur))
    return out


def load_tokenizer(path):
    j = json.load(open(path, encoding="utf-8"))
    m = j["model"]
    assert m["type"] == "BPE", "chỉ hỗ trợ tokenizer BPE"
    vocab = m["vocab"]
    n = max(vocab.values()) + 1
    inv = [None] * n
    for t, i in vocab.items():
        inv[i] = t
    assert all(t is not None for t in inv), "vocab có lỗ hổng id"
    merges = []
    for it in m["merges"]:
        a, b = (it.split(" ", 1) if isinstance(it, str) else it)
        merges.append((a, b))
    eos = vocab.get("<eos>", -1)
    pre = json.dumps(j.get("pre_tokenizer"))
    assert "ByteLevel" in pre, "chỉ hỗ trợ pre_tokenizer ByteLevel"
    assert '"add_prefix_space": false' in pre.lower() or '"add_prefix_space":false' in pre.lower(), "add_prefix_space phải là false"
    return inv, merges, eos


def pack_str(s):
    b = s.encode("utf-8")
    assert len(b) < 65536
    return struct.pack("<H", len(b)) + b


def quantize_rows(w):
    w = np.ascontiguousarray(w, dtype=np.float32)
    sc = np.abs(w).max(axis=1) / 127.0
    sc = np.where(sc == 0, 1.0, sc).astype(np.float32)
    q = np.clip(np.rint(w / sc[:, None]), -127, 127).astype(np.int8)
    return sc, q


def q4_quantize(w):
    """w [rows, cols] float -> (scales fp16 [rows, nb], packed uint8 [rows, nb, 16]); nb = ceil(cols/32). Kiểu Q4_0: hệ số = (giá trị lớn nhất theo trị tuyệt đối, có dấu) / -8."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    rows, cols = w.shape
    nb = (cols + 31) // 32
    pad = np.zeros((rows, nb * 32), np.float32)
    pad[:, :cols] = w
    blk = pad.reshape(rows, nb, 32)
    idx = np.abs(blk).argmax(-1)
    m = np.take_along_axis(blk, idx[..., None], -1)[..., 0]
    d16 = (m / -8.0).astype(np.float16)
    d = d16.astype(np.float32)
    inv = np.where(d != 0, 1.0 / np.where(d != 0, d, 1.0), 0.0).astype(np.float32)
    q = np.clip(np.floor(blk * inv[..., None] + 8.5), 0, 15).astype(np.uint8)
    return d16, (q[..., :16] | (q[..., 16:] << 4)).astype(np.uint8)


def q4_dequantize(d16, packed, cols):
    q = np.concatenate([packed & 15, packed >> 4], -1).astype(np.float32)
    w = (q - 8.0) * d16.astype(np.float32)[..., None]
    return w.reshape(w.shape[0], -1)[:, :cols]


def write_yui(sd, cfg, tokenizer_json, dst, quant="f32", stats=None):
    """sd: {tên tham số PyTorch: np.ndarray float32}; cfg: config.json của train_lm.py (+ tuỳ chọn heads, min_k, max_k, min_strength)."""
    V, D, NL, E = cfg["vocab"], cfg["d"], cfg["layers"], cfg["experts"]
    H = cfg.get("heads", 4)           # CtxLM mặc định 4 head
    F = 2 * D
    L = cfg["L"]
    min_k, max_k = cfg.get("min_k", 1), cfg.get("max_k", 8)
    ms, top_p = cfg.get("min_strength", 0.3), float(cfg["top_p"])
    n_shared = 1
    assert quant in ("f32", "int8", "q4", "q4all"), quant
    q8 = quant == "int8"
    ver = 2 if quant.startswith("q4") else 1
    qfield = {"f32": 0, "int8": 1, "q4": 2, "q4all": 2}[quant]

    def kind(name):   # 0 f32, 1 int8, 2 q4
        if quant == "f32":
            return 0
        if quant == "int8":
            return 1
        if quant == "q4all":
            return 2
        return 2 if (".experts." in name or ".shared." in name) else 1
    inv, merges, eos = load_tokenizer(tokenizer_json)
    assert len(inv) == V, f"vocab tokenizer {len(inv)} != config {V}"
    b2u = bytes_to_unicode()
    assert len({b2u[b] for b in range(256)} - set(inv)) == 0, "tokenizer thiếu ký tự byte-level"

    with open(dst, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<12I", ver, qfield, V, D, NL, H, E, F, n_shared, L, min_k, max_k))
        f.write(struct.pack("<2f", top_p, ms))
        # tokenizer
        f.write(struct.pack("<I", len(inv)))
        for t in inv:
            f.write(pack_str(t))
        f.write(struct.pack("<I", len(merges)))
        for a, b in merges:
            f.write(pack_str(a) + pack_str(b))
        f.write(struct.pack("<i", eos))
        rg = unicode_ranges()
        f.write(struct.pack("<I", len(rg)))
        for a, b, c in rg:
            f.write(struct.pack("<3I", a, b, c))

        def put_f32(name, shape):
            w = np.asarray(sd[name], dtype=np.float32)
            assert tuple(w.shape) == tuple(shape), f"{name}: {w.shape} != {shape}"
            f.write(np.ascontiguousarray(w).tobytes())

        def put_mat(name, rows, cols):
            w = np.asarray(sd[name], dtype=np.float32)
            assert tuple(w.shape) == (rows, cols), f"{name}: {w.shape} != {(rows, cols)}"
            k = kind(name)
            if ver == 2:
                f.write(struct.pack("<B", k))
            if k == 1:
                sc, q = quantize_rows(w)
                f.write(sc.tobytes())
                f.write(q.tobytes())
            elif k == 2:
                d16, packed = q4_quantize(w)
                f.write(d16.tobytes())
                f.write(packed.tobytes())
                if stats is not None:   # sai số tương đối của lượng tử hoá (để biết 4-bit làm mất bao nhiêu)
                    e = stats.setdefault(name.split(".")[0] if not name.startswith("blocks") else name.split(".", 2)[2].split(".")[0], [0.0, 0.0])
                    e[0] += float(((w - q4_dequantize(d16, packed, cols)) ** 2).sum())
                    e[1] += float((w ** 2).sum())
            else:
                f.write(np.ascontiguousarray(w).tobytes())

        put_mat("emb.weight", V, D)
        for l in range(NL):
            p = f"blocks.{l}."
            put_f32(p + "n1.weight", (D,)); put_f32(p + "n1.bias", (D,))
            put_mat(p + "attn.qkv.weight", 3 * D, D)
            put_mat(p + "attn.o.weight", D, D)
            put_f32(p + "n2.weight", (D,)); put_f32(p + "n2.bias", (D,))
            put_f32(p + "moe.router.gate.weight", (E, D))
            put_f32(p + "moe.router.strength_head.weight", (1, D))
            put_f32(p + "moe.router.strength_head.bias", (1,))
            for e in range(E):
                q = f"{p}moe.experts.{e}."
                put_mat(q + "w_gate.weight", F, D); put_mat(q + "w_up.weight", F, D); put_mat(q + "w_down.weight", D, F)
            for s in range(n_shared):
                q = f"{p}moe.shared.{s}."
                put_mat(q + "w_gate.weight", F, D); put_mat(q + "w_up.weight", F, D); put_mat(q + "w_down.weight", D, F)
        put_f32("norm.weight", (D,)); put_f32("norm.bias", (D,))
        put_mat("head.weight", V, D)
    return os.path.getsize(dst)


class LazySD:
    """Đọc từng tensor khi cần (float32) thay vì đổi cả state_dict sang numpy một lượt: model lớn (1B-10B) không tràn RAM khi xuất."""

    def __init__(self, m):
        self.m = m

    def __getitem__(self, k):
        return self.m[k].detach().float().cpu().numpy()

    def __contains__(self, k):
        return k in self.m

    def keys(self):
        return self.m.keys()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="thư mục có config.json, tokenizer.json và checkpoint")
    ap.add_argument("--ckpt", default="auto", help="auto = sft_best.pt nếu có, không thì best.pt")
    ap.add_argument("--dst", default=None, help="file .yui đầu ra; bỏ trống = <tên model trong config.json>_<quant>.yui")
    ap.add_argument("--quant", choices=["f32", "int8", "q4", "q4all"], default="f32",
                    help="f32 | int8 | q4 (expert 4-bit, còn lại int8) | q4all (mọi ma trận 4-bit; nhỏ nhất nhưng chất lượng giảm nhiều nhất)")
    a = ap.parse_args()
    import torch
    cfg = json.load(open(os.path.join(a.out, "config.json"), encoding="utf-8"))
    ck = a.ckpt
    if ck == "auto":
        ck = "sft_best.pt" if os.path.exists(os.path.join(a.out, "sft_best.pt")) else "best.pt"
    path = os.path.join(a.out, ck)
    try:   # mmap: không nạp cả checkpoint (kể cả trạng thái optimizer) vào RAM
        obj = torch.load(path, map_location="cpu", mmap=True)
    except (TypeError, RuntimeError, ValueError):
        obj = torch.load(path, map_location="cpu")
    sd = LazySD(obj["model"])
    dst = a.dst or f"{cfg.get('name') or 'model'}_{a.quant}.yui"
    stats = {}
    n = write_yui(sd, cfg, os.path.join(a.out, "tokenizer.json"), dst, a.quant, stats)
    print(f"đã ghi {dst} ({n / 1e6:.1f} MB) từ {ck}, quant={a.quant}" + (f", tên model: {cfg['name']}" if cfg.get("name") else ""))
    try:   # README riêng cho model này, nằm cạnh file .yui: <tên>.README.md (gom các bản q8/q4/... cùng tên)
        import model_card
        folder = os.path.dirname(os.path.abspath(dst))
        base = re.sub(r"_(f32|int8|q8|q4all|q4)\.yui$", "", os.path.basename(dst))
        if os.path.basename(dst) != base:
            model_card.write(folder, model_card.yui_card(folder, base, cfg, a.out, ck), name=f"{base}.README.md")
            print(f"đã ghi {base}.README.md cạnh file .yui")
    except Exception as e:  # noqa: BLE001
        print(f"bỏ qua README ({str(e)[:80]})")
    if stats:
        tot_e = sum(v[0] for v in stats.values()); tot_n = sum(v[1] for v in stats.values())
        print(f"sai số lượng tử hoá 4-bit (căn bậc hai tỉ lệ năng lượng): tổng {100 * (tot_e / max(tot_n, 1e-12)) ** 0.5:.2f}%  "
              + ", ".join(f"{k} {100 * (v[0] / max(v[1], 1e-12)) ** 0.5:.2f}%" for k, v in sorted(stats.items())[:6])
              + "\nKiểm chất lượng thật: python verify_cpp.py --out ... --model <file q4>  (so với PyTorch) rồi thử chat vài câu trước khi dùng.")


if __name__ == "__main__":
    main()
