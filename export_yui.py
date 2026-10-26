"""
export_yui.py - đóng gói model Adaptive-MoE (checkpoint .pt + config.json + tokenizer.json) thành MỘT file .yui cho yui.cpp.

    python export_yui.py --out thu_muc_model --ckpt auto --dst model.yui            # float32 (~200 MB)
    python export_yui.py --out thu_muc_model --ckpt auto --dst model_q8.yui --quant int8   # int8 theo hàng (~50 MB)

File .yui gồm: cấu hình, tokenizer (vocab + merges + bảng chữ cái/số Unicode cho bước tách từ) và toàn bộ trọng số.
Router (gate, strength) và LayerNorm luôn giữ float32 để việc chọn expert giống hệt PyTorch; int8 chỉ áp cho ma trận lớn
(embedding, qkv, o, expert, head), mỗi hàng một hệ số scale.
Định dạng (little-endian): xem hàm write_yui; yui.cpp đọc đúng thứ tự này.
"""
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


def write_yui(sd, cfg, tokenizer_json, dst, quant="f32"):
    """sd: {tên tham số PyTorch: np.ndarray float32}; cfg: config.json của train_lm.py (+ tuỳ chọn heads, min_k, max_k, min_strength)."""
    V, D, NL, E = cfg["vocab"], cfg["d"], cfg["layers"], cfg["experts"]
    H = cfg.get("heads", 4)           # CtxLM mặc định 4 head
    F = 2 * D
    L = cfg["L"]
    min_k, max_k = cfg.get("min_k", 1), cfg.get("max_k", 8)
    ms, top_p = cfg.get("min_strength", 0.3), float(cfg["top_p"])
    n_shared = 1
    q8 = quant == "int8"
    inv, merges, eos = load_tokenizer(tokenizer_json)
    assert len(inv) == V, f"vocab tokenizer {len(inv)} != config {V}"
    b2u = bytes_to_unicode()
    assert len({b2u[b] for b in range(256)} - set(inv)) == 0, "tokenizer thiếu ký tự byte-level"

    with open(dst, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<12I", 1, 1 if q8 else 0, V, D, NL, H, E, F, n_shared, L, min_k, max_k))
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
            if q8:
                sc, q = quantize_rows(w)
                f.write(sc.tobytes())
                f.write(q.tobytes())
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="thư mục có config.json, tokenizer.json và checkpoint")
    ap.add_argument("--ckpt", default="auto", help="auto = sft_best.pt nếu có, không thì best.pt")
    ap.add_argument("--dst", required=True)
    ap.add_argument("--quant", choices=["f32", "int8"], default="f32")
    a = ap.parse_args()
    import torch
    cfg = json.load(open(os.path.join(a.out, "config.json"), encoding="utf-8"))
    ck = a.ckpt
    if ck == "auto":
        ck = "sft_best.pt" if os.path.exists(os.path.join(a.out, "sft_best.pt")) else "best.pt"
    obj = torch.load(os.path.join(a.out, ck), map_location="cpu")
    sd = {k: v.float().numpy() for k, v in obj["model"].items()}
    n = write_yui(sd, cfg, os.path.join(a.out, "tokenizer.json"), a.dst, a.quant)
    print(f"đã ghi {a.dst} ({n / 1e6:.1f} MB) từ {ck}, quant={a.quant}")


if __name__ == "__main__":
    main()
