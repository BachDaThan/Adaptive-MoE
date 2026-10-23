"""
elastic_ctx.py - cửa sổ ngữ cảnh co giãn (sink + sliding window) cho Adaptive-MoE.
Đặt cạnh adaptive_moe.py.   python elastic_ctx.py --selftest
                            python elastic_ctx.py --train --text file.txt

Ý tưởng: token i chỉ nhìn các token j thoả  j < sink  HOẶC  i-j < W.
  - Lúc train: áp mặt nạ này (W ngẫu nhiên theo từng mẫu) -> giống 'random k'.
  - Lúc suy luận: KV cache chỉ giữ sink + (W-1) token gần nhất -> bộ nhớ KV bị chặn ở sink+W,
    và KẾT QUẢ y hệt chạy lại cả chuỗi với cùng mặt nạ (self-test kiểm tra điều này).
RoPE được áp trước khi cache nên vị trí tuyệt đối giữ nguyên sau khi xoá token giữa.
Giao diện: model(ids, past=None) -> (logits, past) ; past = {"kv": [(k,v)...], "n": số token đã thấy}
"""
import argparse, glob, math, os, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from adaptive_moe import AdaptiveMoE, set_k_cap, _confidence


def rope(x, pos):
    half = x.shape[-1] // 2
    inv = 1.0 / (10000 ** (torch.arange(half, device=x.device).float() / half))
    ang = pos[:, None].float() * inv[None]
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    x1, x2 = x[..., :half], x[..., half:]
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], -1)


class Attn(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.W, self.sink = None, 4          # W: None | int | LongTensor[B]

    def mask(self, S, device):
        i = torch.arange(S, device=device)[:, None]
        j = torch.arange(S, device=device)[None, :]
        m = j <= i
        if self.W is None:
            return m
        W = self.W
        if torch.is_tensor(W):
            W = W.to(device).view(-1, 1, 1)
            return (m & (((i - j) < W) | (j < self.sink)))[:, None]   # [B,1,S,S]
        return m & (((i - j) < W) | (j < self.sink))

    def forward(self, x, pos, past=None):
        B, S, D = x.shape
        q, k, v = self.qkv(x).view(B, S, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        q, k = rope(q, pos), rope(k, pos)
        if past is not None:
            assert S == 1, "có past thì chỉ nạp 1 token mỗi lần"
            k = torch.cat([past[0], k], 2)
            v = torch.cat([past[1], v], 2)
            y = F.scaled_dot_product_attention(q, k, v)          # cache đã được cắt đúng -> nhìn hết
        else:
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=self.mask(S, x.device))
        return self.o(y.transpose(1, 2).reshape(B, S, D)), (k, v)


class Block(nn.Module):
    def __init__(self, d, n_heads, n_experts, d_ff, **moe_kw):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = Attn(d, n_heads)
        self.moe = AdaptiveMoE(d, d_ff, n_experts, **moe_kw)

    def forward(self, x, pos, past=None):
        a, pr = self.attn(self.n1(x), pos, past)
        x = x + a
        y, _, _ = self.moe(self.n2(x))
        return x + y, pr


class CtxLM(nn.Module):
    def __init__(self, vocab=256, d=64, n_layers=2, n_heads=4, n_experts=8, d_ff=128, **moe_kw):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.blocks = nn.ModuleList(Block(d, n_heads, n_experts, d_ff, **moe_kw) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids, past=None):
        n = 0 if past is None else past["n"]
        pos = torch.arange(n, n + ids.shape[1], device=ids.device)
        x = self.emb(ids)
        kv = []
        for i, b in enumerate(self.blocks):
            x, pr = b(x, pos, None if past is None else past["kv"][i])
            kv.append(pr)
        return self.head(self.norm(x)), {"kv": kv, "n": n + ids.shape[1]}


def set_window(model, W, sink=4):
    for m in model.modules():
        if isinstance(m, Attn):
            m.W, m.sink = W, sink


def evict(past, W, sink):
    """Giữ sink token đầu + (W-1) token gần nhất, để token mới (tự nó) là token thứ W."""
    keep = W - 1
    out = []
    for k, v in past["kv"]:
        T = k.shape[2]
        if T > sink + keep:
            k = torch.cat([k[:, :, :sink], k[:, :, T - keep:]], 2)
            v = torch.cat([v[:, :, :sink], v[:, :, T - keep:]], 2)
        out.append((k, v))
    return {"kv": out, "n": past["n"]}


@torch.no_grad()
def generate_ctx(model, ids, max_new_tokens=64, window=None, sink=4, tiers=(8,),
                 min_confidence=0.0, confidence="maxprob", eos_id=None):
    """Sinh greedy với: cửa sổ co giãn (window) + cascade k (tiers) + KV cache bị chặn bộ nhớ."""
    was = model.training
    model.eval()
    ks = []
    try:
        set_window(model, window, sink)
        set_k_cap(model, tiers[-1])
        logits, past = model(ids, None)
        for step in range(max_new_tokens):
            nxt = logits[:, -1].float().argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], 1)
            if eos_id is not None and (nxt == eos_id).all():
                break
            if step == max_new_tokens - 1:
                break
            if window is not None:
                past = evict(past, window, sink)
            for k in tiers:
                set_k_cap(model, k)
                lg, pr = model(nxt, past)
                if _confidence(lg[:, -1], confidence).min().item() >= min_confidence:
                    break
            logits, past = lg, pr
            ks.append(k)
    finally:
        set_k_cap(model, None)
        set_window(model, None)
        model.train(was)
    return ids, ks, past


@torch.no_grad()
def _greedy_full(model, ids, n, k, window, sink):
    model.eval()
    set_k_cap(model, k)
    set_window(model, window, sink)
    for _ in range(n):
        lg, _ = model(ids, None)
        ids = torch.cat([ids, lg[:, -1].argmax(-1, keepdim=True)], 1)
    set_k_cap(model, None)
    set_window(model, None)
    return ids


def selftest(device):
    torch.manual_seed(0)
    model = CtxLM(n_experts=8, d_ff=128, min_k=1, max_k=8, top_p=0.9,
                  n_shared=1, compute_coef=0.0).to(device).eval()
    prompt = torch.randint(0, 256, (2, 40), device=device)
    ok = True
    # 1) cache + xoá token == chạy lại cả chuỗi với cùng mặt nạ (sink+window)
    for W, k in ((8, 2), (16, 8), (None, 4)):
        a, _, past = generate_ctx(model, prompt, 30, window=W, sink=2, tiers=(k,))
        b = _greedy_full(model, prompt, 30, k, W, 2)
        good = torch.equal(a, b)
        mem = past["kv"][0][0].shape[2]
        ok &= good
        print(f"[1] W={W} k={k}: ids khớp = {good} | KV giữ {mem} token (tổng {a.shape[1]})  {'OK' if good else 'FAIL'}")
        if W is not None:
            ok &= mem <= 2 + W
    # 2) W theo từng mẫu (tensor) == W scalar chạy riêng từng hàng
    x = torch.randint(0, 256, (3, 32), device=device)
    Ws = torch.tensor([4, 16, 32], device=device)
    set_k_cap(model, 2)
    set_window(model, Ws, 2)
    with torch.no_grad():
        full, _ = model(x, None)
    err = 0.0
    for r in range(3):
        set_window(model, int(Ws[r]), 2)
        with torch.no_grad():
            one, _ = model(x[r:r + 1], None)
        err = max(err, (full[r] - one[0]).abs().max().item())
    set_window(model, None); set_k_cap(model, None)
    good = err < 1e-3
    ok &= good
    print(f"[2] W theo mẫu vs scalar: max lệch {err:.2e}  {'OK' if good else 'FAIL'}")
    # 3) bộ nhớ KV bị chặn khi chuỗi dài
    long = torch.randint(0, 256, (1, 8), device=device)
    _, _, past = generate_ctx(model, long, 200, window=16, sink=4, tiers=(2,))
    mem = past["kv"][0][0].shape[2]
    good = mem <= 4 + 16
    ok &= good
    print(f"[3] sinh 200 token, W=16: KV chỉ {mem} token  {'OK' if good else 'FAIL'}")
    print("TẤT CẢ OK" if ok else "CÓ LỖI - gửi mình log này")
    return ok


# ---------------- thí nghiệm: train W ngẫu nhiên vs train đủ ngữ cảnh ----------------
def load_hf(name, field, mb, config=None):
    from datasets import load_dataset
    ds = load_dataset(name, config, split="train", streaming=True)
    buf, n = [], 0
    for ex in ds:
        t = ex[field]
        buf.append(t)
        n += len(t.encode("utf-8"))
        if n >= mb * 1_000_000:
            break
    return "\n".join(buf).encode("utf-8")


def load_bytes(path, hf=None, field="text", mb=8, config=None):
    if hf:
        data = load_hf(hf, field, mb, config)
    elif path and os.path.exists(path):
        data = open(path, "rb").read()
    else:
        print("!! Không có --text, dùng file .py/.md trong thư mục (quá nhỏ, chỉ để chạy thử)")
        data = b"".join(open(f, "rb").read() for f in glob.glob("*.py") + glob.glob("*.md"))
    t = torch.tensor(list(data), dtype=torch.long)
    cut = int(len(t) * 0.95)
    print(f"{len(t):,} byte | train {cut:,} | val {len(t) - cut:,}")
    return t[:cut], t[cut:]


def batch(data, B, L, device, g):
    ix = torch.randint(0, len(data) - L - 1, (B,), generator=g)
    x = torch.stack([data[i:i + L] for i in ix])
    y = torch.stack([data[i + 1:i + L + 1] for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def eval_w(model, val, W, L, device, sink, n=40, k=2):
    g = torch.Generator().manual_seed(123)
    model.eval(); set_window(model, W, sink); set_k_cap(model, k)
    tot = 0.0
    for _ in range(n):
        x, y = batch(val, 16, L, device, g)
        lg, _ = model(x, None)
        tot += F.cross_entropy(lg.float().view(-1, lg.shape[-1]), y.view(-1)).item()
    set_window(model, None); set_k_cap(model, None)
    return tot / n


def train_one(name, rand_w, train, val, args, device):
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    model = CtxLM(d=args.d, n_layers=args.layers, n_experts=8, d_ff=2 * args.d,
                  min_k=1, max_k=8, top_p=args.top_p, n_shared=1, compute_coef=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    choices = torch.tensor([w for w in (8, 16, 32, 64, 128, args.L) if w <= args.L], device=device)
    print(f"--- {name}: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M tham số ---")
    t0 = time.time()
    for step in range(args.steps + 1):
        model.train(); set_k_cap(model, 2)
        set_window(model, choices[torch.randint(len(choices), (args.B,), device=device)] if rand_w else None, args.sink)
        x, y = batch(train, args.B, args.L, device, g)
        lg, _ = model(x, None)
        loss = F.cross_entropy(lg.float().view(-1, lg.shape[-1]), y.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            print(f"step {step} loss={loss.item():.3f} {time.time() - t0:.0f}s")
    set_window(model, None); set_k_cap(model, None)
    return model


def run_train(args, device):
    train, val = load_bytes(args.text, args.hf, args.field, args.mb, args.config)
    res = {}
    for name, rw in (("full-context", False), ("random-window", True)):
        m = train_one(name, rw, train, val, args, device)
        res[name] = {W: eval_w(m, val, None if W == args.L else W, args.L, device, args.sink)
                     for W in (8, 16, 32, 64, 128, args.L) if W <= args.L}
    print("\n===== val loss khi ÉP cửa sổ lúc suy luận (KV giữ sink+W token) =====")
    Ws = list(res["full-context"].keys())
    print("W".ljust(10) + "".join(str(w).rjust(9) for w in Ws))
    for name in res:
        print(name.ljust(10)[:10] + "".join(f"{res[name][w]:9.4f}" for w in Ws))
    print("(W = %d là ngữ cảnh đầy đủ)" % args.L)


def decode_speed(model, val, W, k, device, sink, n_new=128, P=128):
    prompt = val[:P][None].to(device)
    generate_ctx(model, prompt, 8, window=W, sink=sink, tiers=(k,))   # khởi động
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    generate_ctx(model, prompt, n_new, window=W, sink=sink, tiers=(k,))
    if device == "cuda": torch.cuda.synchronize()
    return n_new / (time.time() - t0)


def run_grid(args, device):
    """Train MỘT model co giãn cả k lẫn W, rồi đo lưới (k x W): val loss / tok/s (có cache)."""
    train, val = load_bytes(args.text, args.hf, args.field, args.mb, args.config)
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    model = CtxLM(d=args.d, n_layers=args.layers, n_experts=8, d_ff=2 * args.d,
                  min_k=1, max_k=8, top_p=args.top_p, n_shared=1, compute_coef=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    ks = (1, 2, 4, 8)
    Ws = [w for w in (8, 16, 32, 64, 128, args.L) if w <= args.L]
    choices = torch.tensor(Ws, device=device)
    t0 = time.time()
    for step in range(args.steps + 1):
        model.train()
        set_k_cap(model, ks[torch.randint(len(ks), (1,)).item()])
        set_window(model, choices[torch.randint(len(Ws), (args.B,), device=device)], args.sink)
        x, y = batch(train, args.B, args.L, device, g)
        lg, _ = model(x, None)
        loss = F.cross_entropy(lg.float().view(-1, lg.shape[-1]), y.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            print(f"step {step} loss={loss.item():.3f} {time.time() - t0:.0f}s")
    set_window(model, None); set_k_cap(model, None)
    print("\n===== val loss / tok/s (decode có KV cache), hàng = k, cột = W =====")
    print("k\\W".ljust(5) + "".join(str(w).rjust(15) for w in Ws))
    print("KV".ljust(5) + "".join(str(args.sink + w if w < args.L else args.L).rjust(15) for w in Ws) + "   (token KV giữ)")
    for k in ks:
        row = str(k).ljust(5)
        for w in Ws:
            W = None if w == args.L else w
            l = eval_w(model, val, W, args.L, device, args.sink, k=k)
            v = decode_speed(model, val, W, k, device, args.sink)
            row += f"{l:.4f}/{v:5.0f}".rjust(15)
        print(row)
    print("(tok/s ở model nhỏ bị chi phối bởi overhead Python, chỉ mang tính tham khảo)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--train", action="store_true")
    ap.add_argument("--grid", action="store_true")
    ap.add_argument("--text", default=None)
    ap.add_argument("--hf", default=None, help="tên dataset HF, ví dụ lấy từ lm_experiment.py")
    ap.add_argument("--config", default=None)
    ap.add_argument("--field", default="text")
    ap.add_argument("--mb", type=int, default=8, help="số MB text lấy từ HF")
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--sink", type=int, default=4)
    ap.add_argument("--top_p", type=float, default=0.9, help="đặt 0.999 để k_cap thật sự là giới hạn chặn")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    if a.selftest:
        if not selftest(dev):
            raise SystemExit(1)
    if a.train:
        run_train(a, dev)
    if a.grid:
        run_grid(a, dev)
