"""
elastic_kv.py - KV cache cho adaptive_generate + self-test.
Copy cạnh adaptive_moe.py. Chạy: python elastic_kv.py --selftest

Giao diện model cần có: model(ids, past=None) -> (logits [B,S,V], past)
TinyCachedLM bên dưới là bản mẫu để bạn ghép vào model thật.

Lưu ý thiết kế: token cũ trong cache được tính ở k lúc nó được xử lý, nên cascade + cache
KHÔNG cho kết quả y hệt chạy lại cả chuỗi ở một k cố định. Với k cố định thì phải khớp (self-test kiểm tra).
"""
import argparse, time
import torch
import torch.nn as nn
import torch.nn.functional as F
from adaptive_moe import AdaptiveMoE, set_k_cap, _confidence


class CachedAttention(nn.Module):
    def __init__(self, d, n_heads):
        super().__init__()
        self.h = n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)

    def forward(self, x, past=None):
        B, S, D = x.shape
        q, k, v = self.qkv(x).view(B, S, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        if past is not None:
            k = torch.cat([past[0], k], 2)
            v = torch.cat([past[1], v], 2)
        T = k.shape[2]
        mask = torch.ones(S, T, dtype=torch.bool, device=x.device).tril(T - S)
        y = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        return self.o(y.transpose(1, 2).reshape(B, S, D)), (k, v)


class Block(nn.Module):
    def __init__(self, d, n_heads, n_experts, d_ff, **moe_kw):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = CachedAttention(d, n_heads)
        self.moe = AdaptiveMoE(d, d_ff, n_experts, **moe_kw)

    def forward(self, x, past=None):
        a, present = self.attn(self.n1(x), past)
        x = x + a
        y, _, _ = self.moe(self.n2(x))
        return x + y, present


class TinyCachedLM(nn.Module):
    def __init__(self, vocab=256, d=64, n_layers=2, n_heads=4, n_experts=8, d_ff=128,
                 max_len=512, **moe_kw):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(max_len, d)
        self.blocks = nn.ModuleList(Block(d, n_heads, n_experts, d_ff, **moe_kw) for _ in range(n_layers))
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)

    def forward(self, ids, past=None):
        off = 0 if past is None else past[0][0].shape[2]
        x = self.emb(ids) + self.pos(torch.arange(off, off + ids.shape[1], device=ids.device))
        presents = []
        for i, b in enumerate(self.blocks):
            x, pr = b(x, None if past is None else past[i])
            presents.append(pr)
        return self.head(self.norm(x)), presents


@torch.no_grad()
def adaptive_generate_kv(model, input_ids, max_new_tokens=32, tiers=(2, 4, 8),
                         min_confidence=0.6, confidence="maxprob", eos_id=None):
    """Như adaptive_generate nhưng có KV cache. Prefill ở tầng cao nhất, mỗi bước chỉ chạy 1 token.
    Bước kém tự tin: chạy lại ĐÚNG token đó ở tầng kế tiếp, từ cache trước bước (bỏ cache thử sai)."""
    was_training = model.training
    model.eval()
    ids, ks = input_ids, []
    try:
        set_k_cap(model, tiers[-1])
        logits, past = model(ids, None)
        for step in range(max_new_tokens):
            nxt = logits[:, -1].float().argmax(-1, keepdim=True)
            ids = torch.cat([ids, nxt], 1)
            if eos_id is not None and (nxt == eos_id).all():
                break
            if step == max_new_tokens - 1:
                break
            for k in tiers:
                set_k_cap(model, k)
                lg, pr = model(nxt, past)
                if _confidence(lg[:, -1], confidence).min().item() >= min_confidence:
                    break
            logits, past = lg, pr
            ks.append(k)
    finally:
        set_k_cap(model, None)
        model.train(was_training)
    return ids, ks


@torch.no_grad()
def _greedy_no_cache(model, ids, n, k):
    set_k_cap(model, k)
    for _ in range(n):
        lg, _ = model(ids, None)
        ids = torch.cat([ids, lg[:, -1].argmax(-1, keepdim=True)], 1)
    set_k_cap(model, None)
    return ids


def selftest(device):
    torch.manual_seed(0)
    model = TinyCachedLM(n_experts=8, d_ff=128, min_k=1, max_k=8, top_p=0.9,
                         n_shared=1, compute_coef=0.0).to(device).eval()
    prompt = torch.randint(0, 256, (2, 16), device=device)
    ok = True

    # 1) k cố định: logits từng bước có cache phải khớp chạy lại cả chuỗi
    for k in (1, 2, 4, 8):
        set_k_cap(model, k)
        with torch.no_grad():
            full, _ = model(prompt, None)
            lg, past = model(prompt[:, :8], None)
            outs = [lg]
            for t in range(8, 16):
                lg, past = model(prompt[:, t:t + 1], past)
                outs.append(lg)
            inc = torch.cat(outs, 1)
        err = (full - inc).abs().max().item()
        good = err < 1e-3
        ok &= good
        print(f"[1] k={k}: max|logit full - cached| = {err:.2e}  {'OK' if good else 'FAIL'}")
    set_k_cap(model, None)

    # 2) sinh greedy ở k cố định: cache == không cache
    for k in (2, 8):
        a, _ = adaptive_generate_kv(model, prompt, 24, tiers=(k,), min_confidence=0.0)
        b = _greedy_no_cache(model, prompt, 24, k)
        good = torch.equal(a, b)
        ok &= good
        print(f"[2] generate k={k}: ids khớp = {good}  {'OK' if good else 'FAIL'}")

    # 3) cascade: k trong tiers, đủ số bước
    ids, ks = adaptive_generate_kv(model, prompt, 20, tiers=(1, 2, 4, 8), min_confidence=0.02)
    good = ids.shape[1] == 16 + 20 and len(ks) == 19 and set(ks) <= {1, 2, 4, 8}
    ok &= good
    print(f"[3] cascade: len={ids.shape[1]} ks dùng={sorted(set(ks))}  {'OK' if good else 'FAIL'}")

    # 4) tốc độ (chỉ để tham khảo)
    long = torch.randint(0, 256, (1, 128), device=device)
    for name, fn in (("cache", lambda: adaptive_generate_kv(model, long, 64, tiers=(2,), min_confidence=0.0)),
                     ("no cache", lambda: _greedy_no_cache(model, long, 64, 2))):
        if device == "cuda": torch.cuda.synchronize()
        t0 = time.time(); fn()
        if device == "cuda": torch.cuda.synchronize()
        print(f"[4] {name}: {time.time() - t0:.2f}s cho 64 token")
    print("TẤT CẢ OK" if ok else "CÓ LỖI - gửi mình log này")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    raise SystemExit(0 if selftest(dev) else 1)
