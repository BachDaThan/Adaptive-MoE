"""
lm_experiment.py - train một LM nhỏ trên tiếng Việt thật, rồi báo cáo số expert (k)
theo loại token, so sánh adaptive với baseline top-k cố định.

Colab:
    !pip -q install datasets
    !python lm_experiment.py --mode adaptive --target_k 3.0
    !python lm_experiment.py --mode fixed --k 3
Dữ liệu: mặc định tải một phần Wikipedia tiếng Việt (wikimedia/wikipedia, 20231101.vi).
Nếu không tải được, truyền file text của bạn: --text_file duong_dan.txt

Câu hỏi thí nghiệm:
  1) Adaptive có đạt val loss gần với fixed-k ở cùng số expert trung bình không?
  2) k có tăng ở token khó (loss cao) và tên riêng/từ hiếm, giảm ở từ chức năng/dấu câu không?
"""
import argparse
import json
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from adaptive_moe import AdaptiveMoE, difficulty_loss, gather_aux_loss, set_k_cap
from vi_data import CATS, build_vocab, categorize, encode, tokenize


# ----------------------------------------------------------------------------
# Dữ liệu
# ----------------------------------------------------------------------------
def load_text(args):
    if args.text_file:
        with open(args.text_file, encoding="utf-8") as f:
            return f.read()[: args.max_chars]
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", "20231101.vi", split="train", streaming=True)
    parts, n = [], 0
    for ex in ds:
        parts.append(ex["text"])
        n += len(ex["text"])
        if n >= args.max_chars:
            break
    return "\n".join(parts)


# ----------------------------------------------------------------------------
# Model: decoder-only transformer nhỏ, FFN = AdaptiveMoE
# ----------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, d, heads, moe_kw):
        super().__init__()
        self.h = heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.proj = nn.Linear(d, d, bias=False)
        self.ln2 = nn.LayerNorm(d)
        self.moe = AdaptiveMoE(d_model=d, **moe_kw)

    def forward(self, x):
        B, S, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, S, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(B, S, D))
        y, _, _ = self.moe(self.ln2(x))
        return x + y


class LM(nn.Module):
    def __init__(self, vocab, d, layers, heads, seq, moe_kw):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.pos = nn.Embedding(seq, d)
        kws = moe_kw if isinstance(moe_kw, list) else [moe_kw] * layers
        self.blocks = nn.ModuleList(Block(d, heads, kws[i]) for i in range(layers))
        self.ln_f = nn.LayerNorm(d)
        self.head = nn.Linear(d, vocab, bias=False)
        self.head.weight = self.emb.weight          # weight tying

    def forward(self, ids):
        S = ids.shape[1]
        x = self.emb(ids) + self.pos(torch.arange(S, device=ids.device))
        for b in self.blocks:
            x = b(x)
        return self.head(self.ln_f(x))


# ----------------------------------------------------------------------------
# Đánh giá + báo cáo
# ----------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, val_ids, val_cat, seq, bs, device, vocab):
    model.eval()
    moes = [m for m in model.modules() if isinstance(m, AdaptiveMoE)]
    n = (len(val_ids) - 1) // seq
    X = val_ids[: n * seq].reshape(n, seq)
    Y = val_ids[1: n * seq + 1].reshape(n, seq)
    C_in = val_cat[: n * seq].reshape(n, seq)
    C_tgt = val_cat[1: n * seq + 1].reshape(n, seq)

    losses, ks, layer_ks, diffs = [], [], [], []
    for i in range(0, n, bs):
        x = torch.from_numpy(X[i:i + bs]).to(device)
        y = torch.from_numpy(Y[i:i + bs]).to(device)
        b = x.shape[0]
        logits = model(x)
        loss = F.cross_entropy(logits.reshape(-1, vocab), y.reshape(-1), reduction="none").view(b, seq)
        per_layer = torch.stack([m.last_info["k_per_token"].view(b, seq) for m in moes])  # [L,b,S]
        if moes[0].last_diff_logit is not None:
            d = torch.stack([torch.sigmoid(m.last_diff_logit).view(b, seq) for m in moes]).mean(0)
            diffs.append(d.cpu().numpy())
        losses.append(loss.cpu().numpy())
        ks.append(per_layer.mean(0).cpu().numpy())
        layer_ks.append(per_layer.mean((1, 2)).cpu().numpy() * b)
    losses = np.concatenate(losses)
    ks = np.concatenate(ks)
    layer_avg = (np.sum(layer_ks, axis=0) / n).tolist()

    res = {
        "val_loss": float(losses.mean()),
        "val_ppl": float(math.exp(losses.mean())),
        "avg_k": float(ks.mean()),
        "avg_k_per_layer": layer_avg,
        "corr_k_loss": float(np.corrcoef(ks.ravel(), losses.ravel())[0, 1]) if ks.std() > 0 else 0.0,
    }

    def table(cat_arr):
        rows = {}
        for ci, name in enumerate(CATS):
            m = cat_arr == ci
            if m.sum() == 0:
                continue
            rows[name] = {"share_%": float(100 * m.mean()), "loss": float(losses[m].mean()),
                          "k": float(ks[m].mean())}
        return rows

    if diffs:
        dd = np.concatenate(diffs)
        res["corr_pred_difficulty_loss"] = float(np.corrcoef(dd.ravel(), losses.ravel())[0, 1])
    res["by_target_category"] = table(C_tgt)
    res["by_input_category"] = table(C_in)

    q = np.quantile(losses, [0.25, 0.5, 0.75])
    edges = [-np.inf, *q, np.inf]
    res["k_by_loss_quartile"] = [float(ks[(losses > edges[i]) & (losses <= edges[i + 1])].mean())
                                 for i in range(4)]
    return res


def print_report(res, title):
    print(f"\n===== {title} =====")
    print(f"val_loss={res['val_loss']:.4f}  ppl={res['val_ppl']:.1f}  avg_k={res['avg_k']:.2f}")
    print("avg_k theo layer:", " ".join(f"{v:.2f}" for v in res["avg_k_per_layer"]))
    print(f"tương quan Pearson(k, loss của token) = {res['corr_k_loss']:+.3f}")
    if "corr_pred_difficulty_loss" in res:
        print(f"tương quan Pearson(độ khó dự đoán, loss thật) = {res['corr_pred_difficulty_loss']:+.3f}")
    print("k theo tứ phân vị loss (dễ -> khó):", " ".join(f"{v:.2f}" for v in res["k_by_loss_quartile"]))
    for key, label in [("by_target_category", "Theo loại token SẮP ĐƯỢC DỰ ĐOÁN (độ khó thật)"),
                       ("by_input_category", "Theo loại token ĐẦU VÀO (cái router nhìn thấy)")]:
        print(f"\n{label}")
        print(f"{'loại':<14}{'tỉ lệ %':>9}{'loss':>8}{'k':>7}")
        for name, r in res[key].items():
            print(f"{name:<14}{r['share_%']:>9.1f}{r['loss']:>8.2f}{r['k']:>7.2f}")


# ----------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["adaptive", "fixed"], default="adaptive")
    ap.add_argument("--k", type=int, default=3, help="số expert khi --mode fixed")
    ap.add_argument("--target_k", type=float, default=3.0)
    ap.add_argument("--top_p", type=float, default=0.7)
    ap.add_argument("--compute_coef", type=float, default=0.5)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--n_shared", type=int, default=1)
    ap.add_argument("--diff_head", action="store_true",
                    help="router có đầu dự đoán độ khó (huấn luyện bằng loss thật của token) và đặt top_p theo từng token")
    ap.add_argument("--diff_target", choices=["loss", "gain"], default="loss",
                    help="loss: đầu độ khó học theo loss của token. gain: học theo mức loss GIẢM được "
                         "khi thêm expert (k=1 so với k=gain_high), đo bằng 2 forward phụ")
    ap.add_argument("--gain_every", type=int, default=4, help="cứ N bước đo gain một lần (tiết kiệm thời gian)")
    ap.add_argument("--gain_high", type=int, default=3)
    ap.add_argument("--layer_k", default=None,
                    help="k cố định riêng cho từng layer, vd 1,2,2,3 (dùng với --mode fixed)")
    ap.add_argument("--p_min", type=float, default=0.4)
    ap.add_argument("--p_max", type=float, default=0.95)
    ap.add_argument("--diff_coef", type=float, default=0.1)
    ap.add_argument("--ste", action="store_true", help="gradient của task loss chảy vào quyết định chọn expert")
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--d_ff", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--rare_count", type=int, default=30)
    ap.add_argument("--max_chars", type=int, default=8_000_000)
    ap.add_argument("--text_file", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True

    print("Đang tải và tokenize dữ liệu...")
    toks = tokenize(load_text(args))
    split = int(len(toks) * 0.95)
    train_toks, val_toks = toks[:split], toks[split:]
    itos, stoi, cnt = build_vocab(train_toks, args.vocab)
    V = len(itos)
    train_ids = torch.from_numpy(encode(train_toks, stoi))
    val_ids = encode(val_toks, stoi)
    val_cat = categorize(val_toks, cnt, args.rare_count)
    print(f"{len(toks):,} token | vocab {V} | train {len(train_ids):,} | val {len(val_ids):,}")

    if args.mode == "adaptive":
        moe_kw = dict(d_ff=args.d_ff, n_experts=args.experts, top_p=args.top_p,
                      target_k=args.target_k, compute_coef=args.compute_coef,
                      n_shared=args.n_shared, noise_std=0.1, ste=args.ste,
                      difficulty=args.diff_head, p_min=args.p_min, p_max=args.p_max)
    else:  # top-k cố định: min_k = max_k = k, top_p = 1
        moe_kw = dict(d_ff=args.d_ff, n_experts=args.experts, min_k=args.k, max_k=args.k,
                      top_p=1.0, compute_coef=0.0, n_shared=args.n_shared, noise_std=0.1)
    if args.layer_k:
        ks = [int(v) for v in args.layer_k.split(",")]
        assert len(ks) == args.layers, "--layer_k phải có đúng một số cho mỗi layer"
        moe_kw = [dict(moe_kw, min_k=k, max_k=k, top_p=1.0, compute_coef=0.0, target_k=None,
                       difficulty=False) for k in ks]
    model = LM(V, args.d_model, args.layers, args.heads, args.seq, moe_kw).to(device)
    nn.init.normal_(model.emb.weight, std=0.02)   # init nhỏ, tránh loss khởi đầu ~177
    nn.init.normal_(model.pos.weight, std=0.02)
    print(f"{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M tham số | thiết bị: {device}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    warm = min(200, args.steps // 10)

    def lr_at(s):
        if s < warm:
            return args.lr * (s + 1) / warm
        p = (s - warm) / max(1, args.steps - warm)
        return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * p)))

    t0 = time.time()
    for step in range(args.steps + 1):
        model.train()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        ix = torch.randint(0, len(train_ids) - args.seq - 1, (args.bs,))
        x = torch.stack([train_ids[i:i + args.seq] for i in ix]).to(device)
        y = torch.stack([train_ids[i + 1:i + args.seq + 1] for i in ix]).to(device)
        logits = model(x)
        tok = F.cross_entropy(logits.view(-1, V), y.view(-1), reduction="none")
        task = tok.mean()
        loss = task + gather_aux_loss(model)
        if args.diff_head:
            if args.diff_target == "loss":
                loss = loss + args.diff_coef * difficulty_loss(model, tok)
            elif step % args.gain_every == 0:
                # nhãn "lợi ích của việc thêm expert": loss(k=1) - loss(k=gain_high), không gradient
                moes = [m for m in model.modules() if isinstance(m, AdaptiveMoE)]
                saved = [(m.last_diff_logit, m.last_info) for m in moes]
                model.eval()
                with torch.no_grad():
                    set_k_cap(model, 1)
                    low = F.cross_entropy(model(x).view(-1, V), y.view(-1), reduction="none")
                    set_k_cap(model, args.gain_high)
                    high = F.cross_entropy(model(x).view(-1, V), y.view(-1), reduction="none")
                    set_k_cap(model, None)
                model.train()
                for m, (dl_, inf_) in zip(moes, saved):   # khôi phục kết quả của forward chính
                    m.last_diff_logit, m.last_info = dl_, inf_
                loss = loss + args.diff_coef * difficulty_loss(model, low - high)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 100 == 0:
            ks = [m.last_info["avg_k"].item() for m in model.modules() if isinstance(m, AdaptiveMoE)]
            print(f"step {step:5d}  loss={task.item():.3f}  avg_k={sum(ks) / len(ks):.2f}  "
                  f"{time.time() - t0:.0f}s")

    title = f"{args.mode}{'+ste' if args.ste else ''}{'+diff' if args.diff_head else ''}{'+gain' if args.diff_head and args.diff_target == 'gain' else ''}" + (f" k={args.layer_k or args.k}" if args.mode == "fixed" else f" target_k={args.target_k}")
    res = evaluate(model, val_ids, val_cat, args.seq, args.bs, device, V)
    print_report(res, title)
    out = args.out or f"results_{args.mode}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump({"args": vars(args), "result": res}, f, ensure_ascii=False, indent=2)
    print(f"\nĐã lưu {out}")


if __name__ == "__main__":
    main()
