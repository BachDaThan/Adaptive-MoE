"""
headroom.py - đo xem cấp k theo từng token có THẮNG cấp k ngẫu nhiên (= fixed trộn) hay không.
Đặt cạnh adaptive_moe.py và elastic_ctx.py.

python headroom.py --hf wikimedia/wikipedia --config 20231101.vi --field text --mb 8

Quy trình: train 1 model elastic (k ngẫu nhiên 1/2/4), chạy val ở k=1 và k=2, lấy loss từng token.
gain = loss(k=1) - loss(k=2). Với ngân sách f (tỉ lệ token được nâng lên k=2), so các cách chọn:
  random   : bốc ngẫu nhiên (baseline = 'fixed trộn')
  entropy  : cascade - nâng token mà dự đoán k=1 kém chắc chắn nhất
  learn_*  : MLP nhỏ dự đoán gain từ đặc trưng sớm (emb), sau layer 1 (l1) hoặc cuối mạng (fin)
(oracle đã bỏ: chọn theo loss thật cộng cả nhiễu từng token nên cho số >100%, không phải trần có nghĩa)
Xấp xỉ: ghép loss ở mức token (bỏ qua ảnh hưởng chéo giữa token qua attention).
"""
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from adaptive_moe import set_k_cap
from elastic_ctx import CtxLM, set_window, load_bytes, batch


def train_model(train, args, device, vocab):
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    model = CtxLM(vocab=vocab, d=args.d, n_layers=args.layers, n_experts=args.experts, d_ff=2 * args.d,
                  min_k=1, max_k=max(8, 2 * args.k_hi), top_p=args.top_p, n_shared=1,
                  compute_coef=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    ks = tuple(sorted({1, 2, args.k_hi, 2 * args.k_hi}))
    for step in range(args.steps + 1):
        model.train()
        set_window(model, None)
        set_k_cap(model, ks[torch.randint(len(ks), (1,)).item()])
        x, y = batch(train, args.B, args.L, device, g)
        lg, _ = model(x, None)
        loss = F.cross_entropy(lg.float().view(-1, lg.shape[-1]), y.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 500 == 0:
            print(f"step {step} loss={loss.item():.3f}")
    set_k_cap(model, None)
    return model


@torch.no_grad()
def collect(model, data, n_batches, args, device, seed):
    """Trả về loss k=1, loss k=2, và các nhóm đặc trưng: emb (trước mọi layer), L1 (sau layer 1), final (+entropy, maxprob)."""
    g = torch.Generator().manual_seed(seed)
    cap = {}
    hooks = [
        model.emb.register_forward_hook(lambda m, i, o: cap.__setitem__("emb", o.detach())),
        model.blocks[0].register_forward_hook(lambda m, i, o: cap.__setitem__("l1", o[0].detach())),
        model.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("fin", o.detach())),
    ]
    model.eval(); set_window(model, None)
    ks_eval = sorted({k for k in (1, 2, 3, args.k_hi) if k <= args.k_hi})
    acc = {n: [] for n in ("L1", "L2", "emb", "l1", "fin", "E", "P", "tok")}
    lk = {k: [] for k in ks_eval}
    flat = lambda t: t.float().reshape(-1, t.shape[-1])
    for _ in range(n_batches):
        x, y = batch(data, 16, args.L, device, g)
        for k in ks_eval:
            set_k_cap(model, k)
            lg, _ = model(x, None)
            lg = lg.float().view(-1, lg.shape[-1])
            ce = F.cross_entropy(lg, y.view(-1), reduction="none")
            lk[k].append(ce)
            if k == 1:
                acc["L1"].append(ce)
            if k == args.k_hi:
                acc["L2"].append(ce)
            if k == 1:
                acc["tok"].append(x.reshape(-1))
                p = lg.softmax(-1)
                acc["E"].append(-(p * p.clamp_min(1e-9).log()).sum(-1))
                acc["P"].append(p.max(-1).values)
                for n in ("emb", "l1", "fin"):
                    acc[n].append(flat(cap[n]))
    for h in hooks:
        h.remove()
    set_k_cap(model, None)
    d = {n: torch.cat(v) for n, v in acc.items()}
    d["Lk"] = {k: torch.cat(v) for k, v in lk.items()}
    d["feat_emb"] = d["emb"]
    d["feat_l1"] = d["l1"]
    d["feat_fin"] = torch.cat([d["fin"], d["E"][:, None], d["P"][:, None]], 1)
    return d


def fit_predictor(X, y, device):
    net = nn.Sequential(nn.Linear(X.shape[1], 128), nn.GELU(), nn.Linear(128, 1)).to(device)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=0.01)
    N = X.shape[0]
    for ep in range(6):
        perm = torch.randperm(N, device=device)
        for i in range(0, N, 4096):
            idx = perm[i:i + 4096]
            loss = F.mse_loss(net(X[idx]).squeeze(-1), y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
    return net


def load_bpe(train, val, vocab):
    """BPE byte-level cỡ lm_experiment (16000). Chia train/val THEO DÒNG NGẪU NHIÊN để val cùng phân phối."""
    import random
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    text = bytes(torch.cat([train, val]).tolist()).decode("utf-8", errors="ignore")
    lines = [l for l in text.split("\n") if l.strip()]
    random.Random(0).shuffle(lines)
    cut_l = int(len(lines) * 0.95)
    tr_lines, va_lines = lines[:cut_l], lines[cut_l:]
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    trn = trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<unk>"],
                              initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(tr_lines, trn)
    enc = lambda ls: torch.tensor([i for e in tok.encode_batch(ls) for i in e.ids], dtype=torch.long)
    tr_ids, va_ids = enc(tr_lines), enc(va_lines)
    print(f"BPE: vocab {tok.get_vocab_size()} | train {len(tr_ids):,} | val {len(va_ids):,} token (chia theo dòng ngẫu nhiên)")
    return tr_ids, va_ids, tok.get_vocab_size()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val = load_bytes(args.text, args.hf, args.field, args.mb, args.config)
    vocab = 256
    if args.bpe:
        train, val, vocab = load_bpe(train, val, args.vocab)
    model = train_model(train, args, device, vocab)
    tr = collect(model, train, 100, args, device, seed=5)
    va = collect(model, val, 40, args, device, seed=7)
    N = va["L1"].numel()
    gain = va["L1"] - va["L2"]
    ytr = tr["L1"] - tr["L2"]
    print(f"\nval: {N:,} token | loss k=1: {va['L1'].mean():.4f} | loss k={args.k_hi}: {va['L2'].mean():.4f} "
          f"| gain trung bình/token: {gain.mean():.4f}")
    g = torch.Generator(device=device).manual_seed(0)
    counts = torch.bincount(train.to(device), minlength=vocab).float()
    logf = torch.log(counts[va["tok"]] + 1)
    scores = {"random": torch.rand(N, device=device, generator=g), "entropy": va["E"],
              "rare": -logf, "common": logf}
    corrs = {}
    for name in ("emb", "l1", "fin"):
        net = fit_predictor(tr["feat_" + name], ytr, device)
        with torch.no_grad():
            p = net(va["feat_" + name]).squeeze(-1)
        scores["learn_" + name] = p
        corrs[name] = torch.corrcoef(torch.stack([p, gain]))[0, 1].item()
    print("tương quan(dự đoán, gain thật): " + ", ".join(f"{k}={v:.3f}" for k, v in corrs.items()))
    total = (va["L1"].mean() - va["L2"].mean()).item()
    print("\n% mức gain của k=k_hi đầy đủ lấy lại được khi nâng f token lên k=k_hi")
    print("(rare/common: luật tần suất token, không cần học; entropy/learn_fin cần chạy xong k=1 cả mạng rồi mới quyết định; learn_emb quyết định TRƯỚC khi tính; learn_l1 sau layer 1)")
    print("f".ljust(6) + "".join(n.rjust(11) for n in scores))
    for f in (0.1, 0.25, 0.5, 0.75):
        n = int(f * N)
        row = f"{f:<6}"
        for name, s in scores.items():
            sel = s.topk(n).indices
            loss = (va["L1"].sum() - gain[sel].sum()) / N
            share = 100 * (va["L1"].mean() - loss).item() / total
            row += f"{share:10.0f}%"
        print(row)
    if args.k_hi > 2:
        print("\nSO VỚI FIXED k CÙNG CHI PHÍ TRUNG BÌNH (cost = 1 + f*(k_hi-1)) - đây mới là baseline công bằng")
        for m in sorted(m for m in va["Lk"] if 1 < m < args.k_hi):
            fixed = 100 * (va["L1"].mean() - va["Lk"][m].mean()).item() / total
            f = (m - 1) / (args.k_hi - 1)
            n = int(f * N)
            parts = []
            for name, s_ in scores.items():
                sel = s_.topk(n).indices
                loss = (va["L1"].sum() - gain[sel].sum()) / N
                parts.append(f"{name}={100 * (va['L1'].mean() - loss).item() / total:.0f}%")
            print(f"cost {m}: fixed k={m} lấy {fixed:.0f}% gain | phân bổ k=1/k={args.k_hi} với f={f:.2f}: " + ", ".join(parts))
    print("\nrandom ~ f% là baseline 'fixed trộn'. Cột nào cao hơn random rõ rệt ở f=0.25 là tín hiệu dùng được.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None)
    ap.add_argument("--hf", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--field", default="text")
    ap.add_argument("--mb", type=int, default=8)
    ap.add_argument("--L", type=int, default=256)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--top_p", type=float, default=0.999)
    ap.add_argument("--bpe", action="store_true", help="dùng BPE thay vì byte (k ảnh hưởng nhiều hơn)")
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--k_hi", type=int, default=2, help="so sánh k=1 với k=k_hi")
    main(ap.parse_args())
