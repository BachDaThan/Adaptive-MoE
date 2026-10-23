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


def train_model(train, args, device):
    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    model = CtxLM(d=args.d, n_layers=args.layers, n_experts=8, d_ff=2 * args.d,
                  min_k=1, max_k=8, top_p=args.top_p, n_shared=1, compute_coef=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    ks = (1, 2, 4)
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
    acc = {n: [] for n in ("L1", "L2", "emb", "l1", "fin", "E", "P")}
    flat = lambda t: t.float().reshape(-1, t.shape[-1])
    for _ in range(n_batches):
        x, y = batch(data, 16, args.L, device, g)
        for k in (1, 2):
            set_k_cap(model, k)
            lg, _ = model(x, None)
            lg = lg.float().view(-1, lg.shape[-1])
            acc["L%d" % k].append(F.cross_entropy(lg, y.view(-1), reduction="none"))
            if k == 1:
                p = lg.softmax(-1)
                acc["E"].append(-(p * p.clamp_min(1e-9).log()).sum(-1))
                acc["P"].append(p.max(-1).values)
                for n in ("emb", "l1", "fin"):
                    acc[n].append(flat(cap[n]))
    for h in hooks:
        h.remove()
    set_k_cap(model, None)
    d = {n: torch.cat(v) for n, v in acc.items()}
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


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val = load_bytes(args.text, args.hf, args.field, args.mb, args.config)
    model = train_model(train, args, device)
    tr = collect(model, train, 100, args, device, seed=5)
    va = collect(model, val, 40, args, device, seed=7)
    N = va["L1"].numel()
    gain = va["L1"] - va["L2"]
    ytr = tr["L1"] - tr["L2"]
    print(f"\nval: {N:,} token | loss k=1: {va['L1'].mean():.4f} | loss k=2: {va['L2'].mean():.4f} "
          f"| gain trung bình/token: {gain.mean():.4f}")
    g = torch.Generator(device=device).manual_seed(0)
    scores = {"random": torch.rand(N, device=device, generator=g), "entropy": va["E"]}
    corrs = {}
    for name in ("emb", "l1", "fin"):
        net = fit_predictor(tr["feat_" + name], ytr, device)
        with torch.no_grad():
            p = net(va["feat_" + name]).squeeze(-1)
        scores["learn_" + name] = p
        corrs[name] = torch.corrcoef(torch.stack([p, gain]))[0, 1].item()
    print("tương quan(dự đoán, gain thật): " + ", ".join(f"{k}={v:.3f}" for k, v in corrs.items()))
    total = (va["L1"].mean() - va["L2"].mean()).item()
    print("\n% mức gain của k=2 đầy đủ lấy lại được khi nâng f token lên k=2")
    print("(entropy/learn_fin cần chạy xong k=1 cả mạng rồi mới quyết định; learn_emb quyết định TRƯỚC khi tính; learn_l1 sau layer 1)")
    print("f".ljust(6) + "".join(n.rjust(13) for n in scores))
    for f in (0.1, 0.25, 0.5, 0.75):
        n = int(f * N)
        row = f"{f:<6}"
        for name, s in scores.items():
            sel = s.topk(n).indices
            loss = (va["L1"].sum() - gain[sel].sum()) / N
            share = 100 * (va["L1"].mean() - loss).item() / total
            row += f"{share:12.0f}%"
        print(row)
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
    main(ap.parse_args())
