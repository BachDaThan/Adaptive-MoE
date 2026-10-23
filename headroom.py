"""
headroom.py - đo xem cấp k theo từng token có THẮNG cấp k ngẫu nhiên (= fixed trộn) hay không.
Đặt cạnh adaptive_moe.py và elastic_ctx.py.

python headroom.py --hf wikimedia/wikipedia --config 20231101.vi --field text --mb 8

Quy trình: train 1 model elastic (k ngẫu nhiên 1/2/4), chạy val ở k=1 và k=2, lấy loss từng token.
gain = loss(k=1) - loss(k=2). Với ngân sách f (tỉ lệ token được nâng lên k=2), so các cách chọn:
  random   : bốc ngẫu nhiên (baseline = 'fixed trộn')
  entropy  : cascade - nâng token mà dự đoán k=1 kém chắc chắn nhất
  learned  : MLP nhỏ trên hidden state k=1 dự đoán gain
  oracle   : biết trước gain thật (trần lý thuyết, gian lận)
Xấp xỉ: ghép loss ở mức token (bỏ qua ảnh hưởng chéo giữa token qua attention).
Nếu oracle ≈ random thì không bộ phân bổ nào thắng nổi fixed ở quy mô này.
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
    g = torch.Generator().manual_seed(seed)
    cap = {}
    hook = model.norm.register_forward_hook(lambda m, i, o: cap.__setitem__("h", o.detach()))
    model.eval(); set_window(model, None)
    L1, L2, H, E, P = [], [], [], [], []
    for _ in range(n_batches):
        x, y = batch(data, 16, args.L, device, g)
        out = {}
        for k in (1, 2):
            set_k_cap(model, k)
            lg, _ = model(x, None)
            lg = lg.float().view(-1, lg.shape[-1])
            out[k] = F.cross_entropy(lg, y.view(-1), reduction="none")
            if k == 1:
                p = lg.softmax(-1)
                E.append(-(p * p.clamp_min(1e-9).log()).sum(-1))
                P.append(p.max(-1).values)
                H.append(cap["h"].float().reshape(-1, cap["h"].shape[-1]))
        L1.append(out[1]); L2.append(out[2])
    hook.remove(); set_k_cap(model, None)
    return [torch.cat(t) for t in (L1, L2, H, E, P)]


def fit_predictor(tr, args, device):
    L1, L2, H, E, P = tr
    X = torch.cat([H, E[:, None], P[:, None]], 1)
    y = (L1 - L2)
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
    net = fit_predictor(tr, args, device)
    L1, L2, H, E, P = va
    N = L1.numel()
    gain = L1 - L2
    with torch.no_grad():
        pred = net(torch.cat([H, E[:, None], P[:, None]], 1)).squeeze(-1)
    corr = torch.corrcoef(torch.stack([pred, gain]))[0, 1].item()
    print(f"\nval: {N:,} token | loss k=1: {L1.mean():.4f} | loss k=2: {L2.mean():.4f} "
          f"| gain trung bình/token: {gain.mean():.4f}")
    print(f"tương quan(dự đoán gain, gain thật) = {corr:.3f}")
    g = torch.Generator(device=device).manual_seed(0)
    scores = {"random": torch.rand(N, device=device, generator=g), "entropy": E,
              "learned": pred, "oracle": gain}
    total = (L1.mean() - L2.mean()).item()
    print("\nloss val khi nâng f token lên k=2 (trong ngoặc: % mức gain của k=2 đầy đủ lấy lại được)")
    print("f".ljust(6) + "".join(n.rjust(18) for n in scores))
    for f in (0.1, 0.25, 0.5, 0.75):
        n = int(f * N)
        row = f"{f:<6}"
        for name, s in scores.items():
            sel = s.topk(n).indices
            loss = (L1.sum() - gain[sel].sum()) / N
            share = 100 * (L1.mean() - loss).item() / total
            row += f"{loss.item():.4f} ({share:3.0f}%)".rjust(18)
        print(row)
    print("\nĐọc: random ~ f% (fixed trộn). Nếu learned/entropy chỉ ngang random thì không có đất cho bộ phân bổ.")
    print("Nếu oracle cao hơn random nhiều mà learned thấp: cần tín hiệu tốt hơn. Nếu learned gần oracle: có hướng thắng fixed.")


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
