"""
layerk.py - k theo TỪNG LAYER (cùng k cho mọi token trong layer đó). Train 1 model, mỗi layer bốc k độc lập
từ {1,2,4} mỗi bước; rồi quét TẤT CẢ cấu hình (k1,k2,k3,k4) và tìm đường Pareto loss - chi phí.
Chi phí = trung bình k qua các layer (chỉ tính expert được định tuyến).
Chọn cấu hình trên nửa val A, báo loss trên nửa val B (tránh chọn quá khớp trong 81 cấu hình).

python layerk.py --hf wikimedia/wikipedia --config 20231101.vi --field text --mb 8 --experts 32 --L 128 [--seed 1]
Cần adaptive_moe.py, elastic_ctx.py, headroom.py cạnh file này.
"""
import argparse, itertools, random
import torch
import torch.nn.functional as F
from adaptive_moe import set_k_cap
from elastic_ctx import CtxLM, load_bytes, batch
from headroom import load_bpe

CH = (1, 2, 4)


def train_model(train, args, device, vocab):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(1 + args.seed)
    model = CtxLM(vocab=vocab, d=args.d, n_layers=args.layers, n_experts=args.experts, d_ff=2 * args.d,
                  min_k=1, max_k=8, top_p=args.top_p, n_shared=1, compute_coef=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    for step in range(args.steps + 1):
        model.train()
        for b in model.blocks:
            b.moe.k_cap = random.choice(CH)
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
def eval_cfg(model, data, cfg, args, device, seed, n=20):
    g = torch.Generator().manual_seed(seed)
    model.eval()
    for b, k in zip(model.blocks, cfg):
        b.moe.k_cap = k
    tot = 0.0
    for _ in range(n):
        x, y = batch(data, 16, args.L, device, g)
        lg, _ = model(x, None)
        tot += F.cross_entropy(lg.float().view(-1, lg.shape[-1]), y.view(-1)).item()
    set_k_cap(model, None)
    return tot / n


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val = load_bytes(args.text, args.hf, args.field, args.mb, args.config)
    train, val, vocab = load_bpe(train, val, args.vocab)
    half = len(val) // 2
    vA, vB = val[:half], val[half:]
    model = train_model(train, args, device, vocab)
    L = args.layers
    cfgs = list(itertools.product(CH, repeat=L))
    print(f"\nquét {len(cfgs)} cấu hình k theo layer ...")
    LA = {c: eval_cfg(model, vA, c, args, device, 123) for c in cfgs}
    LB = {c: eval_cfg(model, vB, c, args, device, 456) for c in cfgs}
    print(f"\n=== k ĐỒNG ĐỀU (fixed top-k) === (seed={args.seed})")
    for k in CH:
        c = (k,) * L
        print(f"k={k} cost {k:.2f}: A {LA[c]:.4f} | B {LB[c]:.4f}")
    print("\n=== ảnh hưởng trung bình của k ở từng layer (loss A trung bình trên mọi cấu hình có layer đó = k) ===")
    print("layer".ljust(8) + "".join(f"k={k}".rjust(10) for k in CH))
    for i in range(L):
        row = f"{i}".ljust(8)
        for k in CH:
            v = [LA[c] for c in cfgs if c[i] == k]
            row += f"{sum(v) / len(v):10.4f}"
        print(row)
    print("\n=== PARETO: tốt nhất theo A ở mỗi chi phí, báo loss B (cột uniform = fixed cùng chi phí nếu có) ===")
    print("cost".ljust(7) + "cấu hình".ljust(16) + "A".rjust(9) + "B".rjust(9) + "uniform B".rjust(11))
    costs = sorted({sum(c) / L for c in cfgs})
    best_le = {}
    for cost in costs:
        group = [c for c in cfgs if abs(sum(c) / L - cost) < 1e-9]
        c = min(group, key=lambda z: LA[z])
        uni = LB.get((int(cost),) * L) if abs(cost - round(cost)) < 1e-9 and int(round(cost)) in CH else None
        print(f"{cost:<7.2f}{str(c):<16}{LA[c]:9.4f}{LB[c]:9.4f}" + (f"{uni:11.4f}" if uni is not None else "".rjust(11)))
        best_le[cost] = c
    u2 = LB[(2,) * L]
    ok = [(LB[c], c, cost) for cost, c in best_le.items() if cost <= 2.0]
    lb, c, cost = min(ok)
    print(f"\nfixed k=2 (cost 2.00): B {u2:.4f} | tốt nhất cost<=2: {c} (cost {cost:.2f}) B {lb:.4f} | chênh {lb - u2:+.4f} nat")
    print("Số âm = lịch k theo layer thắng k đồng đều ở chi phí không cao hơn.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--text", default=None)
    ap.add_argument("--hf", default=None)
    ap.add_argument("--config", default=None)
    ap.add_argument("--field", default="text")
    ap.add_argument("--mb", type=int, default=8)
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--top_p", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    main(ap.parse_args())
