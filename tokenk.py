"""
tokenk.py - k theo TỪNG token, quyết định từ id token TRƯỚC khi tính (chi phí ~ 1 + tỉ lệ token được nâng).
Cần adaptive_moe.py bản 0.3.1 (đã vá để k_cap nhận tensor [B*S]) cạnh file này, cùng elastic_ctx.py, headroom.py.

python tokenk.py --hf wikimedia/wikipedia --config 20231101.vi --field text --mb 8 --experts 32 --L 128
python tokenk.py ... --train_mode rare      # train luôn bằng luật 'token hiếm' thay vì ngẫu nhiên

Luật so sánh (đều k in {1,2}):
  k1 / k2 : cố định.
  random  : mỗi token nhận k=2 với xác suất f  (= 'fixed trộn', baseline nội suy giữa k=1 và k=2)
  rare    : nâng k=2 cho các loại token HIẾM nhất (tra bảng tần suất train) sao cho ~f số token được nâng
  common  : đối chứng, nâng token phổ biến nhất
Kết quả: val loss THẬT chạy trong từng layer (không phải phép ghép mức token như headroom.py).
"""
import argparse, random
import torch
import torch.nn.functional as F
from adaptive_moe import AdaptiveMoE, set_k_cap
from elastic_ctx import CtxLM, load_bytes, batch
from headroom import load_bpe


def selftest(device):
    torch.manual_seed(0)
    m = AdaptiveMoE(16, 32, 8, min_k=1, max_k=8, top_p=0.999, n_shared=1,
                    compute_coef=0.0).to(device).eval()
    x = torch.randn(2, 6, 16, device=device)
    cap = torch.tensor([1, 2, 1, 2, 2, 1, 1, 1, 2, 2, 1, 2], device=device)
    ok = True
    with torch.no_grad():
        m.k_cap = cap
        _, _, info = m(x)
        good = torch.equal(info["k_per_token"].round().long(), cap)
        ok &= good
        print(f"[selftest] k_cap theo token: k dùng == cap  {'OK' if good else 'FAIL'}")
        m.k_cap = 2
        _, _, info = m(x)
        good = bool((info["k_per_token"].round() == 2).all())
        ok &= good
        print(f"[selftest] k_cap=2 (int) vẫn đúng  {'OK' if good else 'FAIL'}")
        m.k_cap = None
        m(x)
        print("[selftest] k_cap=None chạy được  OK")
    if not ok:
        print("adaptive_moe.py chưa phải bản 0.3.1 đã vá - hãy thay file trong repo bằng file mình đưa.")
    return ok


def make_tables(train, vocab, fs, device):
    counts = torch.bincount(train, minlength=vocab).float()
    tot = counts.sum()
    rare_order = counts.argsort()            # hiếm -> phổ biến
    tabs = {}
    for name, order in (("rare", rare_order), ("common", rare_order.flip(0))):
        share = counts[order].cumsum(0) / tot
        for f in fs:
            t = torch.ones(vocab, dtype=torch.long)
            t[order[share <= f]] = 2
            tabs[(name, f)] = t.to(device)
    return tabs


def caps_for(rule, f, x, tabs, device, rg=None):
    ids = x.reshape(-1)
    if rule == "k1":
        return torch.ones_like(ids)
    if rule == "k2":
        return torch.full_like(ids, 2)
    if rule == "random":
        return 1 + (torch.rand(ids.shape, device=device, generator=rg) < f).long()
    return tabs[(rule, f)][ids]


def train_model(train, args, device, vocab, tabs):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(1 + args.seed)
    model = CtxLM(vocab=vocab, d=args.d, n_layers=args.layers, n_experts=args.experts, d_ff=2 * args.d,
                  min_k=1, max_k=8, top_p=args.top_p, n_shared=1, compute_coef=0.0).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    rule = "random" if args.train_mode == "random" else "rare"
    for step in range(args.steps + 1):
        model.train()
        f = random.choice((0.0, 0.25, 0.5, 0.75, 1.0))
        x, y = batch(train, args.B, args.L, device, g)
        if f == 0.0:
            cap = caps_for("k1", f, x, tabs, device)
        elif f == 1.0:
            cap = caps_for("k2", f, x, tabs, device)
        else:
            cap = caps_for(rule, f, x, tabs, device)
        set_k_cap(model, cap)
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
def eval_rule(model, val, rule, f, tabs, args, device, n=40):
    g = torch.Generator().manual_seed(123)
    rg = torch.Generator(device=device).manual_seed(0)
    model.eval()
    tot = frac = 0.0
    for _ in range(n):
        x, y = batch(val, 16, args.L, device, g)
        cap = caps_for(rule, f, x, tabs, device, rg)
        set_k_cap(model, cap)
        lg, _ = model(x, None)
        tot += F.cross_entropy(lg.float().view(-1, lg.shape[-1]), y.view(-1)).item()
        frac += (cap == 2).float().mean().item()
    set_k_cap(model, None)
    return tot / n, frac / n


@torch.no_grad()
def eval_layers(model, val, layers, args, device, n=40):
    """Đối thủ 'fixed top-k' ở chi phí 1.5: k=2 chỉ ở một số layer, k=1 ở các layer còn lại (cùng cho mọi token)."""
    g = torch.Generator().manual_seed(123)
    model.eval()
    for i, b in enumerate(model.blocks):
        b.moe.k_cap = 2 if i in layers else 1
    tot = 0.0
    for _ in range(n):
        x, y = batch(val, 16, args.L, device, g)
        lg, _ = model(x, None)
        tot += F.cross_entropy(lg.float().view(-1, lg.shape[-1]), y.view(-1)).item()
    set_k_cap(model, None)
    return tot / n


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if not selftest(device):
        raise SystemExit(1)
    if args.selftest:
        return
    train, val = load_bytes(args.text, args.hf, args.field, args.mb, args.config)
    train, val, vocab = load_bpe(train, val, args.vocab)
    fs = (0.25, 0.5, 0.75)
    tabs = make_tables(train, vocab, fs, device)
    model = train_model(train, args, device, vocab, tabs)
    l1, _ = eval_rule(model, val, "k1", 0, tabs, args, device)
    l2, _ = eval_rule(model, val, "k2", 1, tabs, args, device)
    total = l1 - l2
    print(f"\nval loss: k=1 {l1:.4f} | k=2 {l2:.4f} | gain {total:.4f} nat  (train_mode={args.train_mode}, seed={args.seed})")
    print("Mọi luật dưới đây chạy THẬT trong từng layer; chi phí expert ~ 1 + 'nâng'.")
    print("luật".ljust(9) + "f".rjust(6) + "nâng".rjust(8) + "loss".rjust(9) + "gain lấy lại".rjust(14))
    res = {}
    for f in fs:
        for rule in ("random", "rare", "common"):
            loss, frac = eval_rule(model, val, rule, f, tabs, args, device)
            res[(rule, f)] = loss
            print(rule.ljust(9) + f"{f:6.2f}" + f"{frac:8.2f}" + f"{loss:9.4f}" + f"{100 * (l1 - loss) / total:13.0f}%")
    half = args.layers // 2
    patterns = [tuple(range(half)), tuple(range(half, args.layers)), tuple(range(0, args.layers, 2)),
                tuple(range(1, args.layers, 2))]
    print(f"\nĐỐI THỦ fixed ở chi phí {1 + half / args.layers:.2f}: k=2 chỉ ở vài layer (model chưa train riêng cho kiểu này)")
    for p in patterns:
        loss = eval_layers(model, val, p, args, device)
        print(f"  k=2 ở layer {list(p)}: loss {loss:.4f} | lấy lại {100 * (l1 - loss) / total:.0f}%")
    print("  (so với rare/random ở f=0.50 phía trên, cùng chi phí)")
    print("\nrare so với random cùng f (số âm = rare tốt hơn, đơn vị nat):")
    for f in fs:
        print(f"  f={f}: {res[('rare', f)] - res[('random', f)]:+.4f}")
    print("\nĐọc: rare thắng random rõ (vài phần nghìn nat trở lên, ổn định ở cả 3 mức f) thì có chế độ")
    print("'nhẹ hơn k=2' tốt hơn trộn ngẫu nhiên, chỉ cần một bảng tra tần suất. Cần thêm seed để chắc.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
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
    ap.add_argument("--train_mode", choices=("random", "rare"), default="random")
    main(ap.parse_args())
