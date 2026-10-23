"""
bench.py - đo tốc độ thật khi ép mỗi token dùng đúng k expert (k=1 ... số expert).

Trả lời câu hỏi: bớt expert có thực sự nhanh hơn không, hay chi phí gom/rải token lấn át?
Không cần dữ liệu hay train; chỉ cần PyTorch (chạy được trên CPU, nhưng GPU mới có ý nghĩa).

    python bench.py                                  # cấu hình mặc định (giống lm_experiment)
    python bench.py --experts 32 --d_ff 128 --ks 1,4,8,16,32
"""
import argparse
import time

import torch
import torch.nn.functional as F

from adaptive_moe import set_k_cap
from lm_experiment import LM


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def timeit(fn, warmup, iters, device):
    for _ in range(warmup):
        fn()
    sync(device)
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    sync(device)
    return (time.perf_counter() - t0) / iters * 1000.0   # ms mỗi lần chạy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--d_model", type=int, default=256)
    ap.add_argument("--d_ff", type=int, default=512)
    ap.add_argument("--d_ff_shared", type=int, default=None)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--experts", type=int, default=8)
    ap.add_argument("--n_shared", type=int, default=1)
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--ks", default="1,2,4,8")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--no_train", action="store_true", help="bỏ đo forward+backward")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    E = args.experts
    ks = [int(v) for v in args.ks.split(",")]
    assert max(ks) <= E, "k không được vượt quá số expert"
    moe_kw = dict(d_ff=args.d_ff, n_experts=E, min_k=1, max_k=E, top_p=1.0, compute_coef=0.0,
                  n_shared=args.n_shared, d_ff_shared=args.d_ff_shared, rand_k=(1, E, None))
    model = LM(args.vocab, args.d_model, args.layers, args.heads, args.seq, moe_kw).to(device)
    ids = torch.randint(0, args.vocab, (args.bs, args.seq), device=device)
    tokens = args.bs * args.seq
    print(f"thiết bị: {device} | {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M tham số | "
          f"{E} expert/layer | batch {args.bs}x{args.seq}")

    rows = []
    for k in ks:
        set_k_cap(model, k)
        model.eval()

        @torch.no_grad()
        def fwd():
            model(ids)

        t_inf = timeit(fwd, args.warmup, args.iters, device)
        t_train = None
        if not args.no_train:
            model.train()

            def step():
                model.zero_grad(set_to_none=True)
                logits = model(ids)
                F.cross_entropy(logits.view(-1, args.vocab), ids.view(-1)).backward()

            t_train = timeit(step, max(2, args.warmup // 2), max(5, args.iters // 2), device)
        rows.append((k, t_inf, t_train))
    set_k_cap(model, None)

    base_inf = rows[-1][1]
    base_tr = rows[-1][2]
    print(f"\n{'k':>4}{'suy luận ms':>13}{'tok/s':>10}{'so với k=max':>14}", end="")
    print(f"{'train ms':>11}{'so với k=max':>14}" if not args.no_train else "")
    for k, ti, tt in rows:
        line = f"{k:>4}{ti:>13.2f}{tokens / ti * 1000:>10.0f}{ti / base_inf:>13.2f}x"
        if tt is not None:
            line += f"{tt:>11.2f}{tt / base_tr:>13.2f}x"
        print(line)
    print("\n(k=max là mốc 'dùng hết expert'. Tỉ lệ < 1.00x nghĩa là nhanh hơn mốc đó.)")


if __name__ == "__main__":
    main()
