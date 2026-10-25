"""
bench_fast_moe.py - kiểm tra fast_moe.py cho kết quả y hệt vòng lặp gốc và đo nó nhanh hơn bao nhiêu trên GPU của bạn.
Chạy mất khoảng nửa phút. --skew 0 là router cân bằng, --skew 4 làm vài expert "nóng" (lệch tải như router chưa có loss cân bằng).

    python bench_fast_moe.py
    python bench_fast_moe.py --skew 4
"""
import argparse, copy, time
import torch
from adaptive_moe import AdaptiveMoE
from fast_moe import enable_fast_moe, load_report


def timeit(m, x, amp, iters):
    dev = x.device.type

    def step():
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp and dev == "cuda"):
            o, _, _ = m(x)
        o.float().square().mean().backward()
        x.grad = None
    for _ in range(3):
        step()
    if dev == "cuda":
        torch.cuda.synchronize()
    t = time.time()
    for _ in range(iters):
        step()
    if dev == "cuda":
        torch.cuda.synchronize()
    return (time.time() - t) / iters * 1000


def main(a):
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(0)
    base = AdaptiveMoE(a.d, 2 * a.d, a.experts, min_k=1, max_k=8, top_p=0.999, n_shared=1, compute_coef=0.0).to(dev)
    c = torch.randn(a.d, device=dev)
    if a.skew > 0:
        with torch.no_grad():
            base.router.gate.weight[:a.hot] += a.skew * c / (c @ c)   # logit của expert nóng tăng ~skew cho mọi token
    fast = copy.deepcopy(base)
    enable_fast_moe(fast, a.waste)
    for k in (1, 2):
        base.k_cap = fast.k_cap = k
        x = torch.randn(a.B, a.L, a.d, device=dev)
        if a.skew > 0:
            x = x + c
        x.requires_grad_()
        # 1) đúng không (fp32)
        ob, _, _ = base(x)
        ob.square().mean().backward()
        gxb, gwb = x.grad.clone(), base.experts[0].w_down.weight.grad.clone()
        x.grad = None
        of, _, _ = fast(x)
        of.square().mean().backward()
        gxf, gwf = x.grad.clone(), fast.experts[0].w_down.weight.grad.clone()
        x.grad = None
        rep = load_report(fast)
        print(f"k={k}: khác biệt lớn nhất  đầu ra {float((ob - of).abs().max()):.2e}  grad x {float((gxb - gxf).abs().max()):.2e}  "
              f"grad trọng số {float((gwb - gwf).abs().max()):.2e}")
        # 2) nhanh hơn bao nhiêu
        for p in list(base.parameters()) + list(fast.parameters()):
            p.grad = None
        tb = timeit(base, x, a.amp, a.iters)
        tf = timeit(fast, x, a.amp, a.iters)
        r = rep[0] if rep else (float("nan"), "?")
        print(f"      vòng lặp {tb:6.1f} ms   fast {tf:6.1f} ms   nhanh gấp {tb / tf:.2f}x   "
              f"(đường chạy: {r[1]}, expert đông nhất / trung bình = {r[0]:.1f})")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--d", type=int, default=256)
    ap.add_argument("--experts", type=int, default=16)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--waste", type=float, default=3.0)
    ap.add_argument("--skew", type=float, default=0.0)
    ap.add_argument("--hot", type=int, default=2)
    ap.add_argument("--amp", action="store_true")
    ap.add_argument("--iters", type=int, default=20)
    main(ap.parse_args())
