"""
check_load.py - với model đã train (best.pt), router có lệch tải không và --fast_moe nhanh hơn thật bao nhiêu. Không đổi checkpoint.
    python check_load.py --out /content/drive/MyDrive/adaptive_moe --amp
"""
import argparse, os, time, types
import torch
import torch.nn.functional as F
from chat import load
from elastic_ctx import batch
from fast_moe import enable_fast_moe, disable_fast_moe, load_report


def main(a):
    model, tok, cfg, dev = load(types.SimpleNamespace(out=a.out, hf_repo=None, hf_type="dataset", device=None))
    train = torch.load(os.path.join(a.out, "data.pt"))["train"].long()
    x, y = batch(train, a.B, cfg["L"], dev, torch.Generator().manual_seed(0))
    model.train()

    def step():
        with torch.autocast("cuda", dtype=torch.float16, enabled=a.amp and dev == "cuda"):
            lg, _ = model(x, None)
        F.cross_entropy(lg.float().view(-1, lg.shape[-1]), y.view(-1)).backward()
        model.zero_grad(set_to_none=True)

    def timed():
        for _ in range(2):
            step()
        if dev == "cuda":
            torch.cuda.synchronize()
        t = time.time()
        for _ in range(a.iters):
            step()
        if dev == "cuda":
            torch.cuda.synchronize()
        return (time.time() - t) / a.iters * 1000

    for k in (1, 2):
        for b in model.blocks:
            b.moe.k_cap = k
        disable_fast_moe(model)
        tl = timed()
        enable_fast_moe(model, a.waste)
        tf = timed()
        rep = load_report(model)
        print(f"k={k}: một bước (forward + backward) vòng lặp {tl:.0f} ms, fast {tf:.0f} ms, nhanh gấp {tl / tf:.2f}x")
        print("   tải expert (đông nhất / trung bình) theo layer: " + " ".join(f"{r:.1f}({p})" for r, p in rep))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--waste", type=float, default=3.0)
    ap.add_argument("--amp", action="store_true")
    main(ap.parse_args())
