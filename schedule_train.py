"""
schedule_train.py - KIỂM CHỨNG CUỐI: train từ đầu từng lịch k cố định theo layer, so với k đồng đều.
(layerk.py chỉ đánh giá trên model elastic; ở đây mỗi lịch có model riêng, cùng khởi tạo, cùng thứ tự dữ liệu.)
Đo cả thời gian train thật.   bash run_schedule.sh 0     (seed 0, ~35-40 phút trên T4)
"""
import argparse, time
import torch
import torch.nn.functional as F
from adaptive_moe import set_k_cap
from elastic_ctx import CtxLM, load_bytes, batch
from headroom import load_bpe
from layerk import eval_cfg

CONFIGS = [(1, 1, 1, 1), (2, 2, 2, 2), (4, 4, 4, 4), (1, 1, 2, 2), (1, 1, 2, 4), (1, 1, 4, 4), (4, 2, 1, 1)]


def train_fixed(train, cfg, args, device, vocab):
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(1 + args.seed)
    model = CtxLM(vocab=vocab, d=args.d, n_layers=args.layers, n_experts=args.experts, d_ff=2 * args.d,
                  min_k=1, max_k=8, top_p=args.top_p, n_shared=1, compute_coef=0.0).to(device)
    for b, k in zip(model.blocks, cfg):
        b.moe.k_cap = k
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    if device == "cuda": torch.cuda.synchronize()
    t0 = time.time()
    for step in range(args.steps + 1):
        model.train()
        x, y = batch(train, args.B, args.L, device, g)
        lg, _ = model(x, None)
        loss = F.cross_entropy(lg.float().view(-1, lg.shape[-1]), y.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
    if device == "cuda": torch.cuda.synchronize()
    return model, time.time() - t0


def main(args):
    assert args.layers == 4, "CONFIGS viết cho 4 layer"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val = load_bytes(args.text, args.hf, args.field, args.mb, args.config)
    train, val, vocab = load_bpe(train, val, args.vocab)
    rows = []
    for cfg in CONFIGS:
        model, secs = train_fixed(train, cfg, args, device, vocab)
        loss = eval_cfg(model, val, cfg, args, device, 123, n=40)
        rows.append((cfg, sum(cfg) / len(cfg), loss, secs))
        print(f"{cfg} cost {sum(cfg) / 4:.2f}: val {loss:.4f} | train {secs:.0f}s", flush=True)
        del model
        if device == "cuda": torch.cuda.empty_cache()
    u2 = next(r[2] for r in rows if r[0] == (2, 2, 2, 2))
    u4 = next(r[2] for r in rows if r[0] == (4, 4, 4, 4))
    print(f"\n=== train từ đầu, seed {args.seed} ===")
    print("lịch k".ljust(14) + "cost".rjust(6) + "val loss".rjust(10) + "train s".rjust(9) + "  so với uniform cùng/cao hơn")
    for cfg, cost, loss, secs in rows:
        ref = u2 if cost <= 2.0 else u4
        name = "k=2 đồng đều" if cost <= 2.0 else "k=4 đồng đều"
        print(str(cfg).ljust(14) + f"{cost:6.2f}{loss:10.4f}{secs:9.0f}  {loss - ref:+.4f} vs {name}")
    print("\nSố âm = lịch theo layer thắng k đồng đều dù chi phí thấp hơn hoặc bằng. (4,2,1,1) là đối chứng ngược, dự kiến tệ.")


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
