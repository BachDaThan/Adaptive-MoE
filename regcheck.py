"""
regcheck.py - elastic có thật sự hơn fixed khi dữ liệu ít, hay chỉ vì fixed bị overfit?
Mỗi cấu hình train 3000 bước, đo val mỗi --eval_every bước trên 2 nửa val độc lập:
  chọn bước tốt nhất trên nửa A (= dừng sớm), báo loss nửa B tại bước đó (bestB) và ở bước cuối (finalB).
Cấu hình: fixed-k1/k2/k4 (+drop = dropout 0.1, +drop2 = 0.2, +wd = weight decay 0.1), elastic (k ngẫu nhiên theo
token trong {1,2}; báo ở k=1 và k=2), elastic+drop, và lịch k theo layer sched-* (cần --layers 4).
Dropout đặt trên embedding và đầu ra MoE mỗi layer.
Cần adaptive_moe.py 0.3.1, elastic_ctx.py, headroom.py, layerk.py cạnh file này.

python regcheck.py --hf wikimedia/wikipedia --config 20231101.vi --field text --mb 8 --experts 32 --L 128 --seed 0
python summarize_reg.py log_reg_s*.txt
"""
import argparse, random
import torch
import torch.nn.functional as F
from adaptive_moe import set_k_cap
from elastic_ctx import CtxLM, load_bytes, batch
from headroom import load_bpe
from layerk import eval_cfg

# (tên, k cố định hoặc None = elastic, dropout, weight decay)
CONFIGS = [("fixed-k1", 1, 0.0, 0.01), ("fixed-k1+drop", 1, 0.1, 0.01), ("fixed-k2", 2, 0.0, 0.01),
           ("fixed-k2+drop", 2, 0.1, 0.01), ("fixed-k2+wd", 2, 0.0, 0.1), ("elastic", None, 0.0, 0.01),
           # vòng 2: elastic có dropout, dropout mạnh hơn, k=4, và lịch k theo layer có dropout
           ("elastic+drop", None, 0.1, 0.01), ("fixed-k2+drop2", 2, 0.2, 0.01), ("fixed-k4+drop", 4, 0.1, 0.01),
           ("sched-1124+drop", (1, 1, 2, 4), 0.1, 0.01), ("sched-4211+drop", (4, 2, 1, 1), 0.1, 0.01),
           ("sched-1122+drop", (1, 1, 2, 2), 0.1, 0.01)]


def add_dropout(model, p):
    model.emb.register_forward_hook(lambda m, i, o: F.dropout(o, p, m.training))

    def hook(m, i, o):
        y, aux, info = o
        return (F.dropout(y, p, m.training), aux, info)
    for b in model.blocks:
        b.moe.register_forward_hook(hook)


def run_one(name, k, drop, wd, train, vA, vB, vocab, args, device):
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(1 + args.seed)
    model = CtxLM(vocab=vocab, d=args.d, n_layers=args.layers, n_experts=args.experts, d_ff=2 * args.d,
                  min_k=1, max_k=8, top_p=args.top_p, n_shared=1, compute_coef=0.0).to(device)
    if drop > 0:
        add_dropout(model, drop)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=wd)
    L = args.layers
    if k is None:
        ks = [(1,) * L, (2,) * L]
    elif isinstance(k, tuple):
        assert len(k) == L, "lịch k theo layer cần đúng số layer"
        ks = [k]
    else:
        ks = [(k,) * L]
    hist = {cfg: [] for cfg in ks}
    for step in range(args.steps + 1):
        model.train()
        x, y = batch(train, args.B, args.L, device, g)
        if k is not None:
            for b, kk in zip(model.blocks, ks[0]):
                b.moe.k_cap = kk
        else:
            f = random.choice((0.0, 0.25, 0.5, 0.75, 1.0))
            set_k_cap(model, 1 + (torch.rand(x.numel(), device=device) < f).long())
        lg, _ = model(x, None)
        loss = F.cross_entropy(lg.float().view(-1, lg.shape[-1]), y.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step > 0 and (step % args.eval_every == 0 or step == args.steps):
            for cfg in ks:
                a = eval_cfg(model, vA, cfg, args, device, 111, n=20)
                b = eval_cfg(model, vB, cfg, args, device, 222, n=20)
                hist[cfg].append((step, a, b, loss.item()))
    for cfg in ks:
        best = min(hist[cfg], key=lambda t: t[1])
        fin = hist[cfg][-1]
        row = name if k is not None else f"{name}@k{cfg[0]}"
        print(f"RESULT seed={args.seed} row={row} best_step={best[0]} bestB={best[2]:.4f} finalB={fin[2]:.4f} "
              f"final_train_loss={fin[3]:.3f}", flush=True)
    del model, opt
    if device == "cuda":
        torch.cuda.empty_cache()


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train, val = load_bytes(args.text, args.hf, args.field, args.mb, args.config)
    train, val, vocab = load_bpe(train, val, args.vocab)
    half = len(val) // 2
    vA, vB = val[:half], val[half:]
    only = set(args.only.split(",")) if args.only else None
    valid = [c[0] for c in CONFIGS]
    if only:
        bad = sorted(only - set(valid))
        if bad:
            raise SystemExit(f"[regcheck] tên cấu hình không có trong file này: {bad}\n"
                             f"Các tên hợp lệ: {valid}\n"
                             "Nếu thiếu elastic+drop / fixed-k4+drop / sched-*: file regcheck.py trên máy chạy là bản cũ, hãy thay bằng bản mới.")
    todo = [c for c in CONFIGS if not only or c[0] in only]
    print(f"[regcheck] seed={args.seed} sẽ chạy {len(todo)} cấu hình: {[c[0] for c in todo]}", flush=True)
    for name, k, drop, wd in todo:
        print(f"[regcheck] bắt đầu {name}", flush=True)
        run_one(name, k, drop, wd, train, vA, vB, vocab, args, device)


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
    ap.add_argument("--eval_every", type=int, default=250)
    ap.add_argument("--top_p", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--only", default=None, help="tên cấu hình, cách nhau bằng dấu phẩy")
    main(ap.parse_args())
