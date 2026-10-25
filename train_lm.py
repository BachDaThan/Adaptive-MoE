"""
train_lm.py - train một LM MoE thật theo công thức rút ra từ các thí nghiệm, chạy được trên Colab T4 miễn phí:
  - k elastic: mỗi bước một tỉ lệ f token (f chọn ngẫu nhiên trong {0, .25, .5, .75, 1}) dùng k=2, phần còn lại k=1;
    sau khi train chọn k lúc suy luận (một model, nhiều mức ngân sách)
  - dropout 0.2 trên embedding và đầu ra MoE (dữ liệu ít), lưu checkpoint tốt nhất theo val loss
  - fp16 (autocast + GradScaler), gradient accumulation, AdamW hoặc AdamW 8-bit (bitsandbytes)
  - tự lưu checkpoint lên thư mục --out (đặt trên Google Drive) và TỰ TIẾP TỤC khi chạy lại cùng lệnh;
    --max_minutes để dừng êm trước khi Colab ngắt phiên
  - nhiều ngôn ngữ dùng chung một ByteLevel BPE (không bao giờ OOV), xem langcheck.py

Cần adaptive_moe.py 0.3.1, elastic_ctx.py, langcheck.py cạnh file này; pip install torch datasets tokenizers.
Mọi kết quả trong README là ở model khoảng 20M tham số; các preset --size lớn hơn CHƯA được kiểm chứng.

Colab:
    from google.colab import drive; drive.mount('/content/drive')
    !python train_lm.py --out /content/drive/MyDrive/adaptive_moe/run1 --langs vi --mb 100 --size small --amp --max_minutes 600
    # bị ngắt thì chạy lại đúng lệnh trên, nó đọc last.pt và đi tiếp
    !python train_lm.py --out /content/drive/MyDrive/adaptive_moe/run1 --generate "Việt Nam là" --k 2

Chạy nhanh để kiểm tra lỗi:  python train_lm.py --out run_test --mb 2 --steps 200 --eval_every 100 --save_every 100
"""
import argparse, json, math, os, random, time
import torch
import torch.nn.functional as F
from adaptive_moe import set_k_cap
from elastic_ctx import CtxLM, batch
from langcheck import read_wiki, read_file, split_docs, train_bpe, encode_docs, add_dropout

PRESETS = {  # d_ff = 2 * d như trong các thí nghiệm
    "small": dict(d=256, layers=6, experts=16, lr=1e-3),
    "medium": dict(d=384, layers=8, experts=32, lr=6e-4),
    "large": dict(d=512, layers=8, experts=32, lr=4e-4),
}


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------- dữ liệu
def build_data(args):
    path = os.path.join(args.out, "data.pt")
    if os.path.exists(path):
        d = torch.load(path)
        print(f"[train] dùng lại dữ liệu đã token hoá: {path}", flush=True)
        return d["train"].long(), {l: v.long() for l, v in d["val"].items()}, d["vocab"], d["langs"]
    langs = args.langs.split(",")
    files = dict(kv.split("=", 1) for kv in args.text_files.split(",")) if args.text_files else {}
    tr_docs, va_docs = {}, {}
    for lang in langs:
        docs = read_file(files[lang], args.mb) if lang in files else read_wiki(lang, args.mb, args.seed, args.min_chars)
        tr_docs[lang], va_docs[lang] = split_docs(docs, args.val_frac, args.seed)
        print(f"[train] {lang}: {len(tr_docs[lang])} bài train, {len(va_docs[lang])} bài val", flush=True)
    tok = train_bpe([d for l in langs for d in tr_docs[l]], args.vocab)
    tok.save(os.path.join(args.out, "tokenizer.json"))
    ids = {l: encode_docs(tok, tr_docs[l]) for l in langs}
    if args.balance:
        n = min(len(v) for v in ids.values())
        ids = {l: v[:n] for l, v in ids.items()}
    train = torch.cat(list(ids.values()))
    vals = {l: encode_docs(tok, va_docs[l]) for l in langs}
    V = tok.get_vocab_size()
    atomic_save({"train": train.int(), "val": {l: v.int() for l, v in vals.items()}, "vocab": V, "langs": langs}, path)
    print(f"[train] {len(train)} token train, vocab {V}", flush=True)
    return train, vals, V, langs


# ---------------------------------------------------------------- model
def build_model(cfg, device):
    m = CtxLM(vocab=cfg["vocab"], d=cfg["d"], n_layers=cfg["layers"], n_experts=cfg["experts"], d_ff=2 * cfg["d"],
              min_k=1, max_k=8, top_p=cfg["top_p"], n_shared=1, compute_coef=0.0).to(device)
    return m


def set_cap(model, k):
    for b in model.blocks:
        b.moe.k_cap = k


@torch.no_grad()
def evaluate(model, vals, ks, args, device, amp):
    model.eval()
    res = {}
    for k in ks:
        set_cap(model, k)
        per = {}
        for lang, v in vals.items():
            nw = min(args.eval_batches * args.B, (len(v) - 1) // args.L)
            if nw < 1:
                continue
            xs = torch.stack([v[i * args.L:(i + 1) * args.L] for i in range(nw)])
            ys = torch.stack([v[i * args.L + 1:(i + 1) * args.L + 1] for i in range(nw)])
            tot, n = 0.0, 0
            for s in range(0, nw, args.B):
                x, y = xs[s:s + args.B].to(device), ys[s:s + args.B].to(device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                    lg, _ = model(x, None)
                tot += F.cross_entropy(lg.float().view(-1, lg.shape[-1]), y.reshape(-1), reduction="sum").item()
                n += y.numel()
            per[lang] = tot / n
        res[k] = per
    return res


def lr_at(step, args, lr):
    if step < args.warmup:
        return lr * (step + 1) / args.warmup
    t = (step - args.warmup) / max(1, args.steps - args.warmup)
    return lr * (args.min_lr_ratio + (1 - args.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


# ---------------------------------------------------------------- sinh văn bản
@torch.no_grad()
def generate(args, device):
    from tokenizers import Tokenizer
    cfg = json.load(open(os.path.join(args.out, "config.json"), encoding="utf-8"))
    tok = Tokenizer.from_file(os.path.join(args.out, "tokenizer.json"))
    model = build_model(cfg, device)
    ck = torch.load(os.path.join(args.out, "best.pt"), map_location=device)
    model.load_state_dict(ck["model"])
    model.eval()
    set_cap(model, args.k)
    ids = tok.encode(args.generate).ids
    L = cfg["L"]
    for _ in range(args.max_new):
        x = torch.tensor([ids[-L:]], dtype=torch.long, device=device)
        lg, _ = model(x, None)
        logits = lg[0, -1].float() / max(args.temp, 1e-5)
        if args.top_k > 0:
            v, _ = logits.topk(args.top_k)
            logits[logits < v[-1]] = -float("inf")
        ids.append(int(torch.multinomial(logits.softmax(-1), 1)))
    print(tok.decode(ids))


# ---------------------------------------------------------------- train
def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    if args.generate is not None:
        return generate(args, device)
    P = PRESETS[args.size]
    d = args.d or P["d"]; layers = args.layers or P["layers"]; experts = args.experts or P["experts"]
    lr = args.lr or P["lr"]
    train, vals, V, langs = build_data(args)
    cfg = dict(vocab=V, d=d, layers=layers, experts=experts, top_p=args.top_p, L=args.L, langs=langs, k_max=args.k_max)
    json.dump(cfg, open(os.path.join(args.out, "config.json"), "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    random.seed(args.seed); torch.manual_seed(args.seed)
    model = build_model(cfg, device)
    if args.drop > 0:
        add_dropout(model, args.drop)
    npar = sum(p.numel() for p in model.parameters())
    opt = None
    bytes_per = 16
    if args.opt == "adamw8bit":
        try:
            import bitsandbytes as bnb
            opt = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=args.wd)
            bytes_per = 10
        except Exception as e:  # noqa: BLE001
            print(f"[train] không dùng được AdamW 8-bit ({str(e)[:60]}), quay về AdamW thường")
    if opt is None:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.wd)
    print(f"[train] {npar / 1e6:.1f}M tham số; trọng số + gradient + optimizer khoảng {npar * bytes_per / 1e9:.1f} GB "
          f"(chưa tính activation); device {device}", flush=True)
    amp = args.amp and device == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    step, best, bad = 0, (1e9, 0), 0
    last = os.path.join(args.out, "last.pt")
    if os.path.exists(last):
        ck = torch.load(last, map_location=device)
        model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"])
        if ck.get("scaler"):
            scaler.load_state_dict(ck["scaler"])
        step, best, bad = ck["step"], tuple(ck["best"]), ck["bad"]
        print(f"[train] tiếp tục từ bước {step} (val tốt nhất {best[0]:.4f} ở bước {best[1]})", flush=True)
    ks = [int(x) for x in args.eval_ks.split(",")]
    g = torch.Generator().manual_seed(args.seed + step)
    t0, t_log, tok_log = time.time(), time.time(), 0

    def save_last():
        atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict() if amp else None,
                     "step": step, "best": best, "bad": bad}, last)

    state = {"amp": amp}
    while step < args.steps:
        model.train()
        for gr in opt.param_groups:
            gr["lr"] = lr_at(step, args, lr)
        tot = 0.0
        for _ in range(args.accum):
            x, y = batch(train, args.B, args.L, device, g)
            if args.k_max <= 2:
                f = random.choice((0.0, 0.25, 0.5, 0.75, 1.0))
                set_k_cap(model, 1 + (torch.rand(x.numel(), device=device) < f).long())
            else:  # k ngẫu nhiên đều trong 1..k_max (như mục 1 trong README; chưa kiểm chứng ở quy mô lớn)
                set_k_cap(model, torch.randint(1, args.k_max + 1, (x.numel(),), device=device))
            try:
                with torch.autocast("cuda", dtype=torch.float16, enabled=state["amp"]):
                    lg, _ = model(x, None)
                loss = F.cross_entropy(lg.float().view(-1, lg.shape[-1]), y.view(-1))
                scaler.scale(loss / args.accum).backward()
            except RuntimeError as e:
                if state["amp"] and "dtype" in str(e).lower():
                    print(f"[train] fp16 không tương thích với MoE ở đây ({str(e)[:80]}); chạy lại không --amp", flush=True)
                    raise SystemExit(1)
                raise
            tot += loss.item() / args.accum
            tok_log += x.numel()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        opt.zero_grad(set_to_none=True)
        step += 1
        if step % args.log_every == 0:
            now = time.time()
            print(f"[train] bước {step}/{args.steps} loss {tot:.4f} lr {opt.param_groups[0]['lr']:.2e} "
                  f"{tok_log / (now - t_log):.0f} token/giây", flush=True)
            t_log, tok_log = now, 0
        if step % args.eval_every == 0 or step == args.steps:
            res = evaluate(model, vals, ks, args, device, state["amp"])
            show = "  ".join(f"k={k}: " + " ".join(f"{l} {v:.4f}" for l, v in res[k].items()) for k in ks)
            sel = sum(res[args.select_k].values()) / len(res[args.select_k])
            print(f"[train] ĐÁNH GIÁ bước {step}: {show}", flush=True)
            if sel < best[0]:
                best, bad = (sel, step), 0
                atomic_save({"model": model.state_dict(), "step": step, "val": res}, os.path.join(args.out, "best.pt"))
                print(f"[train] val tốt nhất mới {sel:.4f}, đã lưu best.pt", flush=True)
            else:
                bad += 1
        if step % args.save_every == 0 or step == args.steps:
            save_last()
        if args.patience and bad >= args.patience:
            print(f"[train] dừng sớm: {bad} lần đánh giá liên tiếp không cải thiện", flush=True)
            save_last()
            return
        if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
            save_last()
            print(f"[train] hết {args.max_minutes} phút, đã lưu last.pt; chạy lại cùng lệnh để tiếp tục", flush=True)
            return
    print(f"[train] xong. val tốt nhất {best[0]:.4f} ở bước {best[1]}; dùng best.pt để suy luận (--generate)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="thư mục lưu checkpoint (nên nằm trên Google Drive)")
    ap.add_argument("--langs", default="vi")
    ap.add_argument("--text_files", default=None)
    ap.add_argument("--mb", type=float, default=100, help="MB văn bản mỗi ngôn ngữ")
    ap.add_argument("--min_chars", type=int, default=1000)
    ap.add_argument("--val_frac", type=float, default=0.02)
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--balance", action="store_true", help="cắt cho mọi ngôn ngữ cùng số token train")
    ap.add_argument("--size", choices=list(PRESETS), default="small")
    ap.add_argument("--d", type=int, default=0)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--experts", type=int, default=0)
    ap.add_argument("--top_p", type=float, default=0.999)
    ap.add_argument("--k_max", type=int, default=2)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--B", type=int, default=16, help="batch mỗi lần chạy; batch hiệu dụng = B * accum")
    ap.add_argument("--accum", type=int, default=2)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--lr", type=float, default=0.0)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--min_lr_ratio", type=float, default=0.1)
    ap.add_argument("--opt", choices=["adamw", "adamw8bit"], default="adamw")
    ap.add_argument("--amp", action="store_true", help="fp16 autocast (T4 không có bf16)")
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_batches", type=int, default=20)
    ap.add_argument("--eval_ks", default="1,2")
    ap.add_argument("--select_k", type=int, default=2)
    ap.add_argument("--patience", type=int, default=0, help="dừng sau ngần này lần đánh giá không cải thiện (0 = tắt)")
    ap.add_argument("--save_every", type=int, default=500)
    ap.add_argument("--log_every", type=int, default=50)
    ap.add_argument("--max_minutes", type=float, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--generate", default=None, help="chế độ sinh văn bản từ best.pt với prompt này")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--max_new", type=int, default=100)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    main(ap.parse_args())
