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

Đồng bộ lên Hugging Face (tuỳ chọn): thêm  --hf_repo ten/repo  (mặc định kiểu dataset, riêng tư). Tự tải config, tokenizer,
data, last.pt, best.pt về nếu thư mục --out chưa có, và tự đẩy last.pt, best.pt lên mỗi --hf_every bước, khi dừng, khi Ctrl+C.
Cần đăng nhập trước: from huggingface_hub import notebook_login; notebook_login()

Thêm dữ liệu web tiếng Việt (nhiều hơn Wikipedia nhiều lần): --langs vi,web --hf_dataset HuggingFaceFW/fineweb-2:vie_Latn:text --hf_mb 300
(đổi dữ liệu thì phải train tokenizer mới, nên dùng thư mục --out mới, không resume được từ checkpoint cũ).

Tuỳ chọn tăng tốc / cân bằng tải: --fast_moe (bmm thay vòng lặp expert, xem fast_moe.py) và --aux 1 (cộng loss cân bằng tải của router;
mặc định 0 = như các thí nghiệm trong README, vốn chưa dùng loss này).

Chạy nhanh để kiểm tra lỗi:  python train_lm.py --out run_test --mb 2 --steps 200 --eval_every 100 --save_every 100
"""
import argparse, json, math, os, random, re, time
import torch
import torch.nn.functional as F
from adaptive_moe import set_k_cap, gather_aux_loss
from elastic_ctx import CtxLM, batch
from langcheck import read_wiki, read_file, split_docs, train_bpe, encode_docs, add_dropout

PRESETS = {  # d_ff = 2 * d như trong các thí nghiệm
    "small": dict(d=256, layers=6, experts=16, lr=1e-3),
    "medium": dict(d=384, layers=8, experts=32, lr=6e-4),
    "large": dict(d=512, layers=8, experts=32, lr=4e-4),
}


CTX = {}


def hf_pull(args, files):
    if not args.hf_repo:
        return
    from huggingface_hub import hf_hub_download
    for fn in files:
        if os.path.exists(os.path.join(args.out, fn)):
            continue
        try:
            hf_hub_download(repo_id=args.hf_repo, filename=fn, repo_type=args.hf_type, local_dir=args.out)
            print(f"[train] đã tải {fn} từ Hugging Face", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[train] không tải được {fn} từ Hugging Face ({str(e)[:70]})", flush=True)


def hf_push(args, files):
    if not args.hf_repo:
        return
    try:
        from huggingface_hub import HfApi, CommitOperationAdd
        api = HfApi()
        api.create_repo(repo_id=args.hf_repo, repo_type=args.hf_type, private=not args.hf_public, exist_ok=True)
        ops = [CommitOperationAdd(path_in_repo=fn, path_or_fileobj=os.path.join(args.out, fn))
               for fn in files if os.path.exists(os.path.join(args.out, fn))]
        if ops:   # một commit duy nhất cho mọi file, ít lượt ghi hơn việc upload từng file
            api.create_commit(repo_id=args.hf_repo, repo_type=args.hf_type, operations=ops, commit_message="checkpoint")
            CTX["pushes"] = CTX.get("pushes", 0) + 1
            if args.hf_squash and CTX["pushes"] % args.hf_squash == 0:   # xoá lịch sử cũ để không phình dung lượng (không hoàn tác được)
                api.super_squash_history(repo_id=args.hf_repo, repo_type=args.hf_type)
                print("[train] đã gộp lịch sử repo Hugging Face", flush=True)
        print(f"[train] đã đẩy {', '.join(files)} lên Hugging Face", flush=True)
    except Exception as e:  # noqa: BLE001
        print(f"[train] LỖI đẩy lên Hugging Face: {str(e)[:100]}", flush=True)


def atomic_save(obj, path):
    tmp = path + ".tmp"
    torch.save(obj, tmp)
    os.replace(tmp, path)


# ---------------------------------------------------------------- dữ liệu
def read_hf(spec, mb, seed, min_chars):
    """spec = 'ten/dataset:config:truong_van_ban', vd HuggingFaceFW/fineweb-2:vie_Latn:text (đọc streaming, không tải cả bộ)."""
    from datasets import load_dataset
    name, config, field = (spec.split(":") + ["text"])[:3]
    ds = load_dataset(name, config or None, split="train", streaming=True).shuffle(seed=seed, buffer_size=10000)
    docs, size = [], 0
    for ex in ds:
        t = (ex.get(field) or "").strip()
        if len(t) < min_chars:
            continue
        docs.append(t)
        size += len(t.encode("utf-8"))
        if size >= mb * 1_000_000:
            break
    print(f"[train] {spec}: {len(docs)} bài, {size / 1e6:.0f} MB", flush=True)
    return docs


def read_hf_files(args, prefix, mb):
    """Nạp mọi file <prefix>N.txt (vd pretrain/1.txt, pretrain/2.txt...) từ repo HF --hf_repo; cùng cách tách đoạn như --text_files."""
    from huggingface_hub import HfApi, hf_hub_download
    pat = re.compile("^" + re.escape(prefix) + r"(\d+)\.txt$")
    names = sorted((n for n in HfApi().list_repo_files(args.hf_repo, repo_type=args.hf_type) if pat.match(n)),
                   key=lambda n: int(pat.match(n).group(1)))
    docs, size = [], 0
    for n in names:
        p = hf_hub_download(args.hf_repo, n, repo_type=args.hf_type, local_dir=os.path.join(args.out, "hf_files"), force_download=True)
        for d in read_file(p, mb):
            docs.append(d)
            size += len(d.encode("utf-8"))
    print(f"[train] {prefix}*.txt trên HF: {len(names)} file, {len(docs)} đoạn, {size / 1e6:.1f} MB", flush=True)
    return docs


def parse_upsample(spec):
    """'yui=20,abc=5' -> {'yui': 20, 'abc': 5}: nhân bản bài train của ngôn ngữ nhỏ (chỉ train, không đụng val)."""
    out = {}
    for kv in (spec or "").split(","):
        if "=" in kv:
            k, v = kv.split("=", 1)
            out[k.strip()] = max(1, int(v))
    return out


def encode_chunked(tok, docs, chunk=5000):
    """Gọi encode_docs theo từng cụm bài rồi nối lại: kết quả y hệt nhưng không giữ cả corpus dưới dạng list Python (nguyên nhân OOM, mã thoát -9)."""
    parts = []
    for i in range(0, len(docs), chunk):
        parts.append(encode_docs(tok, docs[i:i + chunk]).int())
    return torch.cat(parts) if parts else torch.zeros(0, dtype=torch.int32)


def sample_docs(docs, max_chars):
    out, n = [], 0
    for d in docs:
        out.append(d)
        n += len(d)
        if n >= max_chars:
            break
    return out


def build_data(args):
    path = os.path.join(args.out, "data.pt")
    if os.path.exists(path):
        d = torch.load(path)
        print(f"[train] dùng lại dữ liệu đã token hoá: {path}", flush=True)
        return d["train"].long(), {l: v.long() for l, v in d["val"].items()}, d["vocab"], d["langs"]
    langs = args.langs.split(",")
    files = dict(kv.split("=", 1) for kv in args.text_files.split(",")) if args.text_files else {}
    tr_docs, va_docs = {}, {}
    ups = parse_upsample(args.upsample)
    hf_files = dict(kv.split("=", 1) for kv in (args.hf_files or "").split(",") if "=" in kv)   # lang -> thư mục trên repo HF
    extra = {}   # --hf_extra "news=repo:config:truong:MB,..." -> {lang: (spec, MB)}
    for kv in (args.hf_extra or "").split(","):
        if "=" in kv:
            name, v = kv.split("=", 1)
            parts = v.split(":")
            extra[name.strip()] = (":".join(parts[:3]), float(parts[3]) if len(parts) > 3 else args.hf_mb)
    for lang in langs:
        if lang == "web" and args.hf_dataset:
            docs = read_hf(args.hf_dataset, args.hf_mb, args.seed, args.min_chars)
        elif lang in hf_files:
            docs = read_hf_files(args, hf_files[lang], args.mb)
        elif lang in extra:
            docs = read_hf(extra[lang][0], extra[lang][1], args.seed, args.min_chars)
        elif lang in files:
            docs = read_file(files[lang], args.mb)
        else:
            docs = read_wiki(lang, args.mb, args.seed, args.min_chars)
        tr_docs[lang], va_docs[lang] = split_docs(docs, args.val_frac, args.seed)
        print(f"[train] {lang}: {len(tr_docs[lang])} bài train, {len(va_docs[lang])} bài val", flush=True)
        if lang in ups and ups[lang] > 1:
            tr_docs[lang] = tr_docs[lang] * ups[lang]
            print(f"[train] {lang}: nhân {ups[lang]} lần -> {len(tr_docs[lang])} bài train (val giữ nguyên)", flush=True)
    del docs
    tok_path = os.path.join(args.out, "tokenizer.json")
    if os.path.exists(tok_path) and not args.new_tokenizer:
        # QUAN TRỌNG: dùng lại tokenizer cũ để last.pt vẫn khớp. ByteLevel BPE không bao giờ OOV nên dùng cho dữ liệu mới vẫn được.
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(tok_path)
        print(f"[train] dùng lại tokenizer có sẵn ({tok.get_vocab_size()} token), không train BPE mới", flush=True)
    else:
        if os.path.exists(os.path.join(args.out, "last.pt")):
            raise SystemExit("[train] DỪNG: sắp train tokenizer mới nhưng --out đã có last.pt (train với tokenizer khác, id token sẽ lệch, "
                             "resume là hỏng model). Dùng --out mới, hoặc xoá last.pt/best.pt nếu muốn train lại từ đầu.")
        sample = [d for l in langs for d in sample_docs(tr_docs[l], int(args.bpe_mb * 1e6 / len(langs)))]
        print(f"[train] train BPE trên {len(sample)} bài mẫu (--bpe_mb {args.bpe_mb})", flush=True)
        tok = train_bpe(sample, args.vocab)
        del sample
        tok.save(tok_path)
    ids = {l: encode_chunked(tok, tr_docs[l]) for l in langs}
    del tr_docs
    if args.balance:
        n = min(len(v) for v in ids.values())
        ids = {l: v[:n] for l, v in ids.items()}
    train = torch.cat(list(ids.values()))
    del ids
    vals = {l: encode_chunked(tok, va_docs[l]) for l in langs}
    V = tok.get_vocab_size()
    atomic_save({"train": train, "val": vals, "vocab": V, "langs": langs}, path)
    CTX["data_built"] = True
    print(f"[train] {len(train)} token train, vocab {V}", flush=True)
    return train.long(), {l: v.long() for l, v in vals.items()}, V, langs


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
        hf_pull(args, ["config.json", "tokenizer.json", "best.pt"])
        return generate(args, device)
    hf_pull(args, ["config.json", "tokenizer.json", "data.pt", "last.pt", "best.pt"])
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
    if args.fast_moe:
        from fast_moe import enable_fast_moe
        print(f"[train] fast_moe: đã bật cho {enable_fast_moe(model, args.fast_waste)} layer MoE", flush=True)
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
    hf_push(args, ["config.json", "tokenizer.json"] + (["data.pt"] if step == 0 or CTX.get("data_built") else []))
    if step >= args.steps:
        print(f"[train] last.pt đang ở bước {step} >= --steps {args.steps}, tức là đã train xong ({'val tốt nhất %.4f ở bước %d' % best}). "
              f"Muốn train tiếp thì tăng --steps (vd {2 * args.steps}); lịch learning rate sẽ co giãn theo --steps mới.", flush=True)
        return
    ks = [int(x) for x in args.eval_ks.split(",")]
    g = torch.Generator().manual_seed(args.seed + step)
    t0, t_log, tok_log = time.time(), time.time(), 0

    def save_last():
        atomic_save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict() if amp else None,
                     "step": step, "best": best, "bad": bad}, last)

    def finish():
        save_last()
        hf_push(args, ["last.pt", "best.pt"])
    CTX["on_exit"] = finish
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
                total_loss = loss + args.aux * gather_aux_loss(model) if args.aux > 0 else loss
                scaler.scale(total_loss / args.accum).backward()
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
            ups_ = parse_upsample(args.upsample)
            sel_l = [l for l in res[args.select_k] if l not in ups_] or list(res[args.select_k])   # chọn best.pt theo vi/web, không theo tập nhỏ (nhiễu)
            sel = sum(res[args.select_k][l] for l in sel_l) / len(sel_l)
            print(f"[train] ĐÁNH GIÁ bước {step}: {show}", flush=True)
            if args.fast_moe:
                from fast_moe import load_report
                lr_ = load_report(model)
                if lr_:
                    print("[train] tải expert (đông nhất / trung bình) theo layer: " + " ".join(f"{r:.1f}({p})" for r, p in lr_), flush=True)
            if sel < best[0]:
                best, bad = (sel, step), 0
                atomic_save({"model": model.state_dict(), "step": step, "val": res}, os.path.join(args.out, "best.pt"))
                print(f"[train] val tốt nhất mới {sel:.4f}, đã lưu best.pt", flush=True)
            else:
                bad += 1
        if step % args.save_every == 0 or step == args.steps:
            save_last()
        if args.hf_repo and step % args.hf_every == 0:
            hf_push(args, ["last.pt", "best.pt"])
        if args.patience and bad >= args.patience:
            print(f"[train] dừng sớm: {bad} lần đánh giá liên tiếp không cải thiện", flush=True)
            finish()
            return
        if args.max_minutes and (time.time() - t0) / 60 > args.max_minutes:
            finish()
            print(f"[train] hết {args.max_minutes} phút, đã lưu last.pt; chạy lại cùng lệnh để tiếp tục", flush=True)
            return
    finish()
    print(f"[train] xong. val tốt nhất {best[0]:.4f} ở bước {best[1]}; dùng best.pt để suy luận (--generate)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="thư mục lưu checkpoint (nên nằm trên Google Drive)")
    ap.add_argument("--langs", default="vi")
    ap.add_argument("--text_files", default=None)
    ap.add_argument("--mb", type=float, default=100, help="MB văn bản mỗi ngôn ngữ")
    ap.add_argument("--min_chars", type=int, default=1000)
    ap.add_argument("--hf_dataset", default=None, help="nguồn web thêm, dùng cùng --langs vi,web; vd HuggingFaceFW/fineweb-2:vie_Latn:text")
    ap.add_argument("--hf_mb", type=float, default=300)
    ap.add_argument("--hf_files", default="", help='nạp file N.txt trên repo HF (--hf_repo) làm một "ngôn ngữ" mới: "gen=pretrain/" (cùng make_data.py pretrain)')
    ap.add_argument("--hf_extra", default="", help='thêm nguồn HF, mỗi nguồn là một "ngôn ngữ" mới trong --langs: "tin=repo:config:truong:MB,phude=repo::text:200"')
    ap.add_argument("--val_frac", type=float, default=0.02)
    ap.add_argument("--upsample", default="", help="nhân bản bài train của ngôn ngữ nhỏ, vd yui=20 (dùng với --text_files yui=duong_dan)")
    ap.add_argument("--bpe_mb", type=float, default=150, help="chỉ lấy tối đa ngần này MB văn bản để train BPE (chống hết RAM)")
    ap.add_argument("--new_tokenizer", action="store_true", help="ép train tokenizer mới (chỉ dùng khi --out mới, không có last.pt)")
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
    ap.add_argument("--fast_moe", action="store_true", help="bmm thay vòng lặp expert khi tải đủ cân bằng (fast_moe.py)")
    ap.add_argument("--fast_waste", type=float, default=3.0)
    ap.add_argument("--aux", type=float, default=0.0, help="hệ số nhân cho loss phụ của router (cân bằng tải, z-loss); 0 = tắt")
    ap.add_argument("--hf_repo", default=None, help="vd BachDaThan/TrainAI: đồng bộ checkpoint với Hugging Face")
    ap.add_argument("--hf_type", default="dataset")
    ap.add_argument("--hf_every", type=int, default=4000)
    ap.add_argument("--hf_public", action="store_true", help="nếu phải tạo repo mới thì tạo công khai (mặc định riêng tư)")
    ap.add_argument("--hf_squash", type=int, default=0, help="sau mỗi N lần đẩy thì gộp lịch sử repo (0 = tắt)")
    ap.add_argument("--generate", default=None, help="chế độ sinh văn bản từ best.pt với prompt này")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--max_new", type=int, default=100)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    try:
        main(ap.parse_args())
    except KeyboardInterrupt:
        print("\n[train] bị ngắt (Ctrl+C): đang lưu last.pt và đẩy lên Hugging Face...", flush=True)
        if CTX.get("on_exit"):
            CTX["on_exit"]()
        raise SystemExit(130)
