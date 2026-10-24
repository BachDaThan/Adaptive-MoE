"""
langcheck.py - đa ngôn ngữ với MỘT tokenizer chung (ByteLevel BPE, không bao giờ OOV) và router có tự chuyên môn hoá theo ngôn ngữ không?

Phần 1 (không cần GPU): đếm token trên 100 ký tự và trên mỗi từ cho từng ngôn ngữ, so
    - BPE chỉ train trên tiếng Việt (vocab --vocab)
    - BPE chung train trên tất cả --langs (cùng vocab)
    - tokenizer tham chiếu công khai (--ref_tokenizer, tải từ HF; bỏ qua nếu không tải được)
Phần 2: train LM nhỏ trên dữ liệu trộn đều các ngôn ngữ (mỗi ngôn ngữ cùng số token), rồi đo THEO TỪNG NGÔN NGỮ:
    loss (nats/token) và bit/ký tự, avg_k của router (mode adaptive), mức dùng từng expert (mode fixed),
    và độ khác nhau giữa các ngôn ngữ (Jensen-Shannon) so với mức nhiễu của cùng một ngôn ngữ chia đôi.

Chọn bước dừng sớm trên nửa A của val từng ngôn ngữ, báo trên nửa B (như regcheck.py).
Lưu ý: tokenizer ở đây là ByteLevel BPE của thư viện `tokenizers`, KHÔNG phải load_bpe của headroom.py,
nên loss không so trực tiếp với các bảng trong README. Muốn so giữa ngôn ngữ thì dùng bit/ký tự, không dùng loss/token.

Cần: pip install torch datasets tokenizers (và transformers nếu muốn tokenizer tham chiếu);
      adaptive_moe.py 0.3.1, elastic_ctx.py cạnh file này.

python langcheck.py --langs vi,en --mb 8 --mode both
python langcheck.py --langs vi,en,zh --mb 8 --mode both --seed 1
python langcheck.py --langs vi,en --only_tok                      # chỉ phần 1
python langcheck.py --text_files vi=vi.txt,en=en.txt --only_tok   # dùng file text có sẵn thay vì Wikipedia
"""
import argparse, copy, json, math, random
import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- dữ liệu
def read_wiki(lang, mb, seed):
    from datasets import load_dataset
    ds = load_dataset("wikimedia/wikipedia", f"20231101.{lang}", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=5000)
    docs, size = [], 0
    for ex in ds:
        t = ex["text"].strip()
        if len(t) < 200:
            continue
        docs.append(t)
        size += len(t.encode("utf-8"))
        if size >= mb * 1_000_000:
            break
    return docs


def read_file(path, mb):
    txt = open(path, encoding="utf-8", errors="ignore").read()
    docs = [p.strip() for p in txt.split("\n\n") if len(p.strip()) >= 200] or [p for p in txt.split("\n") if p.strip()]
    out, size = [], 0
    for d in docs:
        out.append(d)
        size += len(d.encode("utf-8"))
        if size >= mb * 1_000_000:
            break
    return out


def split_docs(docs, val_frac, seed):
    rng = random.Random(seed)
    idx = list(range(len(docs)))
    rng.shuffle(idx)
    nv = max(2, int(len(docs) * val_frac))
    val = [docs[i] for i in idx[:nv]]
    tr = [docs[i] for i in idx[nv:]]
    return tr, val


# ---------------------------------------------------------------- tokenizer
def train_bpe(texts, vocab):
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    tr = trainers.BpeTrainer(vocab_size=vocab, special_tokens=["<eos>"],
                             initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(texts, tr)
    return tok


def encode_docs(tok, docs):
    eos = tok.token_to_id("<eos>")
    ids = []
    for e in tok.encode_batch(docs):
        ids.extend(e.ids)
        ids.append(eos)
    return torch.tensor(ids, dtype=torch.long)


def tok_stats(count_fn, docs, lang):
    """token / 100 ký tự và token / từ (từ = tách theo khoảng trắng; bỏ qua với tiếng Trung/Nhật)."""
    chars = sum(len(d) for d in docs)
    words = sum(len(d.split()) for d in docs)
    n = count_fn(docs)
    per_word = n / words if lang not in ("zh", "ja", "ko", "th") else float("nan")
    return 100.0 * n / chars, per_word


def part1(args, tr_docs, va_docs, joint, vi_only):
    ref = None
    if args.ref_tokenizer:
        try:
            from transformers import AutoTokenizer
            ref = AutoTokenizer.from_pretrained(args.ref_tokenizer)
        except Exception as e:  # noqa: BLE001
            print(f"[langcheck] bỏ qua tokenizer tham chiếu ({args.ref_tokenizer}): {str(e)[:80]}")
    cols = [("BPE Việt (%dk)" % (args.vocab // 1000), lambda ds: sum(len(e.ids) for e in vi_only.encode_batch(ds))),
            ("BPE chung (%dk)" % (args.vocab // 1000), lambda ds: sum(len(e.ids) for e in joint.encode_batch(ds)))]
    if ref is not None:
        cols.append((args.ref_tokenizer.split("/")[-1], lambda ds: sum(len(x) for x in ref(ds, add_special_tokens=False)["input_ids"])))
    print("\n== Phần 1: token / 100 ký tự (và token / từ), trên val ==")
    print("ngôn ngữ".ljust(10) + "".join(c[0].rjust(24) for c in cols))
    res = {}
    for lang, docs in va_docs.items():
        row = []
        for name, fn in cols:
            c100, pw = tok_stats(fn, docs, lang)
            row.append((c100, pw))
            res[(lang, name)] = (c100, pw)
        print(lang.ljust(10) + "".join((f"{c:7.1f}/100ch" + (f" {w:5.2f}/từ" if w == w else "      -    ")).rjust(24) for c, w in row))
    return res


# ---------------------------------------------------------------- model
def add_dropout(model, p):
    model.emb.register_forward_hook(lambda m, i, o: F.dropout(o, p, m.training))

    def hook(m, i, o):
        y, aux, info = o
        return (F.dropout(y, p, m.training), aux, info)
    for b in model.blocks:
        b.moe.register_forward_hook(hook)


def expert_list(moe, n_experts):
    for _, ch in moe.named_children():
        if isinstance(ch, torch.nn.ModuleList) and len(ch) == n_experts:
            return ch
    return None


@torch.no_grad()
def measure(model, toks, args, device, cfg_cap):
    """Chạy trên các cửa sổ liên tiếp của `toks`; trả loss, avg_k theo layer, đếm token từng expert theo layer."""
    model.eval()
    L, B = args.L, args.B
    nw = min(args.eval_batches * B, (len(toks) - 1) // L)
    if nw < 1:
        return None
    xs = torch.stack([toks[i * L:(i + 1) * L] for i in range(nw)])
    ys = torch.stack([toks[i * L + 1:(i + 1) * L + 1] for i in range(nw)])
    nl = len(model.blocks)
    kk = [[0.0, 0] for _ in range(nl)]
    cnt = torch.zeros(nl, args.experts)
    seen = [0] * nl
    hooks = []
    keys_seen = {}

    def mk_moe(l):
        def h(m, i, o):
            info = o[2] if isinstance(o, tuple) and len(o) > 2 else None
            if isinstance(info, dict):
                keys_seen[l] = sorted(info.keys())
                kp = info.get("k_per_token")
                if kp is not None:
                    kk[l][0] += float(kp.float().sum())
                    kk[l][1] += kp.numel()
            seen[l] += i[0].reshape(-1, i[0].shape[-1]).shape[0]
        return h

    def mk_exp(l, e):
        def h(m, i, o):
            cnt[l, e] += i[0].reshape(-1, i[0].shape[-1]).shape[0]
        return h

    for l, b in enumerate(model.blocks):
        b.moe.k_cap = cfg_cap
        hooks.append(b.moe.register_forward_hook(mk_moe(l)))
        ex = expert_list(b.moe, args.experts)
        if ex is not None:
            for e, m in enumerate(ex):
                hooks.append(m.register_forward_hook(mk_exp(l, e)))
    tot, n = 0.0, 0
    for s in range(0, nw, B):
        x, y = xs[s:s + B].to(device), ys[s:s + B].to(device)
        lg, _ = model(x, None)
        tot += F.cross_entropy(lg.float().view(-1, lg.shape[-1]), y.reshape(-1), reduction="sum").item()
        n += y.numel()
    for h in hooks:
        h.remove()
    dense = bool((cnt.sum(1) > 0).all() and all(abs(float(cnt[l].min()) - seen[l]) < 1 for l in range(nl)))
    return {"loss": tot / n, "n": n, "avg_k": [a / b if b else float("nan") for a, b in kk],
            "usage": cnt, "dense": dense, "keys": keys_seen.get(0)}


def js(p, q):
    p = p / p.sum().clamp_min(1e-9)
    q = q / q.sum().clamp_min(1e-9)
    m = 0.5 * (p + q)

    def kl(a, b):
        mask = a > 0
        return float((a[mask] * (a[mask] / b[mask].clamp_min(1e-12)).log()).sum())
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def run_mode(mode, train, vals, lang_stats, args, vocab, device):
    from adaptive_moe import set_k_cap  # noqa: F401  (đảm bảo đúng bản adaptive_moe)
    from elastic_ctx import CtxLM, batch
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    g = torch.Generator().manual_seed(1 + args.seed)
    top_p = args.top_p_fixed if mode == "fixed" else args.top_p_adaptive
    model = CtxLM(vocab=vocab, d=args.d, n_layers=args.layers, n_experts=args.experts, d_ff=2 * args.d,
                  min_k=1, max_k=8, top_p=top_p, n_shared=1, compute_coef=0.0).to(device)
    if args.drop > 0:
        add_dropout(model, args.drop)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    cap = args.k if mode == "fixed" else None
    best, best_state, hist = (1e9, 0), None, []
    for step in range(args.steps + 1):
        model.train()
        x, y = batch(train, args.B, args.L, device, g)
        for b in model.blocks:
            b.moe.k_cap = cap
        lg, _ = model(x, None)
        loss = F.cross_entropy(lg.float().view(-1, lg.shape[-1]), y.view(-1))
        opt.zero_grad(); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step > 0 and (step % args.eval_every == 0 or step == args.steps):
            a = [measure(model, vals[l][0], args, device, cap)["loss"] for l in vals]
            ma = sum(a) / len(a)
            hist.append((step, ma))
            if ma < best[0]:
                best = (ma, step)
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            print(f"[langcheck] {mode} bước {step}: loss nửa A (trung bình các ngôn ngữ) {ma:.4f}", flush=True)
    model.load_state_dict(best_state)
    print(f"\n== Phần 2 ({mode}): bước dừng sớm {best[1]} (chọn trên nửa A), báo trên nửa B ==")
    res = {}
    for lang, (vA, vB) in vals.items():
        rB = measure(model, vB, args, device, cap)
        rA = measure(model, vA, args, device, cap)
        res[lang] = (rA, rB)
    print("ngôn ngữ".ljust(10) + "loss/token".rjust(12) + "bit/ký tự".rjust(12) + ("   avg_k theo layer (trung bình)" if mode == "adaptive" else ""))
    out = {}
    for lang, (rA, rB) in res.items():
        tpc = lang_stats[lang]                       # token mỗi ký tự trên val
        bpc = rB["loss"] * tpc / math.log(2)
        line = lang.ljust(10) + f"{rB['loss']:12.4f}" + f"{bpc:12.4f}"
        if mode == "adaptive":
            ks = rB["avg_k"]
            line += "   " + " ".join(f"{k:.2f}" for k in ks) + f"  ({sum(ks) / len(ks):.2f})"
        print(line)
        out[lang] = {"loss": rB["loss"], "bpc": bpc, "avg_k": rB["avg_k"]}
    first = next(iter(res.values()))[1]
    if first["keys"] is not None:
        print(f"[langcheck] khoá trong info của MoE: {first['keys']}")
    if mode == "fixed":
        if first["usage"].sum() == 0:
            print("[langcheck] không đếm được expert nào (không tìm thấy ModuleList expert); gửi mình danh sách khoá ở trên để chỉnh.")
        elif first["dense"]:
            print("[langcheck] expert được chạy dày (mọi expert nhận mọi token), không đếm được mức dùng; gửi mình info keys để chỉnh.")
        else:
            langs = list(res)
            nl = args.layers
            print("\nĐộ khác nhau mức dùng expert (Jensen-Shannon, nat; càng lớn càng chuyên môn hoá theo ngôn ngữ):")
            print("layer".ljust(8) + "".join(f"{a}-{b}".rjust(10) for i, a in enumerate(langs) for b in langs[i + 1:]) +
                  "".join(f"{a}:A-B".rjust(10) for a in langs))
            sums = {}
            for l in range(nl):
                row = f"{l}".ljust(8)
                vals_row = []
                for i, a in enumerate(langs):
                    for b in langs[i + 1:]:
                        v = js(res[a][1]["usage"][l], res[b][1]["usage"][l]); vals_row.append(v); row += f"{v:10.4f}"
                for a in langs:
                    v = js(res[a][0]["usage"][l], res[a][1]["usage"][l]); vals_row.append(v); row += f"{v:10.4f}"
                print(row)
            print("(cột 'ngôn ngữ:A-B' là mức nhiễu: cùng một ngôn ngữ, hai nửa val khác nhau. Chỉ tin chuyên môn hoá khi cột giữa các ngôn ngữ lớn hơn hẳn mức nhiễu.)")
            top = {}
            for lang in langs:
                u = res[lang][1]["usage"][nl - 1]
                u = u / u.sum()
                top[lang] = [(int(i), round(float(u[i]), 3)) for i in u.argsort(descending=True)[:4]]
            print(f"4 expert dùng nhiều nhất ở layer cuối: {top}")
            out["_top_last_layer"] = top
    return out


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    langs = args.langs.split(",")
    files = dict(kv.split("=", 1) for kv in args.text_files.split(",")) if args.text_files else {}
    tr_docs, va_docs = {}, {}
    for lang in langs:
        docs = read_file(files[lang], args.mb) if lang in files else read_wiki(lang, args.mb, args.seed)
        tr_docs[lang], va_docs[lang] = split_docs(docs, args.val_frac, args.seed)
        print(f"[langcheck] {lang}: {len(tr_docs[lang])} bài train, {len(va_docs[lang])} bài val", flush=True)
    joint = train_bpe([d for l in langs for d in tr_docs[l]], args.vocab)
    vi_only = train_bpe(tr_docs.get("vi", tr_docs[langs[0]]), args.vocab)
    p1 = part1(args, tr_docs, va_docs, joint, vi_only)
    if args.only_tok:
        return
    ids_tr = {l: encode_docs(joint, tr_docs[l]) for l in langs}
    n = min(len(v) for v in ids_tr.values())
    train = torch.cat([v[:n] for v in ids_tr.values()])
    print(f"\n[langcheck] train: mỗi ngôn ngữ {n} token (cắt cho bằng nhau), tổng {len(train)}; vocab {joint.get_vocab_size()}")
    vals, lang_stats = {}, {}
    for l in langs:
        v = encode_docs(joint, va_docs[l])
        h = len(v) // 2
        vals[l] = (v[:h], v[h:])
        lang_stats[l] = len(v) / sum(len(d) for d in va_docs[l])   # token / ký tự
    modes = ["fixed", "adaptive"] if args.mode == "both" else [args.mode]
    allres = {}
    for m in modes:
        allres[m] = run_mode(m, train, vals, lang_stats, args, joint.get_vocab_size(), device)
    if args.out:
        json.dump({"args": vars(args), "tok": {f"{k[0]}|{k[1]}": v for k, v in p1.items()}, "res": allres},
                  open(args.out, "w", encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--langs", default="vi,en")
    ap.add_argument("--text_files", default=None, help="vi=duong_dan.txt,en=duong_dan.txt thay cho Wikipedia")
    ap.add_argument("--mb", type=float, default=8)
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--vocab", type=int, default=16000)
    ap.add_argument("--ref_tokenizer", default="Qwen/Qwen2.5-0.5B")
    ap.add_argument("--only_tok", action="store_true")
    ap.add_argument("--mode", choices=["fixed", "adaptive", "both"], default="both")
    ap.add_argument("--k", type=int, default=2, help="k cố định cho mode fixed")
    ap.add_argument("--top_p_fixed", type=float, default=0.999)
    ap.add_argument("--top_p_adaptive", type=float, default=0.7)
    ap.add_argument("--drop", type=float, default=0.2)
    ap.add_argument("--L", type=int, default=128)
    ap.add_argument("--B", type=int, default=32)
    ap.add_argument("--d", type=int, default=128)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--experts", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--eval_every", type=int, default=500)
    ap.add_argument("--eval_batches", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    main(ap.parse_args())
