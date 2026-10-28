"""
train_sft.py - fine-tune hội thoại cho model đã pretrain bằng train_lm.py. TỰ QUÉT repo Hugging Face tìm các file
1.txt, 2.txt, 3.txt, ... (số nguyên tăng dần, không giới hạn), tải về, gộp lại và train. Bạn chỉ cần up thêm file mới
(vd 7.txt) rồi chạy lại đúng lệnh cũ; nếu không có gì mới (cùng file, cùng nội dung, cùng model gốc) nó tự bỏ qua.

  - Mỗi lần chạy bắt đầu từ model gốc (best.pt của pretrain, hoặc last.pt nếu chưa có), train trên TẤT CẢ file số;
    kết quả lưu thành sft_best.pt (+ sft_state.json) và đẩy lên cùng repo. best.pt gốc KHÔNG bị ghi đè.
  - Chỉ tính loss trên câu trả lời của Yui; giữ ~5% hội thoại làm val để chọn epoch tốt nhất, dừng sớm khi val không giảm.
  - Dùng lại tokenizer.json của pretrain (không thêm token đặc biệt), nên checkpoint luôn khớp.
  - Định dạng file xem sft_format.py.

Kaggle (cần HF_TOKEN trong Secrets, quyền write, vì repo là private):
    import os; from kaggle_secrets import UserSecretsClient
    os.environ["HF_TOKEN"] = UserSecretsClient().get_secret("HF_TOKEN")
    !python train_sft.py --out /kaggle/working/sft_run --hf_repo BachDaThan/TrainAI-web
Chạy với file có sẵn trên máy (không cần HF):   python train_sft.py --out run --local thu_muc_chua_1.txt_2.txt --no_push
Cần adaptive_moe.py, elastic_ctx.py, langcheck.py, train_lm.py, sft_format.py cạnh file này.
"""
import argparse, hashlib, json, math, os, random, re, time
import torch
import torch.nn.functional as F
from adaptive_moe import set_k_cap
from langcheck import add_dropout
from train_lm import build_model, set_cap, hf_pull, hf_push, atomic_save
from sft_format import load_convs, encode_pairs, parse_text

STATE = "sft_state.json"


def sha(b):
    return hashlib.sha256(b).hexdigest()[:16]


def check_token(args):
    if args.local or not args.hf_repo:
        return
    tok = os.environ.get("HF_TOKEN")
    if not tok:
        try:
            from huggingface_hub import get_token
            tok = get_token()
        except Exception:  # noqa: BLE001
            tok = None
    if not tok:
        print("[sft] CẢNH BÁO: chưa có HF_TOKEN. Repo private sẽ báo 401 Unauthorized. "
              "Kaggle: đặt os.environ['HF_TOKEN'] = UserSecretsClient().get_secret('HF_TOKEN') trước khi chạy.", flush=True)


def resolve_prefix(args):
    """Tự vào Fine-tune/<cỡ>/ theo cỡ model (tính từ config.json). Không có thư mục đó thì giữ cách cũ (gốc repo hoặc --sft_prefix)."""
    if args.sft_prefix or args.local or args.size_tag == "off":
        return
    from data_layout import params_from_arch, size_tag, finetune_prefix, has_numbered
    tag = args.size_tag
    if tag == "auto":
        hf_pull(args, ["config.json"])
        cp = os.path.join(args.out, "config.json")
        if not os.path.exists(cp):
            print("[sft] chưa có config.json nên chưa biết cỡ model; dùng gốc repo như cũ", flush=True)
            return
        cfg = json.load(open(cp, encoding="utf-8"))
        tag = size_tag(params_from_arch(cfg["vocab"], cfg["d"], cfg["layers"], cfg["experts"]))
    from huggingface_hub import HfApi
    pre = finetune_prefix(tag)
    if has_numbered(HfApi().list_repo_files(args.hf_repo, repo_type=args.hf_type), pre):
        args.sft_prefix = pre
        print(f"[sft] model cỡ {tag}: lấy data trong {pre}", flush=True)
    else:
        print(f"[sft] không có thư mục {pre} trên repo: dùng gốc repo như cũ", flush=True)


def discover(args):
    """Trả {tên file: bytes} của mọi file N.txt (N số nguyên), sắp theo N."""
    pat = re.compile(r"^" + re.escape(args.sft_prefix) + r"(\d+)\.txt$")
    data = {}
    if args.local:
        names = [n for n in os.listdir(args.local) if pat.match(n)]
        for n in sorted(names, key=lambda s: int(pat.match(s).group(1))):
            data[n] = open(os.path.join(args.local, n), "rb").read()
    else:
        from huggingface_hub import HfApi, hf_hub_download
        names = [n for n in HfApi().list_repo_files(args.hf_repo, repo_type=args.hf_type) if pat.match(n)]
        d = os.path.join(args.out, "sft_data")
        for n in sorted(names, key=lambda s: int(pat.match(s).group(1))):
            p = hf_hub_download(repo_id=args.hf_repo, filename=n, repo_type=args.hf_type, local_dir=d, force_download=True)
            data[n] = open(p, "rb").read()
    # file dữ liệu thêm (không đánh số), vd chat_chat.jsonl: cùng định dạng JSON hội thoại như N.txt
    for n in [x.strip() for x in args.extra.split(",") if x.strip()]:
        try:
            if args.local:
                data[n] = open(os.path.join(args.local, n), "rb").read()
            else:
                from huggingface_hub import hf_hub_download
                data[n] = open(hf_hub_download(repo_id=args.hf_repo, filename=n, repo_type=args.hf_type,
                                               local_dir=os.path.join(args.out, "sft_data"), force_download=True), "rb").read()
        except Exception as e:  # noqa: BLE001
            raise SystemExit(f"[sft] không lấy được file --extra '{n}': {str(e)[:100]}")
    return data


def pad_batch(items, eos, device):
    T = max(len(i[0]) for i in items)
    ids = torch.full((len(items), T), eos, dtype=torch.long)
    msk = torch.zeros((len(items), T), dtype=torch.float)
    for r, (a, m) in enumerate(items):
        ids[r, :len(a)] = torch.tensor(a)
        msk[r, :len(m)] = torch.tensor(m, dtype=torch.float)
    return ids[:, :-1].to(device), ids[:, 1:].to(device), msk[:, 1:].to(device)


def masked_loss(model, x, y, m, amp):
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        lg, _ = model(x, None)
    ce = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="none").view_as(y)
    return (ce * m).sum(), m.sum()


@torch.no_grad()
def evaluate(model, val, eos, args, device, amp):
    model.eval()
    res = {}
    for k in (1, 2):
        set_cap(model, k)
        tot, n = 0.0, 0.0
        for s in range(0, len(val), args.B):
            x, y, m = pad_batch(val[s:s + args.B], eos, device)
            a, b = masked_loss(model, x, y, m, amp)
            tot += a.item(); n += b.item()
        res[k] = tot / max(n, 1.0)
    return res


def lr_at(step, total, args):
    w = max(1, min(args.warmup, total // 5))
    if step < w:
        return args.lr * (step + 1) / w
    t = (step - w) / max(1, total - w)
    return args.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, t))))


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.out, exist_ok=True)
    check_token(args)

    # 1. dữ liệu: quét + tải mọi file N.txt
    resolve_prefix(args)
    data = discover(args)
    if not data:
        raise SystemExit("[sft] không thấy file nào tên dạng 1.txt, 2.txt, ... (kiểm tra --hf_repo / --sft_prefix / --local)")
    files = {n: sha(b) for n, b in data.items()}
    print(f"[sft] tìm thấy {len(files)} file: {', '.join(files)}", flush=True)

    # 2. model gốc + tokenizer; bỏ qua nếu không có gì mới
    wanted = ["config.json", "tokenizer.json", "best.pt", STATE, "sft_best.pt"]
    if not args.local:
        hf_pull(args, wanted)
    base_name = "best.pt" if (args.base == "best" and os.path.exists(os.path.join(args.out, "best.pt"))) else "last.pt"
    if not os.path.exists(os.path.join(args.out, base_name)) and not args.local:
        hf_pull(args, ["last.pt"])
    base_path = os.path.join(args.out, base_name)
    if not os.path.exists(base_path):
        raise SystemExit(f"[sft] không có {base_name} trong {args.out} (chưa pretrain, hoặc chưa tải được từ HF)")
    ck = torch.load(base_path, map_location=device)
    base_step = int(ck.get("step", -1))
    old = None
    sp = os.path.join(args.out, STATE)
    if os.path.exists(sp) and os.path.exists(os.path.join(args.out, "sft_best.pt")):
        try:
            old = json.load(open(sp, encoding="utf-8"))
        except Exception:  # noqa: BLE001
            old = None
    if old and old.get("files") == files and old.get("base_step") == base_step and not args.force:
        print(f"[sft] không có dữ liệu mới (cùng {len(files)} file, cùng model gốc bước {base_step}); sft_best.pt đã sẵn. "
              f"Thêm --force để train lại.", flush=True)
        return

    cfg = json.load(open(os.path.join(args.out, "config.json"), encoding="utf-8"))
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(args.out, "tokenizer.json"))
    eos = tok.token_to_id("<eos>")
    L = cfg["L"]

    # 3. mã hoá
    for n, b in data.items():
        if n in [x.strip() for x in args.extra.split(",")] or not re.match(r"^\d+\.txt$", os.path.basename(n)):
            cs, bd = parse_text(b.decode("utf-8", errors="ignore"))
            print(f"[sft] {n}: đọc được {len(cs)} hội thoại, {bd} bản ghi không đọc được"
                  f"{'  <-- 0 hội thoại: định dạng file này không phải JSON hội thoại' if not cs else ''}", flush=True)
    convs, st = load_convs({n: b.decode("utf-8", errors="ignore") for n, b in data.items()}, args.max_chars)
    exs, skipped = [], 0
    for c in convs:
        e = encode_pairs(tok, c, eos, L + 1)
        if e is None:
            skipped += 1
        else:
            exs.append((e, c))
    print(f"[sft] {len(convs)} hội thoại hợp lệ (bỏ: {st['hỏng']} hỏng, {st['quá dài']} câu quá dài, {st['trùng']} trùng, "
          f"{skipped} vượt {L} token); trung bình {sum(len(e[0][0]) for e in exs) / max(1, len(exs)):.0f} token", flush=True)
    if len(exs) < 8:
        raise SystemExit("[sft] quá ít hội thoại (<8) để train; thêm dữ liệu.")
    rng = random.Random(args.seed)
    order = list(range(len(exs)))
    rng.shuffle(order)
    nv = min(max(4, int(len(exs) * args.val_frac)), len(exs) // 2)
    val_idx, tr_idx = order[:nv], order[nv:]
    val = [exs[i][0] for i in val_idx]
    train = [exs[i][0] for i in tr_idx]
    print(f"[sft] train {len(train)} / val {len(val)}", flush=True)

    # 4. model
    random.seed(args.seed); torch.manual_seed(args.seed)
    model = build_model(cfg, device)
    model.load_state_dict(ck["model"])
    del ck
    print(f"[sft] model gốc: {base_name} (bước {base_step})", flush=True)
    if args.drop > 0:
        add_dropout(model, args.drop)
    amp = (not args.no_amp) and device == "cuda"
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    spe = math.ceil(len(train) / args.B)
    total = spe * args.epochs
    g = torch.Generator().manual_seed(args.seed)
    r0 = evaluate(model, val, eos, args, device, amp)
    print(f"[sft] val trước khi fine-tune: k=1 {r0[1]:.4f} k=2 {r0[2]:.4f}", flush=True)

    best, bad, step, t0 = (float("inf"), 0), 0, 0, time.time()
    best_sd = None
    for ep in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(train), generator=g).tolist()
        run, nb = 0.0, 0
        for s in range(0, len(perm), args.B):
            for gr in opt.param_groups:
                gr["lr"] = lr_at(step, total, args)
            x, y, m = pad_batch([train[i] for i in perm[s:s + args.B]], eos, device)
            f = random.choice((0.0, 0.25, 0.5, 0.75, 1.0))   # k elastic như lúc pretrain
            set_k_cap(model, 1 + (torch.rand(x.numel(), device=device) < f).long())
            a, b = masked_loss(model, x, y, m, amp)
            loss = a / b.clamp(min=1.0)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt); scaler.update()
            opt.zero_grad(set_to_none=True)
            step += 1; run += loss.item(); nb += 1
        res = evaluate(model, val, eos, args, device, amp)
        sel = res[2]
        print(f"[sft] epoch {ep}/{args.epochs} train {run / nb:.4f} val k=1 {res[1]:.4f} k=2 {res[2]:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)
        if sel < best[0]:
            best, bad = (sel, ep), 0
            best_sd = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            atomic_save({"model": best_sd, "step": step, "val": res, "base_step": base_step, "files": files},
                        os.path.join(args.out, "sft_best.pt"))
        else:
            bad += 1
            if args.patience and bad >= args.patience:
                print(f"[sft] dừng sớm: {bad} epoch liên tiếp val không giảm", flush=True)
                break
    print(f"[sft] val tốt nhất {best[0]:.4f} ở epoch {best[1]} (trước fine-tune {r0[2]:.4f})", flush=True)
    json.dump({"files": files, "base_step": base_step, "n_train": len(train), "n_val": len(val), "best_val": best[0],
               "best_epoch": best[1], "format": "User: ...\\nYui: ...<eos>"},
              open(sp, "w", encoding="utf-8"), ensure_ascii=False, indent=1)

    # 5. thử vài câu bằng model tốt nhất
    try:
        from chat import stream_chat
        model.load_state_dict(best_sd)
        model.eval()
        print("[sft] ---- thử vài câu (k=2) ----", flush=True)
        for e, c in [exs[i] for i in val_idx[:3]]:
            hist, (u, a) = c[:-1], c[-1]
            out = ""
            for out in stream_chat(model, tok, cfg, device, hist, u, k=2, max_new=60, temp=0.7, top_k=30, rep=1.1):
                pass
            print(f"  User: {u}\n  Yui (model): {out}\n  Yui (mẫu):   {a}\n", flush=True)
    except Exception as ex:  # noqa: BLE001
        print(f"[sft] bỏ qua bước thử câu ({str(ex)[:80]})", flush=True)

    if not args.no_push and not args.local:
        hf_push(args, ["sft_best.pt", STATE])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--hf_repo", default="BachDaThan/TrainAI-web")
    ap.add_argument("--hf_type", default="dataset")
    ap.add_argument("--hf_public", action="store_true")
    ap.add_argument("--hf_squash", type=int, default=0)
    ap.add_argument("--sft_prefix", default="", help="thư mục chứa 1.txt, 2.txt... trong repo (vd 'sft/'); mặc định là gốc repo")
    ap.add_argument("--size_tag", default="auto", help="tên thư mục cỡ model trong Fine-tune/, vd 50M. auto = tính từ config.json; off = tắt, dùng --sft_prefix/gốc repo")
    ap.add_argument("--extra", default="", help="thêm file dữ liệu JSON hội thoại không đánh số, cách nhau dấu phẩy (vd chat_chat.jsonl,yui_chat.jsonl)")
    ap.add_argument("--local", default=None, help="đọc N.txt từ thư mục này thay vì Hugging Face")
    ap.add_argument("--no_push", action="store_true")
    ap.add_argument("--base", choices=["best", "last"], default="best")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--patience", type=int, default=3)
    ap.add_argument("--max_chars", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--force", action="store_true")
    main(ap.parse_args())
