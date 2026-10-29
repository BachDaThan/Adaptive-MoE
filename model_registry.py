"""
model_registry.py - đặt tên model, thư mục model trên Hugging Face, và tên "nhận dạng" của model (không cần torch).

Cách xếp trên repo model (--hf_repo):
    models/<tên>/config.json, tokenizer.json, last.pt, best.pt, data.pt     <- train_lm.py (pretrain)
    models/<tên>-sft/config.json, tokenizer.json, sft_best.pt, sft_state.json <- train_sft.py (fine-tune)
Tên: --name tự đặt; không đặt thì tự sinh  yui-<cỡ>-v<N>  (vd yui-1b-v1) với N lớn hơn mọi bản đã có trong repo VÀ không trùng tên repo nào của tài khoản HF.
Không đặt --name mà repo đã có yui-<cỡ>-vN thì TIẾP TỤC bản mới nhất đó (để train tiếp trên phiên Kaggle mới); thêm --new_model để bắt đầu bản mới.
Repo kiểu cũ (checkpoint nằm ngay gốc repo, chưa có thư mục models/) vẫn chạy như cũ.
Tên được ghi vào config.json ("name") và train_sft.py thêm vài hội thoại để model tự biết tên mình.
"""
import re

NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,62}$")
VER_RE = re.compile(r"-v(\d+)(?:-sft\d*)?$")


def sanitize_name(s):
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", (s or "").strip()).strip("-.")
    if not NAME_RE.match(s):
        raise ValueError(f"tên model không hợp lệ: '{s}' (chỉ dùng chữ không dấu, số, . _ -; tối đa 63 ký tự)")
    return s


def base_name(tag):
    return f"yui-{tag.lower()}"


def model_dirs(files):
    """{tên: {tên file}} cho mọi thư mục models/<tên>/ trong danh sách file của repo."""
    out = {}
    for f in files:
        p = f.split("/")
        if len(p) >= 3 and p[0] == "models":
            out.setdefault(p[1], set()).add("/".join(p[2:]))
    return out


def _ver(name):
    m = VER_RE.search(name)
    return int(m.group(1)) if m else 0


def next_name(base, taken):
    """base-v<N> với N = (số lớn nhất đã dùng cho base) + 1, và chắc chắn không nằm trong `taken` (không phân biệt hoa thường)."""
    low = {t.lower() for t in taken}
    used = [_ver(t) for t in taken if t.lower().startswith(base.lower() + "-v") and _ver(t)]
    n = max(used, default=0) + 1
    while f"{base}-v{n}".lower() in low:
        n += 1
    return f"{base}-v{n}"


def taken_names(api, files, repo, repo_type="dataset"):
    """Mọi tên đã bị chiếm: thư mục models/ trong repo + tên các repo (model/dataset) của tài khoản trên HF (quét được thì quét, lỗi mạng thì bỏ qua)."""
    taken = set(model_dirs(files))
    try:
        user = repo.split("/")[0] if "/" in repo else api.whoami()["name"]
        for lst in (api.list_models(author=user), api.list_datasets(author=user)):
            for m in lst:
                taken.add(str(getattr(m, "id", m)).split("/")[-1])
    except Exception:  # noqa: BLE001
        pass
    return taken


def _api(api):
    if api is not None:
        return api
    from huggingface_hub import HfApi
    return HfApi()


def _files(api, repo, repo_type):
    try:
        return api.list_repo_files(repo, repo_type=repo_type)
    except Exception:  # noqa: BLE001
        return []


def setup_identity(args, tag, api=None, log=print):
    """train_lm.py: đặt args.name và args.hf_dir (thư mục trong repo, '' = kiểu cũ ở gốc repo)."""
    args.hf_dir = ""
    new_model = getattr(args, "new_model", False)
    if not args.hf_repo:
        args.name = sanitize_name(args.name) if args.name else None
        return
    api = _api(api)
    files = _files(api, args.hf_repo, args.hf_type)
    dirs = model_dirs(files)
    base = base_name(tag)
    if args.name:
        name = sanitize_name(args.name)
        how = "tiếp tục" if name in dirs else "tạo mới"
    else:
        mine = sorted((n for n in dirs if re.fullmatch(re.escape(base) + r"-v\d+", n)), key=_ver)
        if mine and not new_model and "config.json" in dirs[mine[-1]]:
            name, how = mine[-1], "tiếp tục (thêm --new_model để tạo bản mới)"
        elif not dirs and "config.json" in files and not new_model:
            args.name = None
            log("[train] repo kiểu cũ (checkpoint ở gốc repo, chưa có thư mục models/): giữ nguyên cách cũ. Thêm --name để chuyển sang models/<tên>/")
            return
        else:
            name, how = next_name(base, taken_names(api, files, args.hf_repo, args.hf_type) | set(dirs)), "tạo mới"
    args.name, args.hf_dir = name, f"models/{name}/"
    log(f"[train] tên model: {name} ({how}); thư mục trên {args.hf_repo}: {args.hf_dir}")


def setup_sft(args, api=None, log=print):
    """train_sft.py: chọn model gốc (args.src, thư mục args.hf_dir) và nơi đẩy kết quả (args.out_name, args.sft_dir)."""
    args.hf_dir = args.sft_dir = ""
    args.out_name = args.name
    if args.local or not args.hf_repo:
        return
    api = _api(api)
    files = _files(api, args.hf_repo, args.hf_type)
    dirs = model_dirs(files)
    pre = {n: f for n, f in dirs.items() if "config.json" in f and not re.search(r"-sft\d*$", n)}
    if getattr(args, "src", None):
        src = sanitize_name(args.src)
        if src not in dirs:
            raise SystemExit(f"[sft] không thấy models/{src}/ trong {args.hf_repo} (có: {', '.join(sorted(dirs)) or 'không có'})")
    elif pre:
        src = sorted(pre, key=lambda n: (_ver(n), n))[-1]
        if len(pre) > 1:
            log(f"[sft] repo có nhiều model ({', '.join(sorted(pre))}); lấy bản mới nhất: {src}. Dùng --src để chọn bản khác.")
    elif "config.json" in files:   # repo kiểu cũ
        log("[sft] repo kiểu cũ (checkpoint ở gốc repo): fine-tune như trước")
        return
    else:
        raise SystemExit(f"[sft] chưa có model nào trong {args.hf_repo} (cần train_lm.py trước, hoặc kiểm tra --hf_repo)")
    args.src, args.hf_dir = src, f"models/{src}/"
    if args.name:
        out = sanitize_name(args.name)
    else:
        out = f"{src}-sft"
        if getattr(args, "new_model", False) and out in dirs:   # --new_model: giữ bản cũ, tạo <src>-sft2, -sft3...
            n = 2
            while f"{src}-sft{n}" in dirs:
                n += 1
            out = f"{src}-sft{n}"
    args.out_name, args.sft_dir = out, f"models/{out}/"
    log(f"[sft] model gốc: {src} ({args.hf_dir}); kết quả sẽ đẩy vào {args.sft_dir} với tên '{out}'")


IDENTITY_NAME_Q = ["bạn là model nào", "bạn là phiên bản nào vậy", "tên model của bạn là gì", "mày là bản nào thế", "bạn đang chạy bản nào", "phiên bản của bạn là gì"]
IDENTITY_PERSONA_Q = ["bạn tên gì", "bạn là ai", "tên bạn là gì vậy", "mày tên gì", "bạn là ai thế", "giới thiệu bản thân đi"]


def identity_jsonl(name, persona="Yui"):
    """Vài hội thoại để model biết tên mình (JSONL cùng định dạng N.txt)."""
    import json
    rows = []
    for q in IDENTITY_NAME_Q:
        for a in (f"mik là {name} nha", f"bản này tên {name} đó", f"{name} nè bạn"):
            rows.append([q, a])
    for q in IDENTITY_PERSONA_Q:
        for a in (f"mik là {persona} nè, đang chạy bản {name}", f"{persona} đây! bản của mik là {name} đó"):
            rows.append([q, a])
    return ("\n".join(json.dumps({"messages": [{"role": "user", "content": q}, {"role": "assistant", "content": a}]}, ensure_ascii=False) for q, a in rows) + "\n").encode("utf-8")


def setup_generate(args, api=None, log=print):
    """train_lm.py --generate: chọn model để chat thử (--name, hoặc bản fine-tune/pretrain mới nhất)."""
    args.hf_dir = ""
    if not args.hf_repo:
        return
    api = _api(api)
    files = _files(api, args.hf_repo, args.hf_type)
    dirs = model_dirs(files)
    ok = {n for n, f in dirs.items() if "config.json" in f}
    if args.name:
        name = sanitize_name(args.name)
    else:
        pre = sorted((n for n in ok if not re.search(r"-sft\d*$", n)), key=lambda n: (_ver(n), n))
        if not pre:
            return   # repo kiểu cũ
        name = pre[-1]
        sfts = sorted(n for n in ok if n.startswith(name + "-sft"))
        name = sfts[-1] if sfts else name
    args.name, args.hf_dir = name, f"models/{name}/"
    log(f"[train] dùng model {name} ({args.hf_dir})")
