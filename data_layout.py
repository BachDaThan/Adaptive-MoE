"""
data_layout.py - cách xếp data trên repo Hugging Face theo cỡ model (không cần torch).

    data/<cỡ>/1.txt, 2.txt, ...        văn bản để train từ đầu (các đoạn cách nhau bằng dòng trống)   vd data/50M/3.txt
    Fine-tune/<cỡ>/1.txt, 2.txt, ...   hội thoại fine-tune (mỗi dòng một {"messages":[...]})            vd Fine-tune/50M/3.txt

<cỡ> (gọi là "tag") tự tính từ cấu hình model: số tham số làm tròn, vd 50M, 120M, 1.5B. Thư mục chưa có thì make_data.py tự tạo khi up file đầu tiên.
train_lm.py / train_sft.py tự tính cùng tag đó từ config.json rồi vào đúng thư mục; không thấy thư mục cỡ đó thì dùng cách xếp cũ
(gốc repo N.txt cho fine-tune, pretrain/N.txt cho data train) nên repo cũ vẫn chạy bình thường.
Đặt biến MODEL_TAG (make_data.py) hoặc --size_tag (train_*.py) nếu muốn tự gọi tên thư mục, vd "50M-v2".
"""
import math
import re


def params_from_arch(V, D, layers, experts, shared=1, ffn=None):
    """Số tham số Adaptive-MoE của repo này (khớp export_yui.py): embedding + head tách riêng; mỗi lớp attention 4D² + (E+shared) expert SwiGLU 3·D·F + router."""
    F = ffn or 2 * D
    per_layer = 4 * D * D + (experts + shared) * 3 * D * F + experts * D + 4 * D
    return 2 * V * D + layers * per_layer


def size_tag(params):
    """Làm tròn số tham số tới bậc 'đẹp' (5M khi dưới 100M, 10M khi dưới 1B, 0,1B trên đó): cùng một model luôn ra cùng một tên. Vd 49.9M -> '50M'."""
    p = float(params)
    if p >= 1e9:
        b = round(p / 1e8) / 10
        return f"{b:g}B"
    if p >= 1e8:
        return f"{int(round(p / 1e7) * 10)}M"
    if p >= 1e6:
        return f"{int(round(p / 5e6) * 5) or 5}M"
    return f"{max(1, int(round(p / 1e3)))}K"


def finetune_prefix(tag):
    return f"Fine-tune/{tag}/"


def data_prefix(tag):
    return f"data/{tag}/"


LEGACY = {"finetune": "", "pretrain": "pretrain/"}


def has_numbered(names, prefix):
    pat = re.compile("^" + re.escape(prefix) + r"\d+\.txt$")
    return any(pat.match(n) for n in names)


def pick_prefix(names, kind, tag):
    """Thư mục theo cỡ nếu đã có file; không thì kiểu cũ nếu có; không có gì thì trả thư mục theo cỡ (sẽ được tạo khi up)."""
    new = finetune_prefix(tag) if kind == "finetune" else data_prefix(tag)
    if has_numbered(names, new):
        return new
    if has_numbered(names, LEGACY[kind]):
        return LEGACY[kind]
    return new


def parse_params(s):
    """'50M' -> 50_000_000; '1.2B'; '300k'; '50000000'."""
    m = re.fullmatch(r"\s*([\d.]+)\s*([kKmMbBgG]?)\s*", str(s))
    if not m:
        raise ValueError(f"không hiểu số tham số '{s}' (vd 50M, 1.2B)")
    return int(float(m.group(1)) * {"": 1, "k": 1e3, "m": 1e6, "b": 1e9, "g": 1e9}[m.group(2).lower()])


def arch_for_params(target, vocab=16000, heads=4):
    """Chọn (d, layers, experts) cho Adaptive-MoE có số tham số gần `target` nhất, hình dáng cân đối (không quá cao/mỏng).
    d chia hết cho 2*heads, experts <= 256 (giới hạn của yui.cpp). Cùng target luôn ra cùng kết quả. Vd 50M -> d=256, 6 lớp, 16 expert (đúng preset small)."""
    best = None
    for d in range(128, 4097, 64):
        try:
            heads_for(d)
        except ValueError:
            continue
        for layers in range(4, 41):
            ratio = d / layers
            if ratio < 20 or ratio > 90:            # tránh model quá sâu-mỏng hoặc quá nông-bè
                continue
            for experts in (8, 16, 32, 64, 128):
                p = params_from_arch(vocab, d, layers, experts)
                err = abs(p - target) / target
                # phạt nhẹ hình dáng lệch khỏi preset (d/layers ~ 40, 16-32 expert) để chọn cấu hình "quen" khi sai số tương đương
                score = err + 0.01 * abs(math.log(ratio / 42)) + 0.004 * abs(math.log2(experts / 24))
                if best is None or score < best[0]:
                    best = (score, d, layers, experts)
    return best[1], best[2], best[3]


def lr_for_params(n):
    """Tốc độ học gợi ý, khớp các preset của train_lm.py (50M: 1e-3, ~250M: 6e-4, ~440M: 4e-4) và giảm dần khi model to."""
    return float(min(1.5e-3, max(1.5e-4, 1e-3 * (5e7 / max(n, 1)) ** 0.3)))


DEFAULT_DATA_REPO = "BachDaThan/TrainAI-web"


def heads_for(d):
    """Số head attention cho chiều ẩn d: 4 khi d <= 512 (giữ nguyên các preset cũ), model lớn hơn thì tăng sao cho mỗi head <= 128 chiều và d chia hết cho 2*head."""
    if d <= 512:
        return 4
    for h in range(4, 65):
        if d // h <= 128 and d % (2 * h) == 0:
            return h
    raise ValueError(f"không chọn được số head cho d={d}")


def find_data_repo(candidates, prefix, repo_type="dataset", api=None):
    """Repo đầu tiên (theo thứ tự ứng viên) có file N.txt trong `prefix`. Dữ liệu và checkpoint có thể nằm ở hai repo khác nhau
    (vd data ở BachDaThan/TrainAI-web, model ở BachDaThan/TrainAI-1B). Trả (repo, danh sách file) hoặc (None, [])."""
    if api is None:
        from huggingface_hub import HfApi
        api = HfApi()
    seen = set()
    for repo in candidates:
        if not repo or repo in seen:
            continue
        seen.add(repo)
        try:
            names = api.list_repo_files(repo, repo_type=repo_type)
        except Exception:  # noqa: BLE001
            continue
        if has_numbered(names, prefix):
            return repo, names
    return None, []
