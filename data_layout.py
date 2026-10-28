"""
data_layout.py - cách xếp data trên repo Hugging Face theo cỡ model (không cần torch).

    data/<cỡ>/1.txt, 2.txt, ...        văn bản để train từ đầu (các đoạn cách nhau bằng dòng trống)   vd data/50M/3.txt
    Fine-tune/<cỡ>/1.txt, 2.txt, ...   hội thoại fine-tune (mỗi dòng một {"messages":[...]})            vd Fine-tune/50M/3.txt

<cỡ> (gọi là "tag") tự tính từ cấu hình model: số tham số làm tròn, vd 50M, 120M, 1.5B. Thư mục chưa có thì make_data.py tự tạo khi up file đầu tiên.
train_lm.py / train_sft.py tự tính cùng tag đó từ config.json rồi vào đúng thư mục; không thấy thư mục cỡ đó thì dùng cách xếp cũ
(gốc repo N.txt cho fine-tune, pretrain/N.txt cho data train) nên repo cũ vẫn chạy bình thường.
Đặt biến MODEL_TAG (make_data.py) hoặc --size_tag (train_*.py) nếu muốn tự gọi tên thư mục, vd "50M-v2".
"""
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
