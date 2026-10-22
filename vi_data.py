"""
vi_data.py - tokenize tiếng Việt ở mức âm tiết + phân loại token để phân tích.

Tiếng Việt viết cách nhau theo âm tiết, nên tokenize bằng regex là đủ, không cần
thư viện ngoài. Giữ nguyên hoa/thường để nhận diện tên riêng.
"""
import re
import unicodedata
from collections import Counter

import numpy as np

TOKEN_RE = re.compile(r"\w+|[^\w\s]", re.UNICODE)

CATS = ["dấu câu", "số", "từ chức năng", "tên riêng", "từ hiếm", "từ nội dung"]
PUNCT, NUMBER, FUNCTION, PROPER, RARE, CONTENT = range(6)

FUNC_WORDS = set("""
là và của có trong được cho một những các này đó với để không khi từ ra đã sẽ đang
như nhưng hay hoặc thì mà ở tại về bị nên vì do cũng rất đến lên vào theo sau trước
còn bởi nếu dù tuy cùng đều chỉ vẫn lại qua trên dưới giữa ngoài nào gì ai đâu sao
thế vậy nó tôi anh chị họ chúng
""".split())

SENT_START = {".", "!", "?", ":", ";", '"', "(", "“", "…", "<nl>"}


def tokenize(text):
    """Chuẩn hóa NFC, tách âm tiết/dấu câu, thêm <nl> ở cuối mỗi dòng."""
    text = unicodedata.normalize("NFC", text)
    toks = []
    for line in text.split("\n"):
        line = line.strip()
        if not line:
            continue
        toks.extend(TOKEN_RE.findall(line))
        toks.append("<nl>")
    return toks


def build_vocab(train_toks, size):
    cnt = Counter(train_toks)
    itos = ["<unk>"] + [t for t, _ in cnt.most_common(size - 1)]
    stoi = {t: i for i, t in enumerate(itos)}
    return itos, stoi, cnt


def encode(toks, stoi):
    return np.array([stoi.get(t, 0) for t in toks], dtype=np.int64)


def categorize(toks, cnt, rare_count=30):
    """
    Mỗi token -> một trong CATS. Thứ tự ưu tiên:
    dấu câu > số > từ chức năng > tên riêng (viết hoa, không đứng đầu câu)
    > từ hiếm (đếm < rare_count trong dữ liệu train) > từ nội dung.
    """
    out = np.empty(len(toks), dtype=np.uint8)
    prev = "<nl>"
    for i, t in enumerate(toks):
        if t == "<nl>" or not any(c.isalnum() for c in t):
            c = PUNCT
        elif any(ch.isdigit() for ch in t):
            c = NUMBER
        elif t.lower() in FUNC_WORDS:
            c = FUNCTION
        elif t[0].isupper() and prev not in SENT_START:
            c = PROPER
        elif cnt.get(t, 0) < rare_count:
            c = RARE
        else:
            c = CONTENT
        out[i] = c
        prev = t
    return out
