"""
teencode.py - sinh teencode / tiếng Việt không dấu từ văn bản thường, để:
  1) tokenizer có sẵn token cho "ko", "dc", "zậy", "hnay", "ntn", "Boss", "Yui"... thay vì băm thành nhiều mảnh byte (xem langcheck.train_bpe_vi);
  2) model pre-train thấy một ít (vài %) văn bản kiểu nhắn tin, nên SFT về sau hiểu người dùng gõ teencode / không dấu (train_lm.py --teen_aug).

Không dùng thư viện ngoài. Có seed => lặp lại được.
  python teencode.py "hôm nay mình không biết làm gì luôn"     # xem thử vài biến thể
"""
import random
import re
import sys
import unicodedata

# từ chuẩn -> các biến thể teencode thường gặp (người Việt nhắn tin). Khớp theo TỪ nguyên (viết thường).
WORDS = {
    "không": ["ko", "k", "hok", "hông", "hong", "khum", "kh", "kg"],
    "được": ["dc", "đc", "đk", "đuoc"],
    "gì": ["j", "zì"],
    "vậy": ["zậy", "dzậy", "z", "v"],
    "rồi": ["r", "rùi", "ùi", "rầu"],
    "biết": ["bik", "bít", "biet"],
    "thôi": ["thui", "thoy", "thoi"],
    "luôn": ["lun", "luon"],
    "mình": ["mik", "mk", "mềnh", "minh"],
    "bạn": ["bn", "bro"],
    "nói": ["ns"],
    "làm": ["lm"],
    "đi": ["ik", "đi", "di"],
    "mới": ["ms"],
    "với": ["vs", "vs"],
    "quá": ["wá", "wa", "qá", "quá"],
    "người": ["ng", "ngừi"],
    "nhưng": ["nhưg", "nhg", "nma", "nhma"],
    "thích": ["thik", "thix", "thick"],
    "yêu": ["iu", "yeu"],
    "vợ": ["vk"], "chồng": ["ck"],
    "cũng": ["cx", "cug"],
    "nữa": ["nx", "nửa"],
    "hôm nay": ["hnay", "hnai", "hum nay", "nay"],
    "bao nhiêu": ["bn", "bnh", "bao nhiu"],
    "bình thường": ["bt", "bthg"],
    "điện thoại": ["dt", "đt", "đth"],
    "sinh nhật": ["sn", "sinh nhat"],
    "học sinh": ["hs"], "sinh viên": ["sv"],
    "như thế nào": ["ntn", "nthn"], "thế nào": ["ntn"],
    "không biết": ["kbt", "hok bik", "ko bik", "k bik"],
    "cảm ơn": ["cám ơn", "camon", "tks", "thanks", "cảm ơn"],
    "xin lỗi": ["sorry", "sr", "xl"],
    "dễ thương": ["dth", "dễ thươngg", "cute"],
    "vui": ["zui", "vui"],
    "buồn": ["bùn", "buồn"],
    "đẹp": ["đẹp", "đẹp"],
    "nhé": ["nhen", "nhaa", "nha"],
    "nhỉ": ["nhỉ", "nhở", "hen"],
    "ừ": ["uk", "ừa", "ờ"],
    "ok": ["oke", "okie", "okela", "ô kê"],
    "trời ơi": ["trời ơiii", "trời ơi", "chời ơi"],
    "tại sao": ["tại sao", "sao", "ts"],
    "chứ": ["chớ", "chứ"],
    "rất": ["rất", "rát", "r.ất"],
    "đúng": ["đúg", "đúng"], "thật": ["thật", "thiệt", "thêt"],
    "cái": ["cái", "cái"], "này": ["nè", "này", "ni"],
    "đây": ["đây", "đây nè"],
    "hả": ["hả", "hảaa", "hử"],
    "bây giờ": ["h", "bh", "bi giờ"], "giờ": ["h", "giờ"],
    "ngày mai": ["mai", "ngày mai", "ngmai"],
    "bạn bè": ["bạn bè", "bb"],
    "facebook": ["fb", "face"], "instagram": ["ig", "insta"],
    "nói chuyện": ["nc", "nói chuyện", "ns chuyện"],
    "tối": ["tối", "toi"],
    "anh": ["a", "anh"], "chị": ["c", "chị"], "em": ["e", "em"],
}
_MAXN = max(len(k.split()) for k in WORDS)
ENDINGS = ["kkk", "hihi", "=))", ":)))", "huhu", "hehe", "😂", "😭", "🥺", "=)))", "~", "nè", "nha", "á", "đó", "lun á"]
STRETCH = {"nha": "nhaa", "nè": "nèe", "quá": "quáaa", "luôn": "luônnn", "ơi": "ơiii", "á": "áaa", "hả": "hảaa", "đi": "điii", "ạ": "ạaa", "vậy": "vậyy"}

_COMBINING = re.compile(r"[\u0300-\u036f]")


def strip_diacritics(s):
    """'Hôm nay trời đẹp' -> 'Hom nay troi dep' (gõ không dấu)."""
    s = unicodedata.normalize("NFD", s)
    s = _COMBINING.sub("", s).replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", s)


def teencodify(text, rng=None, p_word=0.55, p_nodau=0.3, p_lower=0.7, p_stretch=0.12, p_end=0.2):
    """Biến câu chuẩn thành câu nhắn tin: thay từ bằng teencode (xác suất p_word), kéo dài chữ, đuôi cảm thán,
    thường hoá chữ, và (xác suất p_nodau) gõ hẳn không dấu."""
    rng = rng or random
    words = text.split()
    out, i = [], 0
    while i < len(words):
        done = False
        for n in range(min(_MAXN, len(words) - i), 0, -1):
            chunk = " ".join(words[i:i + n])
            core = chunk.strip(".,!?;:…\"'()").lower()
            if core in WORDS and rng.random() < p_word:
                tail = chunk[len(chunk.rstrip(".,!?;:…\"')")):]
                out.append(rng.choice(WORDS[core]) + tail)
                i += n
                done = True
                break
        if not done:
            out.append(words[i])
            i += 1
    t = " ".join(out)
    if rng.random() < p_lower:
        t = t.lower()
    t = " ".join(STRETCH[w] if (w in STRETCH and rng.random() < p_stretch) else w for w in t.split())
    if rng.random() < p_nodau:
        t = strip_diacritics(t)
    if rng.random() < p_end:
        t = t.rstrip(".") + " " + rng.choice(ENDINGS)
    return t


def augment_docs(docs, n, seed=0, lo=60, hi=420):
    """Lấy n đoạn ngắn (60-420 ký tự, cắt ở ranh giới câu) từ docs rồi teencode hoá -> list[str] kiểu tin nhắn."""
    rng = random.Random(seed)
    if not docs or n <= 0:
        return []
    out, tries = [], 0
    while len(out) < n and tries < n * 6:
        tries += 1
        d = docs[rng.randrange(len(docs))]
        if len(d) < lo * 2:
            continue
        s = rng.randrange(0, max(1, len(d) - hi))
        seg = d[s:s + rng.randint(lo, hi)]
        sents = [x.strip() for x in re.split(r"(?<=[.!?…\n])\s+", seg) if len(x.strip()) > 15]
        if len(sents) < 2:
            continue
        seg = " ".join(sents[1:-1] or sents[:1])   # bỏ câu đầu/cuối vì thường bị cắt giữa chừng
        if len(seg) < lo // 2:
            continue
        out.append(teencodify(seg, rng))
    return out


# Từ khoá nhân vật/giọng nói: lặp lại trong mẫu train tokenizer để BPE gộp thành 1 token (vd " Boss", " Yui", " zậy", " hnay")
BOOST_PHRASES = ["Boss", "Yui", "Boss ơi", "Yui nè", "mình nha", "ko", "dc", "đc", "zậy", "hnay", "ntn", "bik", "mik", "thui", "lun", "nè", "nha", "hong", "hăm",
                 "ùi", "kkk", "hihi", "ơi", "nhaaa", "okela", "Boss ơi, Yui nghĩ vầy nè", "Yui giúp Boss liền nha", "dạ Boss", "hnay trời đẹp ghê"]


def boost_docs(repeat=300):
    """Một ít 'văn bản' lặp các từ khoá trên (theo nhiều cách ghép) để chúng đủ tần suất thành token riêng."""
    rng = random.Random(7)
    docs = []
    for _ in range(repeat):
        k = rng.sample(BOOST_PHRASES, 6)
        docs.append(" ".join(k) + ".")
        docs.append(rng.choice(["Boss", "Yui"]) + " " + " ".join(rng.sample(BOOST_PHRASES, 4)) + " " + strip_diacritics(rng.choice(BOOST_PHRASES)))
    return docs


if __name__ == "__main__":
    src = " ".join(sys.argv[1:]) or "Hôm nay mình không biết làm gì luôn, bạn có biết chỗ nào vui không?"
    r = random.Random(1)
    for _ in range(5):
        print(teencodify(src, r))
