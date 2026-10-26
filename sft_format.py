"""
sft_format.py - định dạng hội thoại dùng chung cho train_sft.py và chat.py (không cần torch).

Mỗi lượt được ghép thành:   "User: <câu hỏi>\nYui: <câu trả lời><eos>"
Khi train chỉ tính loss trên phần câu trả lời + <eos> (không học thuộc lòng câu hỏi).
File dữ liệu 1.txt, 2.txt, ...: mỗi bản ghi là một JSON {"messages":[{"role":"user","content":...},{"role":"assistant","content":...}]};
các bản ghi có thể cách nhau bằng xuống dòng hoặc dấu cách, và trong chuỗi được phép có xuống dòng thật (dòng bị ngắt khi copy-paste).
Cũng nhận: mảng JSON, {"conversations":[{"from":"human","value":..}]}, {"user":..,"assistant":..}, {"prompt":..,"response":..}.
"""
import json
import re

USER_TAG, BOT_TAG = "User:", "Yui:"
MAX_MSG_CHARS = 600

_dec = json.JSONDecoder(strict=False)   # strict=False: cho phép ký tự xuống dòng thật bên trong chuỗi
_ws = re.compile(r"\s*")
_USER_ROLES = ("user", "human")
_BOT_ROLES = ("assistant", "bot", "gpt", "model")


def clean(t):
    """Gộp xuống dòng/khoảng trắng thừa thành một dấu cách (dòng bị ngắt do copy-paste)."""
    t = re.sub(r"\s*\n\s*", " ", str(t))
    return re.sub(r"[ \t]{2,}", " ", t).strip()


def to_pairs(msgs):
    """Danh sách tin nhắn -> [(user, assistant), ...]; bỏ tin rỗng hoặc sai thứ tự."""
    pairs, cur = [], None
    for m in msgs:
        if not isinstance(m, dict):
            continue
        role = str(m.get("role", "")).lower()
        text = clean(m.get("content", ""))
        if not text:
            continue
        if role in _USER_ROLES:
            cur = text if cur is None else cur + " " + text
        elif role in _BOT_ROLES and cur is not None:
            pairs.append((cur, text))
            cur = None
    return pairs


def obj_to_pairs(o):
    if not isinstance(o, dict):
        return []
    if isinstance(o.get("messages"), list):
        return to_pairs(o["messages"])
    if isinstance(o.get("conversations"), list):
        return to_pairs([{"role": m.get("role") or m.get("from"), "content": m.get("content") or m.get("value")}
                         for m in o["conversations"] if isinstance(m, dict)])
    for a, b in (("user", "assistant"), ("prompt", "response"), ("instruction", "output"), ("question", "answer")):
        if a in o and b in o:
            return to_pairs([{"role": "user", "content": o[a]}, {"role": "assistant", "content": o[b]}])
    return []


def parse_text(text):
    """Đọc lần lượt từng bản ghi JSON trong một file. Trả (list các cuộc hội thoại, số bản ghi hỏng)."""
    text = text.lstrip("\ufeff")
    convs, bad, i, n = [], 0, 0, len(text)
    while True:
        i = _ws.match(text, i).end()
        if i >= n:
            break
        if text[i] == ",":
            i += 1
            continue
        try:
            o, j = _dec.raw_decode(text, i)
        except ValueError:
            bad += 1
            k = text.find("{", i + 1)
            if k < 0:
                break
            i = k
            continue
        i = j
        for it in (o if isinstance(o, list) else [o]):
            p = obj_to_pairs(it)
            if p:
                convs.append(p)
            else:
                bad += 1
    return convs, bad


def load_convs(texts, max_chars=MAX_MSG_CHARS):
    """texts: {tên file: nội dung}. Trả (convs đã lọc + bỏ trùng, thống kê)."""
    seen, out = set(), []
    stats = {"hỏng": 0, "quá dài": 0, "trùng": 0}
    for name in sorted(texts):
        convs, bad = parse_text(texts[name])
        stats["hỏng"] += bad
        for c in convs:
            if any(len(u) > max_chars or len(a) > max_chars for u, a in c):
                stats["quá dài"] += 1
                continue
            key = tuple(c)
            if key in seen:
                stats["trùng"] += 1
                continue
            seen.add(key)
            out.append(c)
    return out, stats


def _seg(tok, u, a, eos):
    return tok.encode(f"{USER_TAG} {u}\n{BOT_TAG}").ids, tok.encode(" " + a).ids + [eos]


def encode_pairs(tok, pairs, eos, max_len):
    """-> (ids, mask) hoặc None. Quá dài thì bỏ bớt các lượt đầu. mask = 1 ở token câu trả lời và <eos>."""
    segs = [_seg(tok, u, a, eos) for u, a in pairs]
    while segs and sum(len(p) + len(r) for p, r in segs) > max_len:
        segs.pop(0)
    if not segs:
        return None
    ids, mask = [], []
    for p, r in segs:
        ids += p + r
        mask += [0] * len(p) + [1] * len(r)
    return ids, mask


def prompt_ids(tok, history, user_msg, eos, ctx_len, reserve=24):
    """Ghép lịch sử + câu mới thành chuỗi id, kết thúc bằng 'Yui:' để model viết câu trả lời."""
    budget = max(16, ctx_len - reserve)
    segs = [_seg(tok, clean(u), clean(a), eos) for u, a in history if clean(u) and clean(a)]
    final = tok.encode(f"{USER_TAG} {clean(user_msg)}\n{BOT_TAG}").ids
    while segs and sum(len(p) + len(r) for p, r in segs) + len(final) > budget:
        segs.pop(0)
    ids = [t for p, r in segs for t in p + r] + final
    return ids[-budget:]
