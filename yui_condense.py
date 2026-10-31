#!/usr/bin/env python3
"""
yui_condense.py - LỚP BỌC RÚT GỌN cho kết quả MCP / JSON: biến JSON dài, nhiều trường rác thành vài dòng chữ gọn để model nhỏ (1B, ngữ cảnh ngắn) không bị tràn token.

  condense(text, chars)  ->  chuỗi vừa `chars` ký tự
Quy tắc: bỏ null/rỗng và trường nhiễu (…_url, node_id, avatar, etag, _links, ...); danh sách đối tượng -> mỗi mục một đoạn "tên: trường=giá trị" chỉ giữ vài trường quan trọng nhất,
chỉ vài mục đầu + "(+N mục nữa)"; ngày rút về YYYY-MM-DD; số thực làm tròn; chuỗi dài bị cắt; HTML/markdown thừa được dọn. Không phải JSON thì chỉ gọn khoảng trắng + cắt.
Nếu vẫn quá dài thì giảm dần (số mục -> số trường -> độ dài chuỗi) cho tới khi vừa, nên luôn có kết quả hữu ích thay vì bị cắt ngang giữa chừng.
Dữ liệu đi qua đây vẫn là DỮ LIỆU KHÔNG TIN CẬY: yui_agent.sanitize() chạy sau đó như mọi kết quả công cụ.
"""
import json
import re

NOISE_EXACT = {"node_id", "gravatar_id", "avatar_url", "etag", "_links", "links", "href", "self", "sha", "tree_url", "url_template", "permissions", "meta", "metadata",
               "__typename", "cursor", "pageinfo", "page_info", "request_id", "trace_id", "headers", "cache", "score_details"}
NOISE_SUFFIX = ("_url", "_urls", "_template", "_etag")
LABEL_KEYS = ("name", "title", "full_name", "label", "id", "key", "path", "symbol", "slug", "term", "filename")
PRIORITY = ("title", "name", "status", "state", "price", "value", "amount", "count", "total", "description", "summary", "text", "content", "snippet", "message", "body",
            "language", "stars", "stargazers_count", "forks", "type", "url", "html_url", "link", "date", "created_at", "updated_at", "author", "user", "email", "result", "answer")
_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})[T ]\d{2}:\d{2}(:\d{2}(\.\d+)?)?(Z|[+-]\d{2}:?\d{2})?$")
_TAG = re.compile(r"(?s)<[^>]+>")


def _is_noise(k):
    k = str(k).lower()
    return k in NOISE_EXACT or k.endswith(NOISE_SUFFIX) and k not in ("html_url",)


def _s(v, n):
    if isinstance(v, str):
        m = _DATE.match(v)
        if m:
            return m.group(1)
        v = _TAG.sub(" ", v) if "<" in v else v
        v = re.sub(r"[#*`>_]{1,}\s*", "", v) if v.lstrip()[:1] in "#*>`" else v
        v = " ".join(v.split())
        return v if len(v) <= n else v[: max(1, n - 1)].rstrip() + "…"
    if isinstance(v, float):
        return f"{v:.4g}"
    if isinstance(v, bool):
        return "có" if v else "không"
    return str(v)


def _empty(v):
    return v is None or v == "" or v == [] or v == {}


def _flat(v, n, depth):
    """Giá trị lồng nhau -> chuỗi ngắn một dòng."""
    if isinstance(v, dict):
        label = next((v[k] for k in LABEL_KEYS if isinstance(v.get(k), (str, int, float)) and v.get(k) != ""), None)
        if label is not None and depth >= 1:
            return _s(label, n)
        return "{" + _kv(v, 3, n, depth + 1) + "}"
    if isinstance(v, list):
        return "[" + ", ".join(_flat(x, n, depth + 1) for x in v[:3]) + (f", +{len(v) - 3}" if len(v) > 3 else "") + "]"
    return _s(v, n)


def _kv(d, nfields, n, depth):
    items = [(k, v) for k, v in d.items() if not _empty(v) and not _is_noise(k)]
    rank = {k: i for i, k in enumerate(PRIORITY)}
    items.sort(key=lambda kv: rank.get(str(kv[0]).lower(), 99))
    return "; ".join(f"{k}={_flat(v, n, depth)}" for k, v in items[:nfields])


def _label(d):
    for k in LABEL_KEYS:
        v = d.get(k)
        if isinstance(v, (str, int, float)) and v != "":
            return k, _s(v, 40)
    return None, ""


def render(obj, n_items, n_fields, n_str):
    if isinstance(obj, dict) and len(obj) <= 3 and len(obj) >= 1:   # lớp vỏ kiểu {"results": [...]} -> mở ra
        vals = [v for k, v in obj.items() if not _empty(v) and not _is_noise(k)]
        lists = [v for v in vals if isinstance(v, list) and v]
        if len(lists) == 1 and len(vals) <= 3:
            extra = {k: v for k, v in obj.items() if not isinstance(v, (list, dict)) and not _empty(v) and not _is_noise(k)}
            head = ("; ".join(f"{k}={_s(v, n_str)}" for k, v in list(extra.items())[:2]) + " | ") if extra else ""
            return head + render(lists[0], n_items, n_fields, n_str)
    if isinstance(obj, list):
        objs = [x for x in obj if isinstance(x, dict)]
        if objs and len(objs) >= len(obj) // 2:
            out = []
            for i, d in enumerate(objs[:n_items], 1):
                lk, lv = _label(d)
                rest = {k: v for k, v in d.items() if k != lk}
                body = _kv(rest, n_fields, n_str, 1)
                out.append(f"{i}. {lv}: {body}" if lv and body else f"{i}. {lv or body}")
            more = len(obj) - len(out)
            return " | ".join(out) + (f" (+{more} mục nữa)" if more > 0 else "")
        shown = ", ".join(_flat(x, n_str, 1) for x in obj[:n_items * 3])
        return shown + (f" (+{len(obj) - n_items * 3} nữa)" if len(obj) > n_items * 3 else "")
    if isinstance(obj, dict):
        return _kv(obj, max(n_fields * 2, 4), n_str, 0)
    return _s(obj, n_str * 3)


def _find_json(text):
    t = text.strip()
    if t[:1] in "[{":
        try:
            return json.loads(t)
        except ValueError:
            pass
    m = re.search(r"[\[{]", t)
    if m and len(t) - m.start() > 30:
        for end in (t.rfind("}"), t.rfind("]")):
            if end > m.start():
                try:
                    return json.loads(t[m.start(): end + 1])
                except ValueError:
                    continue
    return None


def condense(text, chars=300):
    """Rút gọn một kết quả (chuỗi JSON hoặc chữ thường) cho vừa `chars` ký tự."""
    text = str(text)
    if len(text) <= chars and not text.lstrip().startswith(("{", "[")):
        return " ".join(text.split())
    obj = _find_json(text)
    if obj is None:
        s = " ".join(_TAG.sub(" ", text).split())
        return s if len(s) <= chars else s[: chars - 1].rstrip() + "…"
    for n_items, n_fields, n_str in ((5, 5, 90), (4, 4, 70), (3, 3, 55), (3, 3, 40), (2, 2, 40), (2, 2, 28), (1, 2, 24)):
        out = render(obj, n_items, n_fields, n_str)
        if not out.strip():   # toàn trường rỗng (vd {"entities": [], "relations": []}): giữ nguyên dạng gọn để model biết là "không có gì"
            return " ".join(json.dumps(obj, ensure_ascii=False, separators=(",", ":")).split())[:chars]
        if len(out) <= chars:
            return out
    return out[: chars - 1].rstrip() + "…"
