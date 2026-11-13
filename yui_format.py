"""
yui_format.py - bộ đọc định dạng .yui thuần Python (chỉ dùng thư viện chuẩn; không cần torch/numpy).

Đọc header, META (v3), thông tin tokenizer và BẢNG VỊ TRÍ từng tensor (tên -> offset, size, kiểu) của file .yui v1/v2/t2/v3,
mà KHÔNG nạp trọng số vào RAM (chỉ tính kích thước từng ma trận rồi nhảy qua; kiểu ma trận đọc từ byte kiểu hoặc 2 byte nhóm).
Dùng lại cho: make_manifest.py (P2), verify_cpp.py.

    from yui_format import read_yui, read_meta
    meta = read_meta("model.yui")["meta"]       # v3: đọc nhanh, chỉ header + META (không đụng trọng số); v1/v2: None
    info = read_yui("model_q4.yui")             # đọc + kiểm toàn bộ bố cục (và mục lục nếu v3)
    info["cfg"]["D"], info["tokenizer_sha256"], info["tensors"][0]   # ('emb.weight', offset, size, kind)

Đặc tả: docs/YUI_FORMAT.md. Thứ tự và cách ghi: export_yui.write_yui và yui.cpp::Model::load.
File hỏng/cụt/lạ -> YuiFormatError với thông báo tiếng Việt (không treo, không đọc tràn).
"""
import datetime
import hashlib
import json
import os
import re
import struct

MAGIC = b"YUI1"
FOOT_MAGIC = b"YUIX"
SUPPORTED_VER = (1, 2, 3)
KIND_NAMES = {0: "f32", 1: "int8", 2: "q4", 3: "t2"}
QUANT_NAMES = {0: "f32", 1: "int8", 2: "q4", 3: "t2"}   # trường quant ở header (q4 và q4all cùng = 2)
META_QUANTS = ("f32", "int8", "q4", "q4all", "t2")
DOMAINS = ("general", "chat", "code", "math", "vi-fiction", "translate", "tool", "other")   # danh sách chuẩn (xuất file bắt buộc theo)
META_VERSION = 1

# trần an toàn khi đọc (file lạ/hỏng không được làm ta cấp phát/lặp vô hạn)
MAX_E = 256
MAX_LAYERS = 4096
MAX_DIM = 1 << 20
MAX_META = 65536
MAX_NAME = 128          # tên model và tên tensor trong mục lục
MAX_DESC = 300
MAX_URL = 512
FOOTER_BYTES = 16

_RE_SLUG = re.compile(r"^[a-z0-9][a-z0-9-]{0,31}$")
_RE_LANG = re.compile(r"^[a-z]{2,3}(-[A-Za-z0-9]{2,8})?$")
_RE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
_RE_SHA = re.compile(r"^[0-9a-f]{64}$")
_RE_CTRL = re.compile("[\x00-\x1f\x7f-\x9f\u2028\u2029]")


class YuiFormatError(ValueError):
    pass


class _R:
    """Đọc có kiểm tra biên trên file đã mở (cụt -> YuiFormatError thay vì trả byte thiếu)."""

    def __init__(self, f, size):
        self.f, self.size = f, size

    def tell(self):
        return self.f.tell()

    def seek(self, pos):
        if pos < 0 or pos > self.size:
            raise YuiFormatError("vị trí nằm ngoài file .yui")
        self.f.seek(pos)

    def need(self, n, what="dữ liệu"):
        if n < 0 or self.f.tell() + n > self.size:
            raise YuiFormatError(f"file .yui bị cụt (thiếu {what} tại byte {self.f.tell()}, cần {n}, file {self.size} byte)")

    def read(self, n, what="dữ liệu"):
        self.need(n, what)
        b = self.f.read(n)
        if len(b) != n:
            raise YuiFormatError(f"file .yui bị cụt khi đọc {what}")
        return b

    def unpack(self, fmt, what="dữ liệu"):
        return struct.unpack(fmt, self.read(struct.calcsize(fmt), what))

    def skip(self, n, what="dữ liệu"):
        self.need(n, what)
        self.f.seek(n, os.SEEK_CUR)


def mat_bytes(kind, rows, cols, group=0):
    """Số byte của MỘT ma trận (không tính byte kiểu/2 byte nhóm đứng trước)."""
    if kind == 0:
        return rows * cols * 4
    if kind == 1:
        return rows * 4 + rows * cols                    # hệ số f32 mỗi hàng + int8
    if kind == 2:
        nb = (cols + 31) // 32
        return rows * nb * 2 + rows * nb * 16            # hệ số fp16 mỗi khối + 16 byte 4-bit
    if kind == 3:
        if group < 32 or group % 32 or cols % group or cols % 32:
            raise YuiFormatError(f"ma trận t2 có nhóm {group} không hợp lệ với {cols} cột")
        return rows * (cols // group) * 2 + rows * (cols // 32) * 8
    raise YuiFormatError(f"kiểu ma trận {kind} không hợp lệ")


def arch_id(cfg):
    """Chuỗi cấu trúc tự suy từ header (cfg: khoá V, D, layers, heads, E, F, n_shared)."""
    return f"amoe-d{cfg['D']}-l{cfg['layers']}-h{cfg['heads']}-e{cfg['E']}-s{cfg['n_shared']}-f{cfg['F']}"


def expected_tensor_count(cfg):
    return 4 + cfg["layers"] * (9 + 3 * (cfg["E"] + cfg["n_shared"]))


def count_params(cfg):
    """Tổng số tham số theo cấu hình (khớp với tổng các tensor ở mục 4 của đặc tả)."""
    V, D, NL, E, F, ns = cfg["V"], cfg["D"], cfg["layers"], cfg["E"], cfg["F"], cfg["n_shared"]
    per_layer = 2 * D + 3 * D * D + D * D + 2 * D + E * D + D + 1 + (E + ns) * 3 * F * D
    return V * D + NL * per_layer + 2 * D + V * D


# ---------------------------------------------------------------- META (v3)

def _clean(s):
    return _RE_CTRL.sub(" ", s)


def _is_int(x):
    return isinstance(x, int) and not isinstance(x, bool)


def validate_meta(meta, strict_domains=False):
    """Kiểm cấu trúc META, trả bản sao đã làm sạch ký tự điều khiển. Khoá lạ được giữ nguyên (tương thích về sau).
    strict_domains=True (lúc XUẤT file): mỗi domain phải thuộc DOMAINS. Lúc ĐỌC thì chỉ cần đúng dạng slug."""
    if not isinstance(meta, dict):
        raise YuiFormatError("META phải là một đối tượng JSON")
    m = dict(meta)

    def need(key, typ):
        if key not in m:
            raise YuiFormatError(f"META thiếu khoá bắt buộc '{key}'")
        if not typ(m[key]):
            raise YuiFormatError(f"META: khoá '{key}' sai kiểu")
        return m[key]

    def need_str(key, maxlen, allow_empty=False):
        v = need(key, lambda x: isinstance(x, str))
        v = _clean(v).strip()
        if (not v and not allow_empty) or len(v) > maxlen:
            raise YuiFormatError(f"META: '{key}' rỗng hoặc dài quá {maxlen} ký tự")
        m[key] = v
        return v

    mv = need("meta_version", _is_int)
    if mv != META_VERSION:
        raise YuiFormatError(f"META phiên bản {mv} chưa hỗ trợ (bộ đọc này biết {META_VERSION})")
    need_str("name", MAX_NAME)
    need_str("arch_id", 128)
    if need_str("quant", 16) not in META_QUANTS:
        raise YuiFormatError(f"META: quant '{m['quant']}' không hợp lệ")
    p = need("params", _is_int)
    if p <= 0:
        raise YuiFormatError("META: params phải > 0")
    if not _RE_SHA.match(need_str("tokenizer_sha256", 64)):
        raise YuiFormatError("META: tokenizer_sha256 phải là 64 ký tự hex thường")
    c = need_str("created", 10)
    try:
        if not _RE_DATE.match(c):
            raise ValueError
        datetime.date.fromisoformat(c)
    except ValueError:
        raise YuiFormatError("META: created phải dạng YYYY-MM-DD hợp lệ") from None
    need_str("license", MAX_NAME)       # khoá bắt buộc, giá trị "unknown" được phép
    langs = need("lang", lambda x: isinstance(x, list) and len(x) <= 16)
    if not all(isinstance(x, str) and _RE_LANG.match(x) for x in langs):
        raise YuiFormatError("META: lang phải là mảng mã ngôn ngữ như 'vi', 'en'")
    doms = need("domains", lambda x: isinstance(x, list) and len(x) <= 16)
    for d in doms:
        if not (isinstance(d, str) and _RE_SLUG.match(d)):
            raise YuiFormatError("META: domains phải là mảng slug a-z, 0-9, '-'")
        if strict_domains and d not in DOMAINS:
            raise YuiFormatError(f"domain '{d}' không thuộc danh sách chuẩn {DOMAINS}")
    for key, mx in (("description", MAX_DESC), ("base", MAX_NAME), ("source_url", MAX_URL), ("exporter", MAX_NAME)):
        if key in m:
            need(key, lambda x: isinstance(x, str))
            m[key] = _clean(m[key]).strip()
            if len(m[key]) > mx:
                raise YuiFormatError(f"META: '{key}' dài quá {mx} ký tự")
    return m


def _read_header(r):
    if r.read(4, "magic") != MAGIC:
        raise YuiFormatError("không phải file .yui (sai magic, cần YUI1)")
    (ver, quant, V, D, NL, H, E, F, n_shared, L, min_k, max_k) = r.unpack("<12I", "header")
    top_p, min_strength = r.unpack("<2f", "header")
    if ver not in SUPPORTED_VER:
        raise YuiFormatError(f"phiên bản .yui {ver} chưa hỗ trợ (bộ đọc này biết {SUPPORTED_VER})")
    if quant > 3:
        raise YuiFormatError(f"trường quant {quant} không hợp lệ")
    if ver == 1 and quant > 1:
        raise YuiFormatError("file v1 chỉ có quant f32/int8")
    if not (0 < E <= MAX_E and 0 < NL <= MAX_LAYERS and 0 < H and 0 < D <= MAX_DIM and 0 < V <= MAX_DIM and n_shared <= 64):
        raise YuiFormatError("cấu hình .yui không hợp lệ (E/layers/heads/D/V ngoài khoảng cho phép)")
    if D % H or D % (H * 2):
        raise YuiFormatError(f"D={D} phải chia hết cho 2*heads (heads={H}); engine cần head_dim chẵn")
    cfg = dict(V=V, D=D, layers=NL, heads=H, E=E, F=F, n_shared=n_shared, L=L, min_k=min_k, max_k=max_k,
               top_p=top_p, min_strength=min_strength)
    return ver, quant, cfg


def _read_meta_block(r):
    (n,) = r.unpack("<I", "độ dài META")
    if n == 0 or n > MAX_META:
        raise YuiFormatError(f"META dài {n} byte, ngoài khoảng cho phép (1..{MAX_META})")
    raw = r.read(n, "META")
    try:
        obj = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, ValueError):
        raise YuiFormatError("META không phải JSON UTF-8 hợp lệ") from None
    return validate_meta(obj)


def read_meta(path):
    """Đọc NHANH header + META (v3 đặt META ngay sau header nên không phải qua tokenizer/trọng số).
    -> dict(path, ver, quant, quant_name, cfg, meta, file_size); meta = None với v1/v2. Không kiểm toàn bộ file (dùng read_yui cho việc đó)."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        r = _R(f, size)
        ver, quant, cfg = _read_header(r)
        meta = _read_meta_block(r) if ver == 3 else None
    return dict(path=path, ver=ver, quant=quant, quant_name=QUANT_NAMES[quant], cfg=cfg, meta=meta, file_size=size)


# ---------------------------------------------------------------- đọc toàn bộ

def _parse_index(r, count_expected):
    (count,) = r.unpack("<I", "số mục lục")
    if count != count_expected:
        raise YuiFormatError(f"mục lục có {count} mục, cấu hình cần {count_expected}")
    out = []
    for _ in range(count):
        (n,) = r.unpack("<H", "tên trong mục lục")
        if n == 0 or n > MAX_NAME:
            raise YuiFormatError("tên tensor trong mục lục dài/ngắn bất thường")
        try:
            name = r.read(n, "tên tensor").decode("utf-8")
        except UnicodeDecodeError:
            raise YuiFormatError("tên tensor trong mục lục không phải UTF-8") from None
        off, size, kind, rows, cols, group = r.unpack("<QQBIIH", "mục lục")
        out.append((name, off, size, kind, rows, cols, group))
    return out


def read_yui(path, with_tensors=True):
    """-> dict: ver, quant, cfg{...}, meta (v3, ngược lại None), nvocab, nmerges, eos, nranges, tokenizer_sha256,
    weights_offset, params, file_size, tensors[].

    tensors: danh sách (tên, offset, size, kind) theo đúng thứ tự trong file; offset/size gồm cả byte kiểu và 2 byte nhóm (nếu có),
    kind: 'f32' | 'int8' | 'q4' | 't2' (vector f32 như LayerNorm/router cũng ghi 'f32').
    File v3: kiểm thêm footer + mục lục khớp đúng bố cục tính được, và META khớp header (arch_id, quant, params, tokenizer_sha256)."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        r = _R(f, size)
        ver, quant, cfg = _read_header(r)
        V, D, NL, E, F, n_shared = cfg["V"], cfg["D"], cfg["layers"], cfg["E"], cfg["F"], cfg["n_shared"]
        meta = _read_meta_block(r) if ver == 3 else None

        # tokenizer: vocab + merges + eos là phần băm (bỏ bảng khoảng Unicode vì phụ thuộc phiên bản Python lúc xuất)
        tok_start = r.tell()
        (nvocab,) = r.unpack("<I", "số token")
        if nvocab != V:
            raise YuiFormatError(f"nvocab {nvocab} != V {V}")
        for _ in range(nvocab):
            (n,) = r.unpack("<H", "độ dài token")
            r.skip(n, "token")
        (nmerges,) = r.unpack("<I", "số merge")
        for _ in range(nmerges):
            for _k in range(2):
                (n,) = r.unpack("<H", "độ dài merge")
                r.skip(n, "merge")
        (eos,) = r.unpack("<i", "eos")
        tok_end = r.tell()
        (nranges,) = r.unpack("<I", "số khoảng Unicode")
        r.skip(nranges * 12, "bảng khoảng Unicode")
        weights_offset = r.tell()

        h = hashlib.sha256()
        f.seek(tok_start)
        left = tok_end - tok_start
        while left:
            chunk = f.read(min(left, 1 << 20))
            h.update(chunk)
            left -= len(chunk)
        f.seek(weights_offset)

        tensors, ex, params = [], [], 0   # ex: (name, off, size, kind_id, rows, cols, group) cho việc đối chiếu mục lục

        def vec(name, *shape):
            nonlocal params
            n = 1
            for s in shape:
                n *= s
            off = r.tell()
            r.skip(n * 4, name)
            tensors.append((name, off, n * 4, "f32"))
            ex.append((name, off, n * 4, 0, shape[0] if len(shape) == 2 else 1, shape[-1], 0))
            params += n

        def mat(name, rows, cols):
            nonlocal params
            off = r.tell()
            group = 0
            if ver == 1:
                kind = 0 if quant == 0 else 1
            else:
                (kind,) = r.unpack("<B", name)
                if kind == 3:
                    (group,) = r.unpack("<H", name)
            if kind > 3:
                raise YuiFormatError(f"{name}: kiểu ma trận {kind} không hợp lệ")
            r.skip(mat_bytes(kind, rows, cols, group), name)
            tensors.append((name, off, r.tell() - off, KIND_NAMES[kind]))
            ex.append((name, off, r.tell() - off, kind, rows, cols, group))
            params += rows * cols

        mat("emb.weight", V, D)
        for l in range(NL):
            p = f"blocks.{l}."
            vec(p + "n1.weight", D); vec(p + "n1.bias", D)
            mat(p + "attn.qkv.weight", 3 * D, D)
            mat(p + "attn.o.weight", D, D)
            vec(p + "n2.weight", D); vec(p + "n2.bias", D)
            vec(p + "moe.router.gate.weight", E, D)
            vec(p + "moe.router.strength_head.weight", 1, D)
            vec(p + "moe.router.strength_head.bias", 1)
            for e in range(E):
                q = f"{p}moe.experts.{e}."
                mat(q + "w_gate.weight", F, D); mat(q + "w_up.weight", F, D); mat(q + "w_down.weight", D, F)
            for s in range(n_shared):
                q = f"{p}moe.shared.{s}."
                mat(q + "w_gate.weight", F, D); mat(q + "w_up.weight", F, D); mat(q + "w_down.weight", D, F)
        vec("norm.weight", D); vec("norm.bias", D)
        mat("head.weight", V, D)
        end = r.tell()

        if ver == 3:
            # chân file: u64 index_offset, u32 index_bytes, "YUIX"
            if size - end < FOOTER_BYTES + 4:
                raise YuiFormatError("file .yui v3 thiếu mục lục/chân file (bị cụt?)")
            r.seek(size - FOOTER_BYTES)
            idx_off, idx_bytes, fm = r.unpack("<QI4s", "chân file")
            if fm != FOOT_MAGIC:
                raise YuiFormatError("file .yui v3: chân file sai magic YUIX (bị cụt hoặc ghi đè?)")
            if idx_off != end or idx_off + idx_bytes + FOOTER_BYTES != size:
                raise YuiFormatError("file .yui v3: chân file chỉ sai vị trí mục lục")
            r.seek(idx_off)
            idx = _parse_index(r, len(ex))
            if r.tell() != size - FOOTER_BYTES:
                raise YuiFormatError("file .yui v3: mục lục dài/ngắn hơn số byte khai báo")
            for want, got in zip(ex, idx):
                if want != got:
                    raise YuiFormatError(f"mục lục v3 lệch với bố cục thật tại tensor '{want[0]}'")
            # META phải khớp header
            exp_q = QUANT_NAMES[quant]
            if not (meta["quant"] == exp_q or (quant == 2 and meta["quant"] in ("q4", "q4all"))):
                raise YuiFormatError(f"META.quant '{meta['quant']}' không khớp header ({exp_q})")
            if meta["arch_id"] != arch_id(cfg):
                raise YuiFormatError(f"META.arch_id '{meta['arch_id']}' không khớp cấu hình ({arch_id(cfg)})")
            if meta["params"] != params:
                raise YuiFormatError(f"META.params {meta['params']} không khớp số tham số thật {params}")
            if meta["tokenizer_sha256"] != h.hexdigest():
                raise YuiFormatError("META.tokenizer_sha256 không khớp tokenizer trong file")
        elif end != size:
            raise YuiFormatError(f"file .yui có {size - end} byte thừa sau trọng số (hoặc sai cấu hình)")

    out = dict(path=path, ver=ver, quant=quant, quant_name=QUANT_NAMES[quant], cfg=cfg, meta=meta, nvocab=nvocab,
               nmerges=nmerges, eos=eos, nranges=nranges, tokenizer_sha256=h.hexdigest(), weights_offset=weights_offset,
               params=params, file_size=size)
    if ver == 3:
        out["index_offset"] = end
    if with_tensors:
        out["tensors"] = tensors
    return out


def summary(info):
    c = info["cfg"]
    s = (f"{os.path.basename(info['path'])}: v{info['ver']} {info['quant_name']}, V={c['V']} D={c['D']} layers={c['layers']} "
         f"heads={c['heads']} E={c['E']} L={c['L']}, {info['params'] / 1e6:.2f}M tham số, {info['file_size'] / 1e6:.2f} MB, "
         f"tokenizer {info['tokenizer_sha256'][:12]}")
    m = info.get("meta")
    if m:
        s += f" | META: {m['name']}, domains={','.join(m['domains']) or '-'}, lang={','.join(m['lang']) or '-'}, license={m['license']}"
    return s


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        try:
            print(summary(read_yui(p)))
        except YuiFormatError as e:
            print(f"{p}: LỖI - {e}")
            sys.exit(1)
