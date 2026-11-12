"""
yui_format.py - bộ đọc định dạng .yui thuần Python (chỉ dùng thư viện chuẩn; không cần torch/numpy).

Đọc header, thông tin tokenizer và BẢNG VỊ TRÍ từng tensor (tên -> offset, size, kiểu) của file .yui v1/v2/t2,
mà KHÔNG nạp trọng số vào RAM (chỉ tính kích thước từng ma trận rồi nhảy qua; kiểu ma trận đọc từ byte kiểu hoặc 2 byte nhóm).
Dùng lại cho: .yui v3 (P1), make_manifest.py (P2), verify_cpp.py.

    from yui_format import read_yui
    info = read_yui("model_q4.yui")
    info["cfg"]["D"], info["tokenizer_sha256"], info["tensors"][0]   # ('emb.weight', offset, size, kind)

Thứ tự và cách ghi: xem export_yui.write_yui và yui.cpp::Model::load (đặc tả: docs/YUI_FORMAT.md khi có).
File hỏng/cụt/lạ -> YuiFormatError với thông báo tiếng Việt (không treo, không đọc tràn).
"""
import hashlib
import os
import struct

MAGIC = b"YUI1"
SUPPORTED_VER = (1, 2)
KIND_NAMES = {0: "f32", 1: "int8", 2: "q4", 3: "t2"}
QUANT_NAMES = {0: "f32", 1: "int8", 2: "q4", 3: "t2"}   # trường quant ở header (q4 và q4all cùng = 2)

# trần an toàn khi đọc (file lạ/hỏng không được làm ta cấp phát/lặp vô hạn)
MAX_E = 256
MAX_LAYERS = 4096
MAX_DIM = 1 << 20
MAX_STR = 65535


class YuiFormatError(ValueError):
    pass


class _R:
    """Đọc có kiểm tra biên trên file đã mở (cụt -> YuiFormatError thay vì trả byte thiếu)."""

    def __init__(self, f, size):
        self.f, self.size = f, size

    def tell(self):
        return self.f.tell()

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


def read_yui(path, with_tensors=True):
    """-> dict: ver, quant, cfg{...}, nvocab, nmerges, eos, nranges, tokenizer_sha256, weights_offset, params, file_size, tensors[].

    tensors: danh sách (tên, offset, size, kind) theo đúng thứ tự trong file; offset/size gồm cả byte kiểu và 2 byte nhóm (nếu có),
    kind: 'f32' | 'int8' | 'q4' | 't2' (vector f32 như LayerNorm/router cũng ghi 'f32')."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        r = _R(f, size)
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

        tensors, params = [], 0

        def vec(name, *shape):
            nonlocal params
            n = 1
            for s in shape:
                n *= s
            off = r.tell()
            r.skip(n * 4, name)
            tensors.append((name, off, n * 4, "f32"))
            params += n

        def mat(name, rows, cols):
            nonlocal params
            off = r.tell()
            if ver == 1:
                kind = 0 if quant == 0 else 1
                group = 0
            else:
                (kind,) = r.unpack("<B", name)
                group = 0
                if kind == 3:
                    (group,) = r.unpack("<H", name)
            if kind > 3:
                raise YuiFormatError(f"{name}: kiểu ma trận {kind} không hợp lệ")
            r.skip(mat_bytes(kind, rows, cols, group), name)
            tensors.append((name, off, r.tell() - off, KIND_NAMES[kind]))
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
        if end != size:
            raise YuiFormatError(f"file .yui có {size - end} byte thừa sau trọng số (hoặc sai cấu hình)")

    out = dict(path=path, ver=ver, quant=quant, quant_name=QUANT_NAMES[quant], cfg=cfg, nvocab=nvocab, nmerges=nmerges,
               eos=eos, nranges=nranges, tokenizer_sha256=h.hexdigest(), weights_offset=weights_offset,
               params=params, file_size=size)
    if with_tensors:
        out["tensors"] = tensors
    return out


def summary(info):
    c = info["cfg"]
    return (f"{os.path.basename(info['path'])}: v{info['ver']} {info['quant_name']}, V={c['V']} D={c['D']} layers={c['layers']} "
            f"heads={c['heads']} E={c['E']} L={c['L']}, {info['params'] / 1e6:.2f}M tham số, {info['file_size'] / 1e6:.2f} MB, "
            f"tokenizer {info['tokenizer_sha256'][:12]}")


if __name__ == "__main__":
    import sys
    for p in sys.argv[1:]:
        try:
            print(summary(read_yui(p)))
        except YuiFormatError as e:
            print(f"{p}: LỖI - {e}")
            sys.exit(1)
