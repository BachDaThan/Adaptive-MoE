"""bitnet_pack.py — phần numpy thuần (KHÔNG cần PyTorch) của định dạng ternary "t2" cho yui.cpp.
export_yui.py và bitnet_yui.py cùng dùng file này, nên bố cục bit chỉ có MỘT nguồn."""
import numpy as np


def pack_t2(q):
    """q int8 [R,C] (C % 32 == 0) -> uint8 [R, C/32, 8].
    Mỗi khối 32 trọng số = 8 byte; byte j chứa trọng số j, j+8, j+16, j+24 ở các cặp bit 0-1, 2-3, 4-5, 6-7.
    Mã: 0 -> 0, 1 -> +1, 2 -> -1."""
    q = np.asarray(q, dtype=np.int8)
    R, C = q.shape
    assert C % 32 == 0
    code = np.where(q == -1, 2, q).astype(np.uint8).reshape(R, C // 32, 4, 8)   # [.., s, j]: trọng số j + 8*s
    return (code[:, :, 0] | (code[:, :, 1] << 2) | (code[:, :, 2] << 4) | (code[:, :, 3] << 6)).astype(np.uint8)


def unpack_t2(p):
    p = np.asarray(p, dtype=np.uint8)
    R, nb, _ = p.shape
    code = np.stack([(p >> s) & 3 for s in (0, 2, 4, 6)], 2)                     # [R, nb, 4, 8]
    code = code.astype(np.int8)
    return np.where(code == 2, -1, code).astype(np.int8).reshape(R, nb * 32)


def ternary_quantize_np(w, group):
    """Bản numpy của ternarize() (cho export_yui.py): -> (scale fp16 [R, C/group], packed uint8 [R, C/32, 8], q int8)."""
    w = np.ascontiguousarray(w, dtype=np.float32)
    R, C = w.shape
    wg = w.reshape(R, C // group, group)
    s = np.maximum(np.abs(wg).mean(-1, keepdims=True), 1e-5)
    q = np.clip(np.rint(wg / s), -1, 1).astype(np.int8).reshape(R, C)
    return s[..., 0].astype(np.float16), pack_t2(q), q
