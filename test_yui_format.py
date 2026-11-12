"""Thử yui_format.py: xuất model nhỏ NGẪU NHIÊN (không cần PyTorch) ở f32/int8/q4/q4all/t2, đọc lại bằng bộ đọc thuần Python,
đối chiếu cấu hình + bảng tensor, thử file cụt/thừa/hỏng; nếu có g++ thì nạp cùng file bằng engine C++ để chắc bộ đọc khớp yui.cpp."""
import ctypes, json, os, shutil, struct, subprocess, sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from export_yui import write_yui, bytes_to_unicode  # noqa: E402
from yui_format import read_yui, YuiFormatError, mat_bytes, summary  # noqa: E402

V, D, NL, E, L = 300, 64, 2, 4, 32


def make_sd(tok_path):
    rng = np.random.default_rng(1)
    b2u = bytes_to_unicode()
    vocab = {b2u[b]: b for b in range(256)}
    vocab["<eos>"] = 256
    while len(vocab) < V:
        vocab[f"zz{len(vocab)}"] = len(vocab)
    json.dump({"model": {"type": "BPE", "vocab": vocab, "merges": []},
               "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False}}, open(tok_path, "w"))
    F = 2 * D
    sd = {}

    def w(n, *sh, s=0.05):
        sd[n] = (rng.standard_normal(sh) * s).astype(np.float32)
    w("emb.weight", V, D)
    for l in range(NL):
        p = f"blocks.{l}."
        for n in ("n1", "n2"):
            sd[p + n + ".weight"] = np.ones(D, np.float32); sd[p + n + ".bias"] = np.zeros(D, np.float32)
        w(p + "attn.qkv.weight", 3 * D, D); w(p + "attn.o.weight", D, D)
        w(p + "moe.router.gate.weight", E, D, s=0.3); w(p + "moe.router.strength_head.weight", 1, D)
        sd[p + "moe.router.strength_head.bias"] = np.zeros(1, np.float32)
        for e in range(E):
            for n, sh in (("w_gate", (F, D)), ("w_up", (F, D)), ("w_down", (D, F))):
                w(f"{p}moe.experts.{e}.{n}.weight", *sh)
        for n, sh in (("w_gate", (F, D)), ("w_up", (F, D)), ("w_down", (D, F))):
            w(f"{p}moe.shared.0.{n}.weight", *sh)
    sd["norm.weight"] = np.ones(D, np.float32); sd["norm.bias"] = np.zeros(D, np.float32)
    w("head.weight", V, D)
    return sd, dict(vocab=V, d=D, layers=NL, experts=E, top_p=0.999, L=L)


@pytest.fixture(scope="module")
def files(tmp_path_factory):
    d = tmp_path_factory.mktemp("yui")
    sd, cfg = make_sd(str(d / "tok.json"))
    out = {}
    for q in ("f32", "int8", "q4", "q4all", "t2"):
        p = str(d / f"m_{q}.yui")
        write_yui(sd, cfg, str(d / "tok.json"), p, q)
        out[q] = p
    return out


EXPECT = {"f32": (1, 0, "f32"), "int8": (1, 1, "int8"), "q4": (2, 2, "q4"), "q4all": (2, 2, "q4"), "t2": (2, 3, "t2")}


@pytest.mark.parametrize("q", list(EXPECT))
def test_doc_dung_cau_hinh_va_bang_tensor(files, q):
    info = read_yui(files[q])
    ver, qf, expert_kind = EXPECT[q]
    c = info["cfg"]
    assert (info["ver"], info["quant"]) == (ver, qf)
    assert (c["V"], c["D"], c["layers"], c["E"], c["F"], c["L"], c["heads"]) == (V, D, NL, E, 2 * D, L, 4)
    assert info["file_size"] == os.path.getsize(files[q])
    t = info["tensors"]
    # tensor liên tiếp, không hở, đầu = weights_offset, cuối = hết file
    assert t[0][1] == info["weights_offset"]
    for a, b in zip(t, t[1:]):
        assert a[1] + a[2] == b[1]
    assert t[-1][1] + t[-1][2] == info["file_size"]
    names = [x[0] for x in t]
    assert len(names) == len(set(names))
    assert names[0] == "emb.weight" and names[-1] == "head.weight"
    kinds = {x[0]: x[3] for x in t}
    assert kinds["blocks.0.moe.router.gate.weight"] == "f32" and kinds["blocks.0.n1.weight"] == "f32"   # router/LayerNorm luôn f32
    assert kinds["blocks.1.moe.experts.3.w_up.weight"] == expert_kind
    n_tensors = 1 + NL * (2 + 2 + 2 + 3 + 3 * E + 3) + 2 + 1
    assert len(t) == n_tensors
    assert info["params"] > 0 and info["nvocab"] == V and info["eos"] == 256


def test_tham_so_khong_phu_thuoc_luong_tu_hoa(files):
    assert len({read_yui(files[q])["params"] for q in EXPECT}) == 1
    assert len({read_yui(files[q])["tokenizer_sha256"] for q in EXPECT}) == 1   # cùng tokenizer -> cùng băm


def test_kich_thuoc_nho_dan(files):
    s = {q: os.path.getsize(files[q]) for q in EXPECT}
    assert s["f32"] > s["int8"] > s["q4"] > s["q4all"]


def test_mat_bytes_cong_thuc():
    assert mat_bytes(0, 3, 64) == 3 * 64 * 4
    assert mat_bytes(1, 3, 64) == 3 * 4 + 3 * 64
    assert mat_bytes(2, 3, 40) == 3 * 2 * 2 + 3 * 2 * 16      # 40 cột -> 2 khối, đệm 0
    assert mat_bytes(3, 2, 64, 64) == 2 * 1 * 2 + 2 * 2 * 8
    with pytest.raises(YuiFormatError):
        mat_bytes(3, 2, 64, 48)
    with pytest.raises(YuiFormatError):
        mat_bytes(9, 1, 1)


def test_file_cut_bao_loi_gon(files, tmp_path):
    raw = open(files["q4"], "rb").read()
    for n in (0, 3, 10, 60, 200, len(raw) // 2, len(raw) - 1):
        p = tmp_path / "cut.yui"
        p.write_bytes(raw[:n])
        with pytest.raises(YuiFormatError):
            read_yui(str(p))


def test_file_thua_byte(files, tmp_path):
    p = tmp_path / "extra.yui"
    p.write_bytes(open(files["int8"], "rb").read() + b"\0\0\0")
    with pytest.raises(YuiFormatError, match="thừa"):
        read_yui(str(p))


def test_magic_sai_va_phien_ban_la(files, tmp_path):
    raw = bytearray(open(files["f32"], "rb").read())
    p = tmp_path / "bad.yui"
    bad = bytearray(raw); bad[:4] = b"XXXX"
    p.write_bytes(bad)
    with pytest.raises(YuiFormatError, match="magic"):
        read_yui(str(p))
    for ver in (0, 3, 99):
        bad = bytearray(raw); bad[4:8] = struct.pack("<I", ver)
        p.write_bytes(bad)
        with pytest.raises(YuiFormatError, match="phiên bản"):
            read_yui(str(p))


def test_cau_hinh_ngoai_khoang(files, tmp_path):
    raw = bytearray(open(files["f32"], "rb").read())
    off = {"E": 4 + 4 * 6, "heads": 4 + 4 * 5, "quant": 4 + 4}
    p = tmp_path / "bad.yui"
    for field, val in (("E", 100000), ("E", 0), ("heads", 0), ("heads", 5), ("heads", 64), ("quant", 9)):
        bad = bytearray(raw); bad[off[field]:off[field] + 4] = struct.pack("<I", val)
        p.write_bytes(bad)
        with pytest.raises(YuiFormatError):
            read_yui(str(p))


def test_summary_va_khong_doc_tensor(files):
    info = read_yui(files["q4"], with_tensors=False)
    assert "tensors" not in info
    assert "q4" in summary(info)


@pytest.mark.skipif(shutil.which("g++") is None, reason="không có g++")
def test_engine_cpp_nap_duoc_cung_file(files, tmp_path):
    """Bộ đọc Python và yui.cpp phải đồng ý về cấu hình: engine nạp được file nào thì bộ đọc cũng đọc được, và V/D/layers/E khớp yui_info."""
    so = tmp_path / "libyui.so"
    r = subprocess.run(["g++", "-O1", "-std=c++17", "-fopenmp-simd", "-pthread", "-shared", "-fPIC", "-DYUI_LIB",
                        os.path.join(HERE, "yui.cpp"), "-o", str(so)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    lib = ctypes.CDLL(str(so))
    lib.yui_load.restype = ctypes.c_void_p
    lib.yui_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    lib.yui_info.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.yui_free.argtypes = [ctypes.c_void_p]
    for q in EXPECT:
        err = ctypes.create_string_buffer(256)
        h = lib.yui_load(files[q].encode(), err, 256)
        assert h, f"{q}: engine không nạp được: {err.value.decode()}"
        buf = ctypes.create_string_buffer(8192)
        assert lib.yui_info(h, buf, 8192) > 0
        j = json.loads(buf.value.decode())
        lib.yui_free(h)
        c = read_yui(files[q])["cfg"]
        for k_py, k_js in (("V", "vocab"), ("D", "d"), ("layers", "layers"), ("E", "experts"), ("heads", "heads"), ("L", "ctx")):
            assert j[k_js] == c[k_py], (q, k_js, j)
