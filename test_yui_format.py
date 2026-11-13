"""Thử yui_format.py (v1/v2/t2 + v3): xuất model nhỏ NGẪU NHIÊN (không cần PyTorch) ở f32/int8/q4/q4all/t2, đọc lại bằng bộ đọc thuần Python,
đối chiếu cấu hình + bảng tensor, thử file cụt/thừa/hỏng; nếu có g++ thì nạp cùng file bằng engine C++ để chắc bộ đọc khớp yui.cpp."""
import ctypes, json, os, shutil, struct, subprocess, sys
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from export_yui import write_yui, bytes_to_unicode  # noqa: E402
import yui_format  # noqa: E402
from yui_format import read_yui, read_meta, YuiFormatError, mat_bytes, summary, validate_meta, arch_id, count_params, expected_tensor_count  # noqa: E402

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
    for ver in (0, 4, 99):
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


@pytest.fixture(scope="module")
def engine_lib(tmp_path_factory):
    """Biên dịch yui.cpp thành thư viện dùng chung một lần cho cả module (bỏ qua nếu không có g++)."""
    if shutil.which("g++") is None:
        pytest.skip("không có g++")
    so = tmp_path_factory.mktemp("lib") / "libyui.so"
    r = subprocess.run(["g++", "-O1", "-std=c++17", "-fopenmp-simd", "-pthread", "-shared", "-fPIC", "-DYUI_LIB",
                        os.path.join(HERE, "yui.cpp"), "-o", str(so)], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    lib = ctypes.CDLL(str(so))
    lib.yui_load.restype = ctypes.c_void_p
    lib.yui_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    lib.yui_info.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.yui_meta.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.yui_free.argtypes = [ctypes.c_void_p]
    lib.yui_complete_begin.restype = ctypes.c_void_p
    lib.yui_complete_begin.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    lib.yui_gen_next.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    lib.yui_gen_end.argtypes = [ctypes.c_void_p]
    return lib


def _load(lib, path):
    err = ctypes.create_string_buffer(512)
    h = lib.yui_load(path.encode(), err, 512)
    return h, err.value.decode()


def _info(lib, h):
    buf = ctypes.create_string_buffer(1024)   # app Y-App cấp đúng 1024 byte: yui_info KHÔNG được phình quá
    assert lib.yui_info(h, buf, 1024) > 0
    return json.loads(buf.value.decode())


def _generate(lib, h, n=24):
    err = ctypes.create_string_buffer(256)
    g = lib.yui_complete_begin(h, "xin chào".encode(), json.dumps({"temperature": 0, "max_tokens": n, "seed": 1}).encode(), 4, err, 256)
    assert g, err.value.decode()
    out = b""
    buf = ctypes.create_string_buffer(256)
    while True:
        k = lib.yui_gen_next(g, buf, 256)
        if k <= 0:
            break
        out += buf.raw[:k]
    lib.yui_gen_end(g)
    return out


def test_engine_cpp_nap_duoc_cung_file(files, engine_lib):
    """Bộ đọc Python và yui.cpp phải đồng ý về cấu hình: engine nạp được file nào thì bộ đọc cũng đọc được, và V/D/layers/E khớp yui_info."""
    lib = engine_lib
    for q in EXPECT:
        h, err = _load(lib, files[q])
        assert h, f"{q}: engine không nạp được: {err}"
        j = _info(lib, h)
        buf = ctypes.create_string_buffer(8)
        assert lib.yui_meta(h, buf, 8) == 0 and buf.value == b""       # v1/v2: không có META
        lib.yui_free(h)
        c = read_yui(files[q])["cfg"]
        for k_py, k_js in (("V", "vocab"), ("D", "d"), ("layers", "layers"), ("E", "experts"), ("heads", "heads"), ("L", "ctx")):
            assert j[k_js] == c[k_py], (q, k_js, j)
        assert j["ver"] == (1 if q in ("f32", "int8") else 2)


# ====================================================================== v3: META + mục lục

META_IN = dict(name="thu-nghiem", lang=["vi", "en"], domains=["chat", "general"], license="MIT", description="model thử", created="2026-11-13")
HDR = 60   # magic 4 + 12×u32 + 2×f32


@pytest.fixture(scope="module")
def files3(tmp_path_factory):
    d = tmp_path_factory.mktemp("yui3")
    sd, cfg = make_sd(str(d / "tok.json"))
    out = {"_cfg": cfg, "_tok": str(d / "tok.json"), "_sd": sd}
    for q in EXPECT:
        p = str(d / f"m3_{q}.yui")
        write_yui(sd, cfg, str(d / "tok.json"), p, q, meta=dict(META_IN))
        out[q] = p
    return out


def _meta_span(raw):
    n = struct.unpack("<I", raw[HDR:HDR + 4])[0]
    return HDR + 4, HDR + 4 + n


def _rewrite_meta(raw, mutate):
    """Sửa META nhưng GIỮ NGUYÊN độ dài (đệm khoảng trắng) để mục lục không lệch -> thử riêng các kiểm tra chéo."""
    a, b = _meta_span(raw)
    m = json.loads(raw[a:b].decode("utf-8"))
    m.pop("description", None)   # chừa chỗ để các ca thử có thể dài hơn vài ký tự
    mutate(m)
    blob = json.dumps(m, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    assert len(blob) <= b - a, "ca thử làm META dài hơn"
    return raw[:a] + blob + b" " * (b - a - len(blob)) + raw[b:]


@pytest.mark.parametrize("q", list(EXPECT))
def test_v3_doc_lai_dung(files3, q):
    info = read_yui(files3[q])
    assert info["ver"] == 3 and info["quant"] == EXPECT[q][1]
    m = info["meta"]
    assert m["meta_version"] == 1 and m["name"] == "thu-nghiem" and m["license"] == "MIT"
    assert m["lang"] == ["vi", "en"] and m["domains"] == ["chat", "general"] and m["created"] == "2026-11-13"
    assert m["arch_id"] == f"amoe-d{D}-l{NL}-h4-e{E}-s1-f{2 * D}" == arch_id(info["cfg"])
    assert m["params"] == info["params"] == count_params(info["cfg"])
    assert m["quant"] == q and m["tokenizer_sha256"] == info["tokenizer_sha256"]
    assert len(info["tensors"]) == expected_tensor_count(info["cfg"])
    t = info["tensors"]
    assert t[0][1] == info["weights_offset"] and t[-1][1] + t[-1][2] == info["index_offset"]
    for a, b in zip(t, t[1:]):
        assert a[1] + a[2] == b[1]
    raw = open(files3[q], "rb").read()
    io, ib, fm = struct.unpack("<QI4s", raw[-16:])
    assert fm == b"YUIX" and io == info["index_offset"] and io + ib + 16 == len(raw)
    assert "v3" in summary(info) and "thu-nghiem" in summary(info)


@pytest.mark.parametrize("q", ["q4", "t2"])
def test_v3_trong_so_giong_het_v2(files, files3, q):
    """Phần trọng số v3 (từ sau tokenizer đến hết tensor cuối) phải trùng từng byte với v2: engine tái dùng được load_mat."""
    a, b = read_yui(files[q]), read_yui(files3[q])
    w2 = open(files[q], "rb").read()[a["weights_offset"]:]
    w3 = open(files3[q], "rb").read()[b["weights_offset"]:b["index_offset"]]
    assert w2 == w3
    # tokenizer cũng y hệt (nằm sau META trong v3)
    assert b["tokenizer_sha256"] == a["tokenizer_sha256"]


def test_v3_cung_tensor_voi_v2(files, files3):
    a = [(n, k) for n, _, _, k in read_yui(files["q4"])["tensors"]]
    b = [(n, k) for n, _, _, k in read_yui(files3["q4"])["tensors"]]
    assert a == b


def test_read_meta_nhanh_chi_can_dau_file(files, files3, tmp_path):
    raw = open(files3["q4"], "rb").read()
    a, b = _meta_span(raw)
    p = tmp_path / "head_only.yui"
    p.write_bytes(raw[:b])          # chỉ header + META, không có tokenizer/trọng số
    r = read_meta(str(p))
    assert r["ver"] == 3 and r["meta"]["name"] == "thu-nghiem" and r["cfg"]["D"] == D
    with pytest.raises(YuiFormatError):
        read_yui(str(p))             # đọc đầy đủ thì phải báo cụt
    assert read_meta(files["q4"])["meta"] is None   # v2: không có META


def test_v3_khoa_la_duoc_giu(files3, tmp_path):
    p = str(tmp_path / "x.yui")
    write_yui(files3["_sd"], files3["_cfg"], files3["_tok"], p, "q4", meta=dict(META_IN, tuong_lai={"a": 1}, base="goc", source_url="https://example.org"))
    m = read_yui(p)["meta"]
    assert m["tuong_lai"] == {"a": 1} and m["base"] == "goc" and m["source_url"] == "https://example.org"


def test_v3_mac_dinh_khi_thieu_truong(files3, tmp_path):
    p = str(tmp_path / "x.yui")
    write_yui(files3["_sd"], dict(files3["_cfg"], name="ten-tu-config"), files3["_tok"], p, "int8", meta={})
    m = read_yui(p)["meta"]
    assert m["name"] == "ten-tu-config" and m["license"] == "unknown" and m["lang"] == [] and m["domains"] == []
    assert len(m["created"]) == 10


@pytest.mark.parametrize("bad", [
    dict(domains=["khong-co"]), dict(domains=["Chat"]), dict(domains="chat"), dict(lang=["viet nam"]), dict(lang="vi"),
    dict(name="x" * 129), dict(name=""), dict(license=""), dict(created="13/11/2026"), dict(created="2026-13-40"),
    dict(description="x" * 301), dict(source_url=5),
])
def test_xuat_v3_tu_choi_meta_xau(files3, tmp_path, bad):
    with pytest.raises(YuiFormatError):
        write_yui(files3["_sd"], files3["_cfg"], files3["_tok"], str(tmp_path / "x.yui"), "q4", meta=dict(META_IN, **bad))


def test_xuat_v3_domain_chuan_va_other(files3, tmp_path):
    p = str(tmp_path / "x.yui")
    write_yui(files3["_sd"], files3["_cfg"], files3["_tok"], p, "q4", meta=dict(META_IN, domains=list(yui_format.DOMAINS)))
    assert read_yui(p)["meta"]["domains"] == list(yui_format.DOMAINS) and "other" in yui_format.DOMAINS


def test_doc_chap_nhan_domain_la_nhung_phai_dung_dang_slug(files3):
    """Lúc ĐỌC chỉ kiểm dạng slug (tương thích về sau); lúc XUẤT mới bắt theo danh sách chuẩn."""
    m = dict(META_IN, meta_version=1, arch_id="a", quant="q4", params=1, tokenizer_sha256="0" * 64, domains=["domain-moi"])
    assert validate_meta(m)["domains"] == ["domain-moi"]
    with pytest.raises(YuiFormatError):
        validate_meta(m, strict_domains=True)
    with pytest.raises(YuiFormatError):
        validate_meta(dict(m, domains=["Co Hoa"]))


def test_meta_lam_sach_ky_tu_dieu_khien(files3, tmp_path):
    p = str(tmp_path / "x.yui")
    write_yui(files3["_sd"], files3["_cfg"], files3["_tok"], p, "q4", meta=dict(META_IN, name="a\x00b\x1b[31mc", description="dong1\ndong2"))
    m = read_yui(p)["meta"]
    assert "\x00" not in m["name"] and "\x1b" not in m["name"] and "\n" not in m["description"]


def test_validate_meta_kieu_so_nguyen_khong_nhan_bool():
    base = dict(META_IN, meta_version=1, arch_id="a", quant="q4", params=1, tokenizer_sha256="0" * 64)
    validate_meta(base)
    for k, v in (("params", True), ("meta_version", True), ("params", 0), ("params", "1"), ("quant", "q9"), ("tokenizer_sha256", "ABC")):
        with pytest.raises(YuiFormatError):
            validate_meta(dict(base, **{k: v}))
    for k in ("name", "arch_id", "quant", "params", "tokenizer_sha256", "created", "license", "lang", "domains", "meta_version"):
        with pytest.raises(YuiFormatError, match=k):
            validate_meta({x: y for x, y in base.items() if x != k})
    with pytest.raises(YuiFormatError):
        validate_meta(["khong", "phai", "dict"])
    with pytest.raises(YuiFormatError, match="phiên bản"):
        validate_meta(dict(base, meta_version=2))


def test_v3_cut_bao_loi(files3, tmp_path):
    raw = open(files3["q4"], "rb").read()
    info = read_yui(files3["q4"])
    cuts = [0, 10, 59, 62, 100, info["weights_offset"] - 5, info["weights_offset"] + 10, len(raw) // 2,
            info["index_offset"], info["index_offset"] + 3, len(raw) - 17, len(raw) - 16, len(raw) - 1]
    for n in cuts:
        p = tmp_path / "cut.yui"
        p.write_bytes(raw[:n])
        with pytest.raises(YuiFormatError):
            read_yui(str(p))


def test_v3_chan_file_va_muc_luc_hong(files3, tmp_path):
    raw = open(files3["q4"], "rb").read()
    info = read_yui(files3["q4"])
    p = tmp_path / "bad.yui"
    bad = bytearray(raw); bad[-4:] = b"XXXX"
    p.write_bytes(bad)
    with pytest.raises(YuiFormatError, match="YUIX"):
        read_yui(str(p))
    bad = bytearray(raw); bad[-16:-8] = struct.pack("<Q", info["index_offset"] + 1)
    p.write_bytes(bad)
    with pytest.raises(YuiFormatError, match="chân file"):
        read_yui(str(p))
    p.write_bytes(raw + b"\0")                    # thừa 1 byte sau chân file -> chân file không còn ở cuối
    with pytest.raises(YuiFormatError):
        read_yui(str(p))
    io = info["index_offset"]
    for pos, what in ((io, "số mục"), (io + 4 + 2 + len("emb.weight"), "offset emb"), (io + 4 + 2 + len("emb.weight") + 16, "kind emb")):
        bad = bytearray(raw); bad[pos] ^= 0x01
        p.write_bytes(bad)
        with pytest.raises(YuiFormatError):
            read_yui(str(p))


def test_v3_meta_khong_khop_header(files3, tmp_path):
    raw = open(files3["q4"], "rb").read()
    p = tmp_path / "bad.yui"
    cases = [
        (lambda m: m.update(arch_id=m["arch_id"].replace("-d64-", "-d65-")), "arch_id"),
        (lambda m: m.update(params=m["params"] + 1), "params"),
        (lambda m: m.update(tokenizer_sha256="0" * 64), "tokenizer_sha256"),
        (lambda m: m.update(quant="int8"), "quant"),
        (lambda m: m.update(quant="t2"), "quant"),
    ]
    for mut, key in cases:
        p.write_bytes(_rewrite_meta(raw, mut))
        with pytest.raises(YuiFormatError, match=key):
            read_yui(str(p))
    # q4all trong META vẫn hợp lệ với header quant=2
    p.write_bytes(_rewrite_meta(raw, lambda m: m.update(quant="q4all")))
    assert read_yui(str(p))["meta"]["quant"] == "q4all"


def test_v3_meta_hong_cau_truc(files3, tmp_path):
    raw = open(files3["q4"], "rb").read()
    p = tmp_path / "bad.yui"
    for mut in (lambda m: m.pop("license"), lambda m: m.update(domains="chat"), lambda m: m.update(meta_version=2),
                lambda m: m.update(created="2026-02-30"), lambda m: m.update(name=5)):
        p.write_bytes(_rewrite_meta(raw, mut))
        with pytest.raises(YuiFormatError):
            read_yui(str(p))
    a, b = _meta_span(raw)
    p.write_bytes(raw[:a] + b"{" * (b - a) + raw[b:])               # không phải JSON
    with pytest.raises(YuiFormatError, match="JSON"):
        read_yui(str(p))
    p.write_bytes(raw[:a] + b"\xff" * (b - a) + raw[b:])            # không phải UTF-8
    with pytest.raises(YuiFormatError):
        read_yui(str(p))
    for n in (0, 65537, 0xFFFFFFFF):                                # meta_len ngoài khoảng cho phép
        bad = bytearray(raw); bad[HDR:HDR + 4] = struct.pack("<I", n)
        p.write_bytes(bad)
        with pytest.raises(YuiFormatError, match="META"):
            read_yui(str(p))
        with pytest.raises(YuiFormatError):
            read_meta(str(p))


def test_v3_config_ngoai_khoang_van_bi_chan(files3, tmp_path):
    raw = bytearray(open(files3["q4"], "rb").read())
    p = tmp_path / "bad.yui"
    bad = bytearray(raw); bad[4 + 4 * 6:4 + 4 * 6 + 4] = struct.pack("<I", 100000)   # E
    p.write_bytes(bad)
    with pytest.raises(YuiFormatError):
        read_yui(str(p))


# ====================================================================== v3 trên engine C++

@pytest.mark.parametrize("q", list(EXPECT))
def test_engine_nap_v3_va_tra_meta(files3, engine_lib, q):
    lib = engine_lib
    h, err = _load(lib, files3[q])
    assert h, f"{q}: engine không nạp được v3: {err}"
    j = _info(lib, h)
    assert j["ver"] == 3 and j["d"] == D and j["layers"] == NL and j["experts"] == E and isinstance(j["quant"], str)
    buf = ctypes.create_string_buffer(65537)
    n = lib.yui_meta(h, buf, 65537)
    assert n > 0
    m = json.loads(buf.value.decode("utf-8"))
    assert m == read_yui(files3[q])["meta"]                       # engine trả đúng META mà bộ đọc Python thấy
    assert lib.yui_meta(h, buf, 10) == -2                         # buffer nhỏ -> -2, không ghi tràn
    assert lib.yui_meta(None, buf, 100) == -1
    lib.yui_free(h)


@pytest.mark.parametrize("q_v2,q_v3", [("q4", "q4"), ("t2", "t2")])
def test_engine_v3_chay_giong_het_v2(files, files3, engine_lib, q_v2, q_v3):
    """Cùng trọng số: v2 và v3 phải sinh đúng cùng văn bản (temperature 0)."""
    lib = engine_lib
    h2, _ = _load(lib, files[q_v2]); h3, _ = _load(lib, files3[q_v3])
    assert h2 and h3
    a, b = _generate(lib, h2), _generate(lib, h3)
    lib.yui_free(h2); lib.yui_free(h3)
    assert len(a) > 0 and a == b


def test_engine_v3_f32_int8_giong_v1(files, files3, engine_lib):
    lib = engine_lib
    for q in ("f32", "int8"):
        h1, _ = _load(lib, files[q]); h3, _ = _load(lib, files3[q])
        assert h1 and h3
        a, b = _generate(lib, h1), _generate(lib, h3)
        lib.yui_free(h1); lib.yui_free(h3)
        assert len(a) > 0 and a == b, q


def test_engine_v3_cut_va_hong_bao_loi_khong_crash(files3, engine_lib, tmp_path):
    lib = engine_lib
    raw = open(files3["q4"], "rb").read()
    info = read_yui(files3["q4"])
    io = info["index_offset"]
    p = tmp_path / "bad.yui"

    def must_fail(data, why):
        p.write_bytes(bytes(data))
        h, err = _load(lib, str(p))
        if h:
            lib.yui_free(h)
        assert not h, why
        assert err, why

    for n in (0, 5, 60, 62, 100, info["weights_offset"] - 3, len(raw) // 2, io, io + 5, len(raw) - 17, len(raw) - 16, len(raw) - 1):
        must_fail(raw[:n], f"cắt {n}")
    must_fail(raw + b"\0", "thừa byte sau chân file")
    bad = bytearray(raw); bad[-4:] = b"XXXX"; must_fail(bad, "chân sai magic")
    bad = bytearray(raw); bad[-16:-8] = struct.pack("<Q", io + 1); must_fail(bad, "chân chỉ sai vị trí")
    bad = bytearray(raw); bad[io] ^= 1; must_fail(bad, "số mục sai")
    bad = bytearray(raw); bad[io + 4 + 2 + len("emb.weight")] ^= 1; must_fail(bad, "offset mục lục lệch")
    for n in (0, 65537, 0xFFFFFFFF):
        bad = bytearray(raw); bad[HDR:HDR + 4] = struct.pack("<I", n); must_fail(bad, f"meta_len {n}")
    a, b = _meta_span(raw)
    must_fail(raw[:a] + b"x" * (b - a) + raw[b:], "META không phải JSON object")
    bad = bytearray(raw); bad[4:8] = struct.pack("<I", 4); must_fail(bad, "ver 4")
    # file v3 nguyên vẹn vẫn nạp được sau chuỗi thử trên
    p.write_bytes(raw)
    h, err = _load(lib, str(p))
    assert h, err
    lib.yui_free(h)
