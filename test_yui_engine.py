"""Thử engine C++ trên model .yui NGẪU NHIÊN (không cần PyTorch): kernel nhanh phải khớp --exact về hướng logits, kết quả không phụ thuộc số luồng."""
import json, os, shutil, subprocess, sys, tempfile
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from export_yui import write_yui, bytes_to_unicode, q4_quantize, q4_dequantize, quantize_rows  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="không có g++")


def make_sd(tok_path, V=600, D=64, NL=2, E=8, L=32):
    rng = np.random.default_rng(0)
    b2u = bytes_to_unicode()
    vocab = {b2u[b]: b for b in range(256)}
    vocab["<eos>"] = 256
    while len(vocab) < V:
        vocab[f"zz{len(vocab)}"] = len(vocab)
    json.dump({"model": {"type": "BPE", "vocab": vocab, "merges": []}, "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": False}}, open(tok_path, "w"))
    F = 2 * D
    sd = {}
    def w(n, *sh, s=0.05): sd[n] = (rng.standard_normal(sh) * s).astype(np.float32)
    w("emb.weight", V, D)
    for l in range(NL):
        p = f"blocks.{l}."
        sd[p + "n1.weight"] = np.ones(D, np.float32); sd[p + "n1.bias"] = np.zeros(D, np.float32)
        sd[p + "n2.weight"] = np.ones(D, np.float32); sd[p + "n2.bias"] = np.zeros(D, np.float32)
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
    return sd, dict(vocab=V, d=D, layers=NL, experts=E, top_p=0.999, L=L, k_max=2)


def tiny_model(path, tok_path, quant="int8", **kw):
    sd, cfg = make_sd(tok_path, **kw)
    return write_yui(sd, cfg, tok_path, path, quant)


@pytest.fixture(scope="module")
def built():
    d = tempfile.mkdtemp()
    exe = os.path.join(d, "yui")
    subprocess.check_call(["g++", "-O2", "-std=c++17", "-fopenmp-simd", "-pthread", os.path.join(HERE, "yui.cpp"), "-o", exe])
    model = os.path.join(d, "m.yui")
    tiny_model(model, os.path.join(d, "tok.json"))
    return d, exe, model


def logits(d, exe, model, env=None, extra=()):
    out = os.path.join(d, "l.bin")
    subprocess.check_call([exe, "--model", model, "--ids", "5,9,40,300,17,3,99,100", "--k", "2", "--logits", out, *extra], env={**os.environ, **(env or {})},
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return np.fromfile(out, dtype=np.float32).reshape(8, -1)


def test_fast_close_to_exact(built):
    d, exe, model = built
    a, b = logits(d, exe, model), logits(d, exe, model, extra=["--exact"])
    cos = (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
    assert cos.min() > 0.95


def test_threads_do_not_change_result(built):
    d, exe, model = built
    assert np.array_equal(logits(d, exe, model, env={"YUI_THREADS": "1"}), logits(d, exe, model, env={"YUI_THREADS": "4"}))


def test_mmap_equals_read(built):
    d, exe, model = built
    assert np.array_equal(logits(d, exe, model, env={"YUI_MMAP": "1"}), logits(d, exe, model, env={"YUI_MMAP": "0"}))


def dequantized_f32_model(path, tok_path, quant, **kw):
    """Model float32 mà trọng số đúng bằng những gì file `quant` biểu diễn -> engine đọc file lượng tử với --exact phải ra cùng kết quả."""
    sd, cfg = make_sd(tok_path, **kw)
    out = {}
    for k, v in sd.items():
        big = k == "emb.weight" or k == "head.weight" or k.endswith(("attn.qkv.weight", "attn.o.weight")) or ".experts." in k or ".shared." in k
        if not big:
            out[k] = v
        elif quant == "q4all" or (quant == "q4" and (".experts." in k or ".shared." in k)):
            d, p = q4_quantize(v); out[k] = q4_dequantize(d, p, v.shape[1])
        else:
            sc, q = quantize_rows(v); out[k] = q.astype(np.float32) * sc[:, None]
    return write_yui(out, cfg, tok_path, path, "f32")


@pytest.mark.parametrize("quant", ["q4", "q4all"])
@pytest.mark.parametrize("D", [64, 72])   # 72: số cột không chia hết 32 (khối cuối được đệm 0)
def test_q4_file_matches_its_dequantized_weights(built, tmp_path, quant, D):
    d, exe, _ = built
    tok = os.path.join(d, "tok_q4.json")
    qpath, fpath = os.path.join(d, f"m_{quant}_{D}.yui"), os.path.join(d, f"f_{quant}_{D}.yui")
    tiny_model(qpath, tok, quant, D=D)
    dequantized_f32_model(fpath, tok, quant, D=D)
    ref = logits(d, exe, fpath)
    got = logits(d, exe, qpath, extra=["--exact"])
    assert np.abs(ref - got).max() < 1e-4 * max(1.0, np.abs(ref).max())
    fast = logits(d, exe, qpath)   # kernel nhanh (kích hoạt int8) gần với kết quả chính xác
    cos = (fast * got).sum(1) / (np.linalg.norm(fast, axis=1) * np.linalg.norm(got, axis=1))
    assert cos.min() > 0.9


def test_q4_is_smaller_and_threads_mmap_consistent(built):
    d, exe, _ = built
    tok = os.path.join(d, "tok_sz.json")
    p8, p4, p4a = (os.path.join(d, n) for n in ("s8.yui", "s4.yui", "s4a.yui"))
    n8, n4, n4a = tiny_model(p8, tok, "int8", D=128, V=1000), tiny_model(p4, tok, "q4", D=128, V=1000), tiny_model(p4a, tok, "q4all", D=128, V=1000)
    assert n4 < 0.75 * n8 and n4a < 0.62 * n8, (n8, n4, n4a)
    a = logits(d, exe, p4, env={"YUI_THREADS": "1", "YUI_MMAP": "1"})
    assert np.array_equal(a, logits(d, exe, p4, env={"YUI_THREADS": "4", "YUI_MMAP": "0"}))
