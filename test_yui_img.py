"""Thử đường ảnh của engine C++ (id token âm = hàng embedding ảnh) trên model .yui NGẪU NHIÊN, không cần PyTorch.
Ý chính: đưa vào các hàng embedding CỦA CHÍNH các token chữ "Q" qua đường ảnh thì phải ra y hệt đường chữ thường."""
import ctypes, json, os, shutil, subprocess, sys, tempfile
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from export_yui import write_yui  # noqa: E402
from test_yui_engine import make_sd  # noqa: E402

pytestmark = pytest.mark.skipif(shutil.which("g++") is None, reason="không có g++")


@pytest.fixture(scope="module")
def lib():
    d = tempfile.mkdtemp()
    so = os.path.join(d, "libyui.so")
    subprocess.check_call(["g++", "-O2", "-DYUI_LIB", "-shared", "-fPIC", "-std=c++17", "-pthread", os.path.join(HERE, "yui.cpp"), "-o", so])
    sd, cfg = make_sd(os.path.join(d, "tok.json"), L=128)
    model = os.path.join(d, "m.yui")
    write_yui(sd, cfg, os.path.join(d, "tok.json"), model, "f32")
    L = ctypes.CDLL(so)
    L.yui_load.restype = ctypes.c_void_p
    L.yui_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    L.yui_chat_begin_img.restype = ctypes.c_void_p
    L.yui_chat_begin_img.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    L.yui_complete_begin.restype = ctypes.c_void_p
    L.yui_complete_begin.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    L.yui_gen_next.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_int]
    L.yui_gen_end.argtypes = [ctypes.c_void_p]
    err = ctypes.create_string_buffer(256)
    h = L.yui_load(model.encode(), err, 256)
    assert h, err.value
    return L, h, sd


def run(L, g, limit=60):
    out = b""
    buf = ctypes.create_string_buffer(256)
    while True:
        n = L.yui_gen_next(g, buf, 256)
        if n <= 0: break
        out += buf.raw[:n]
        if len(out) > limit: break
    L.yui_gen_end(g)
    return out


OPTS = json.dumps({"max_new": 24, "temperature": 0, "top_k": 1, "repeat_penalty": 1.0, "spec": 0}).encode()


def test_image_rows_equal_text_rows(lib):
    L, h, sd = lib
    err = ctypes.create_string_buffer(256)
    q = "Hello"
    rows = np.ascontiguousarray(sd["emb.weight"][[ord(c) for c in q]], dtype=np.float32)   # tokenizer thử: token i = byte i
    msgs = json.dumps([{"role": "user", "content": ""}]).encode()
    g1 = L.yui_chat_begin_img(h, msgs, OPTS, 2, rows.ctypes.data, len(q), err, 256)
    assert g1, err.value
    a = run(L, g1)
    g2 = L.yui_complete_begin(h, f"User: {q}\nYui:".encode(), OPTS, 2, err, 256)
    assert g2, err.value
    b = run(L, g2)
    assert a == b and len(a) > 0, (a, b)


def test_image_changes_output_and_threads_ok(lib):
    L, h, sd = lib
    err = ctypes.create_string_buffer(256)
    rng = np.random.default_rng(1)
    msgs = json.dumps([{"role": "user", "content": "mô tả"}]).encode()
    outs = set()
    for _ in range(3):
        img = np.ascontiguousarray(rng.standard_normal((8, 64)).astype(np.float32) * 0.5)
        g = L.yui_chat_begin_img(h, msgs, OPTS, 2, img.ctypes.data, 8, err, 256)
        assert g, err.value
        outs.add(run(L, g))
    assert len(outs) > 1   # ảnh khác nhau -> câu trả lời khác nhau


def test_rejects_bad_image_and_too_small_ctx(lib):
    L, h, sd = lib
    err = ctypes.create_string_buffer(256)
    msgs = json.dumps([{"role": "user", "content": "x"}]).encode()
    assert not L.yui_chat_begin_img(h, msgs, OPTS, 2, None, 4, err, 256)
    img = np.zeros((200, 64), np.float32)   # L=128 -> budget 96 < 200 + 24
    assert not L.yui_chat_begin_img(h, msgs, OPTS, 2, img.ctypes.data, 200, err, 256)
    assert "ngữ cảnh" in err.value.decode()


# ---------- mã hoá ảnh qua ONNX Runtime (cần onnxruntime_c_api.h cạnh yui.cpp + pip install onnx onnxruntime) ----------
def _find_ort_lib():
    try:
        import onnxruntime, glob
        c = glob.glob(os.path.join(os.path.dirname(onnxruntime.__file__), "capi", "libonnxruntime.so*"))
        return c[0] if c else None
    except Exception:  # noqa: BLE001
        return None


@pytest.mark.skipif(not os.path.exists(os.path.join(HERE, "onnxruntime_c_api.h")) or _find_ort_lib() is None
                    or pytest.importorskip("onnx", reason="cần onnx") is None, reason="thiếu header/onnxruntime")
def test_vision_shim_matches_onnxruntime():
    import onnx, onnxruntime as ort
    from onnx import helper, TensorProto, numpy_helper
    d = tempfile.mkdtemp()
    rng = np.random.default_rng(0)
    W = rng.standard_normal((3, 16)).astype(np.float32)
    nodes = [helper.make_node("GlobalAveragePool", ["pixel_values"], ["g"]),
             helper.make_node("Reshape", ["g", "s1"], ["r"]),
             helper.make_node("MatMul", ["r", "W"], ["m"]),
             helper.make_node("Reshape", ["m", "s2"], ["image_embeds"])]
    graph = helper.make_graph(nodes, "t", [helper.make_tensor_value_info("pixel_values", TensorProto.FLOAT, [1, 3, 224, 224])],
                              [helper.make_tensor_value_info("image_embeds", TensorProto.FLOAT, [2, 8])],
                              [numpy_helper.from_array(W, "W"), numpy_helper.from_array(np.array([1, 3], np.int64), "s1"), numpy_helper.from_array(np.array([2, 8], np.int64), "s2")])
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 13)]); model.ir_version = 8
    path = os.path.join(d, "v.onnx"); onnx.save(model, path)
    so = os.path.join(d, "libyui.so")
    subprocess.check_call(["g++", "-O2", "-DYUI_LIB", "-shared", "-fPIC", "-std=c++17", "-pthread", "-I", HERE, os.path.join(HERE, "yui.cpp"), "-o", so])
    os.environ["YUI_ORT_LIB"] = _find_ort_lib()
    L = ctypes.CDLL(so)
    L.yui_vision_load.restype = ctypes.c_void_p
    L.yui_vision_load.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int]
    L.yui_vision_run.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_void_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_int]
    L.yui_vision_free.argtypes = [ctypes.c_void_p]
    err = ctypes.create_string_buffer(300)
    h = L.yui_vision_load(path.encode(), err, 300)
    assert h, err.value
    x = np.ascontiguousarray(rng.standard_normal((1, 3, 224, 224)).astype(np.float32))
    out = np.zeros(16, np.float32)
    n = L.yui_vision_run(h, x.ctypes.data, 224, out.ctypes.data, 16, err, 300)
    assert n == 16, err.value
    ref = ort.InferenceSession(path).run(None, {"pixel_values": x})[0].reshape(-1)
    assert np.allclose(out, ref, atol=1e-5)
    small = np.zeros(4, np.float32)
    assert L.yui_vision_run(h, x.ctypes.data, 224, small.ctypes.data, 4, err, 300) == -2
    L.yui_vision_free(h)
