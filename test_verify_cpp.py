"""Thử verify_cpp.py (không cần PyTorch): xuất model nhỏ v2 và v3 CÙNG trọng số, build ./yui, chạy script với --skip_logits --compare.
Cần g++ và thư viện `tokenizers` (thiếu thì bỏ qua)."""
import json, os, shutil, subprocess, sys
import numpy as np
import pytest

pytest.importorskip("tokenizers")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from export_yui import write_yui  # noqa: E402
import test_yui_format as T  # noqa: E402  (dùng lại make_sd)


@pytest.fixture(scope="module")
def env(tmp_path_factory):
    if shutil.which("g++") is None:
        pytest.skip("không có g++")
    d = tmp_path_factory.mktemp("verify")
    exe = str(d / "yui")
    r = subprocess.run(["g++", "-O1", "-std=c++17", "-fopenmp-simd", "-pthread", os.path.join(HERE, "yui.cpp"), "-o", exe], capture_output=True, text=True)
    assert r.returncode == 0, r.stderr[-500:]
    out = d / "model_dir"
    out.mkdir()
    sd, cfg = T.make_sd(str(out / "tokenizer.json"))
    # tokenizer.json tối thiểu của make_sd chưa đủ trường để thư viện `tokenizers` nạp -> dựng lại bằng chính thư viện (cùng vocab, không merge)
    from tokenizers import Tokenizer, models, pre_tokenizers, decoders
    vocab = json.load(open(out / "tokenizer.json", encoding="utf-8"))["model"]["vocab"]
    tk = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.save(str(out / "tokenizer.json"))
    json.dump(cfg, open(out / "config.json", "w"))
    files = {}
    for q in ("q4", "t2", "f32"):
        write_yui(sd, cfg, str(out / "tokenizer.json"), str(d / f"{q}_v2.yui"), q)                   # v2 (f32: v1)
        write_yui(sd, cfg, str(out / "tokenizer.json"), str(d / f"{q}_v3.yui"), q, meta=dict(T.META_IN))
        files[q] = (str(d / f"{q}_v2.yui"), str(d / f"{q}_v3.yui"))
    sd2, _ = T.make_sd(str(d / "tok2.json"))
    rng = np.random.default_rng(7)
    sd2 = {k: (v + rng.standard_normal(v.shape).astype(np.float32) * 0.02) if v.ndim == 2 else v for k, v in sd.items()}   # trọng số khác
    write_yui(sd2, cfg, str(out / "tokenizer.json"), str(d / "q4_other.yui"), "q4")
    return dict(exe=exe, out=str(out), files=files, other=str(d / "q4_other.yui"), dir=d, cfg=cfg)


def verify(env, model, *extra):
    r = subprocess.run([sys.executable, os.path.join(HERE, "verify_cpp.py"), "--out", env["out"], "--model", model, "--yui", env["exe"],
                        "--skip_logits", *extra], capture_output=True, text=True, timeout=600)
    return r.returncode, r.stdout + r.stderr


@pytest.mark.parametrize("q", ["q4", "t2", "f32"])
def test_v3_giong_het_v2(env, q):
    v2, v3 = env["files"][q]
    rc, out = verify(env, v3, "--compare", v2)
    assert rc == 0, out
    assert "KẾT QUẢ: ĐẠT" in out and "v3" in out and "[tokenizer]" in out and "lệch 0" in out
    assert out.count("max|diff| = 0.00e+00") == 4          # k=1,2 × nhanh/exact


def test_trong_so_khac_thi_khong_dat(env):
    rc, out = verify(env, env["files"]["q4"][1], "--compare", env["other"])
    assert rc == 1 and "KHÔNG ĐẠT" in out and "mong đợi giống hệt" in out


def test_khac_quant_chi_canh_bao(env):
    rc, out = verify(env, env["files"]["q4"][1], "--compare", env["files"]["t2"][1])
    assert "CẢNH BÁO" in out and "KHÔNG kỳ vọng" in out
    assert rc == 0, out


def test_config_json_lech_header(env, tmp_path):
    bad = tmp_path / "bad_dir"
    shutil.copytree(env["out"], bad)
    cfg = dict(env["cfg"], d=env["cfg"]["d"] * 2)
    json.dump(cfg, open(bad / "config.json", "w"))
    e2 = dict(env, out=str(bad))
    rc, out = verify(e2, env["files"]["q4"][1])
    assert rc == 1 and "config.json d=" in out


def test_file_v3_hong_bi_bao_loi(env, tmp_path):
    raw = open(env["files"]["q4"][1], "rb").read()
    p = tmp_path / "cut.yui"
    p.write_bytes(raw[:len(raw) - 10])
    rc, out = verify(env, str(p))
    assert rc == 1 and "KHÔNG ĐẠT" in out and "LỖI" in out


def test_compare_file_hong(env, tmp_path):
    raw = open(env["files"]["q4"][1], "rb").read()
    p = tmp_path / "cut.yui"
    p.write_bytes(raw[:len(raw) // 2])
    rc, out = verify(env, env["files"]["q4"][1], "--compare", str(p))
    assert rc == 1 and "file --compare hỏng" in out
