"""Thử make_manifest.py: dựng thư mục .yui nhỏ ngẫu nhiên (không cần PyTorch), sinh manifest, kiểm nội dung, file hỏng, trùng id, sha256."""
import hashlib
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from export_yui import write_yui  # noqa: E402
from test_yui_format import make_sd, META_IN  # noqa: E402
import make_manifest as mm  # noqa: E402
from yui_format import read_yui, arch_id  # noqa: E402

BASE = "https://huggingface.co/BachDaThan/Yui-Models/resolve/main/"


@pytest.fixture()
def tree(tmp_path):
    sd, cfg = make_sd(str(tmp_path / "tok.json"))
    tok = str(tmp_path / "tok.json")
    root = tmp_path / "hub"
    (root / "chuyen nganh").mkdir(parents=True)
    write_yui(sd, cfg, tok, str(root / "yui-q4.yui"), "q4", meta=dict(META_IN, name="yui-q4"))
    write_yui(sd, cfg, tok, str(root / "chuyen nganh" / "code-q4all.yui"), "q4all",
              meta=dict(META_IN, name="yui-code", domains=["code"], lang=["en"], license="MIT"))
    write_yui(sd, cfg, tok, str(root / "cu_v2.yui"), "t2")            # v2: không META
    return root


def build(root, **kw):
    return mm.build_manifest(str(root), BASE, "2026-11-13", **kw)


def test_manifest_dung_noi_dung(tree):
    m, skipped = build(tree)
    assert skipped == []
    mm.validate_manifest(m)
    by = {e["id"]: e for e in m["models"]}
    assert sorted(by) == ["cu_v2", "yui-code", "yui-q4"]            # đã sắp theo id
    e = by["yui-q4"]
    p = tree / "yui-q4.yui"
    assert e["sha256"] == hashlib.sha256(p.read_bytes()).hexdigest()
    assert e["bytes"] == p.stat().st_size and e["yui_ver"] == 3 and e["quant"] == "q4"
    assert e["domains"] == ["chat", "general"] and e["lang"] == ["vi", "en"] and e["license"] == META_IN["license"]
    assert e["url"] == BASE + "yui-q4.yui"
    info = read_yui(str(p))
    assert e["tokenizer_sha256"] == info["tokenizer_sha256"] and e["params"] == info["params"] and e["ctx"] == info["cfg"]["L"]
    assert e["arch_id"] == arch_id(info["cfg"]) and e["created"] == "2026-11-13"


def test_q4all_va_thu_muc_con_co_dau_cach(tree):
    by = {e["id"]: e for e in build(tree)[0]["models"]}
    e = by["yui-code"]
    assert e["quant"] == "q4all" and e["domains"] == ["code"] and e["license"] == "MIT"
    assert e["file"] == "chuyen nganh/code-q4all.yui"
    assert e["url"] == BASE + "chuyen%20nganh/code-q4all.yui"       # đã mã hoá URL


def test_file_v2_khong_co_meta(tree):
    e = {x["id"]: x for x in build(tree)[0]["models"]}["cu_v2"]
    assert e["yui_ver"] == 2 and e["quant"] == "t2" and e["domains"] == [] and e["lang"] == [] and e["license"] == "unknown"
    assert e["name"] == "cu_v2" and "description" not in e


def test_file_hong_bi_bo_qua(tree):
    raw = (tree / "yui-q4.yui").read_bytes()
    (tree / "hong.yui").write_bytes(raw[:len(raw) // 2])
    (tree / "rac.yui").write_bytes(b"khong phai yui")
    m, skipped = build(tree)
    assert sorted(s[0] for s in skipped) == ["hong.yui", "rac.yui"]
    assert len(m["models"]) == 3


def test_strict_va_thu_muc_rong(tree, tmp_path, capsys):
    (tree / "hong.yui").write_bytes(b"YUI1xx")
    out = tmp_path / "m.json"
    assert mm.main([str(tree), "--out", str(out), "--base-url", BASE, "--strict"]) == 1
    assert not out.exists()                                          # strict: không ghi gì
    assert mm.main([str(tree), "--out", str(out), "--base-url", BASE]) == 0
    assert out.exists()
    empty = tmp_path / "trong"
    empty.mkdir()
    assert mm.main([str(empty), "--out", str(tmp_path / "x.json")]) == 1


def test_trung_id(tree):
    sd, cfg = make_sd(str(tree.parent / "t2.json"))
    write_yui(sd, cfg, str(tree.parent / "t2.json"), str(tree / "ban-sao.yui"), "q4", meta=dict(META_IN, name="yui-q4"))
    with pytest.raises(mm.ManifestError, match="trùng id"):
        build(tree)


def test_base_url_phai_https(tree):
    with pytest.raises(mm.ManifestError):
        mm.build_manifest(str(tree), "http://x.org/", "2026-11-13")
    assert mm.norm_base_url("https://a.b/c") == "https://a.b/c/"


def test_ket_qua_lap_lai_duoc(tree, tmp_path):
    a, b = tmp_path / "a.json", tmp_path / "b.json"
    for o in (a, b):
        assert mm.main([str(tree), "--out", str(o), "--base-url", BASE, "--generated", "2026-11-13"]) == 0
    assert a.read_bytes() == b.read_bytes()
    assert json.loads(a.read_text(encoding="utf-8"))["models"][0]["id"] == "cu_v2"


def test_check_phat_hien_lech(tree, tmp_path):
    out = tmp_path / "m.json"
    assert mm.main([str(tree), "--out", str(out), "--base-url", BASE]) == 0
    assert mm.main([str(tree), "--check", str(out)]) == 0
    raw = bytearray((tree / "yui-q4.yui").read_bytes())
    raw[-30] ^= 0xFF                                                 # đổi 1 byte, cùng dung lượng
    (tree / "yui-q4.yui").write_bytes(bytes(raw))
    assert mm.main([str(tree), "--check", str(out)]) == 1
    (tree / "cu_v2.yui").unlink()
    m = json.loads(out.read_text(encoding="utf-8"))
    pr = mm.check_manifest(m, str(tree))
    assert any("không có file" in p for p in pr) and any("sha256" in p for p in pr)


@pytest.mark.parametrize("mut", [
    lambda m: m.update(manifest_version=2),
    lambda m: m.update(base_url="http://x/"),
    lambda m: m["models"][0].update(sha256="ABC"),
    lambda m: m["models"][0].update(url="http://x/a.yui"),
    lambda m: m["models"][0].update(url="https://x/a.exe"),
    lambda m: m["models"][0].update(file="../ngoai.yui"),
    lambda m: m["models"][0].update(file="/tuyet-doi.yui"),
    lambda m: m["models"][0].update(bytes=0),
    lambda m: m["models"][0].update(bytes=True),
    lambda m: m["models"][0].update(yui_ver=9),
    lambda m: m["models"][0].pop("license"),
    lambda m: m["models"][1].update(id=m["models"][0]["id"]),
    lambda m: m.update(models="x"),
])
def test_validate_tu_choi_manifest_sai(tree, mut):
    m, _ = build(tree)
    mut(m)
    with pytest.raises(mm.ManifestError):
        mm.validate_manifest(m)
