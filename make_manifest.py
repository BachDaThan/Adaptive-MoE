#!/usr/bin/env python3
"""
make_manifest.py - quét thư mục chứa file .yui, đọc + kiểm từng file bằng yui_format.py, rồi ghi manifest.json cho Kho model (P2).
Chỉ dùng thư viện chuẩn (không cần torch/numpy).

Cách xếp trên repo model (vd BachDaThan/Yui-Models): manifest.json nằm ở GỐC repo, các file .yui nằm cùng cấp hoặc trong thư mục con.
Thư mục quét (DIR) phải có CÙNG cấu trúc với gốc repo, vì đường dẫn tương đối của file được nối vào --base-url để ra link tải.

    python make_manifest.py DIR                           # ghi DIR/manifest.json
    python make_manifest.py DIR --out manifest.json --base-url https://huggingface.co/BachDaThan/Yui-Models/resolve/main/
    python make_manifest.py DIR --strict                  # có file hỏng thì KHÔNG ghi và thoát mã 1 (mặc định: bỏ qua file hỏng, cảnh báo)
    python make_manifest.py DIR --check DIR/manifest.json # kiểm manifest còn khớp file thật (dung lượng + sha256)

Mã thoát: 0 = OK · 1 = có lỗi (file hỏng khi --strict, không có model nào, trùng id, manifest không khớp) · 2 = tham số sai.

Định dạng manifest (manifest_version 1):
  {"manifest_version": 1, "generated": "YYYY-MM-DD", "base_url": "https://.../resolve/main/",
   "models": [{"id","name","file","url","bytes","sha256","yui_ver","quant","params","arch_id","tokenizer_sha256",
               "domains":[...],"lang":[...],"license","ctx","description"?,"created"?,"base"?}]}
File v1/v2 không có META: id = tên file bỏ đuôi, domains/lang = [], license = "unknown".
Manifest chỉ là DỮ LIỆU để hiển thị và tải: app không chạy/mở bất cứ thứ gì lấy từ manifest ngoài link tải .yui (https) và sha256 để kiểm.
"""
import argparse
import datetime
import hashlib
import json
import os
import re
import sys
import urllib.parse

from yui_format import YuiFormatError, arch_id, read_yui

MANIFEST_VERSION = 1
DEFAULT_BASE_URL = "https://huggingface.co/BachDaThan/Yui-Models/resolve/main/"
MAX_MODELS = 500                      # trần số model trong 1 manifest (app cũng nên từ chối nhiều hơn)
_RE_SHA = re.compile(r"^[0-9a-f]{64}$")
_RE_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_RE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


class ManifestError(ValueError):
    pass


def sha256_file(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def norm_base_url(url):
    if not isinstance(url, str) or not url.startswith("https://") or len(url) <= len("https://"):
        raise ManifestError("base_url phải bắt đầu bằng https://")
    return url if url.endswith("/") else url + "/"


def _slug_id(s):
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s.strip()).strip("-.")
    if not _RE_ID.match(s):
        raise ManifestError(f"không tạo được id hợp lệ từ '{s}'")
    return s


def find_yui(root):
    """Mọi file *.yui dưới root (đệ quy, bỏ thư mục ẩn), -> danh sách đường dẫn tương đối dạng posix, đã sắp xếp."""
    out = []
    for dp, dn, fn in os.walk(root):
        dn[:] = sorted(d for d in dn if not d.startswith("."))
        for f in sorted(fn):
            if f.lower().endswith(".yui") and not f.startswith("."):
                out.append(os.path.relpath(os.path.join(dp, f), root).replace(os.sep, "/"))
    return sorted(out)


def entry_for(root, rel, base_url):
    """Đọc + kiểm toàn bộ 1 file .yui rồi dựng mục manifest. YuiFormatError nếu file hỏng/lạ."""
    path = os.path.join(root, *rel.split("/"))
    info = read_yui(path, with_tensors=False)
    meta = info.get("meta")
    cfg = info["cfg"]
    stem = os.path.basename(rel)[:-4]
    name = meta["name"] if meta else stem
    e = {
        "id": _slug_id(name),
        "name": name,
        "file": rel,
        "url": base_url + urllib.parse.quote(rel, safe="/"),
        "bytes": info["file_size"],
        "sha256": sha256_file(path),
        "yui_ver": info["ver"],
        "quant": meta["quant"] if meta else info["quant_name"],
        "params": info["params"],
        "arch_id": meta["arch_id"] if meta else arch_id(cfg),
        "tokenizer_sha256": info["tokenizer_sha256"],
        "domains": list(meta["domains"]) if meta else [],
        "lang": list(meta["lang"]) if meta else [],
        "license": meta["license"] if meta else "unknown",
        "ctx": cfg["L"],
    }
    if meta:
        for k in ("description", "created", "base"):
            if meta.get(k):
                e[k] = meta[k]
    return e


def build_manifest(root, base_url=DEFAULT_BASE_URL, generated=None):
    """-> (manifest dict, [(file, lý do bỏ qua)]). Trùng id giữa 2 file -> ManifestError (không đoán bừa)."""
    base_url = norm_base_url(base_url)
    models, skipped, seen = [], [], {}
    for rel in find_yui(root):
        try:
            e = entry_for(root, rel, base_url)
        except (YuiFormatError, ManifestError, OSError) as ex:
            skipped.append((rel, str(ex)))
            continue
        if e["id"] in seen:
            raise ManifestError(f"trùng id '{e['id']}' giữa '{seen[e['id']]}' và '{rel}' (đổi --name khi xuất hoặc đổi tên file)")
        seen[e["id"]] = rel
        models.append(e)
    models.sort(key=lambda m: m["id"])
    m = {"manifest_version": MANIFEST_VERSION,
         "generated": generated or datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d"),
         "base_url": base_url, "models": models}
    return m, skipped


def validate_manifest(m):
    """Kiểm cấu trúc manifest (dùng cho test và làm đặc tả cho hub_service.dart). Sai -> ManifestError."""
    def bad(msg):
        raise ManifestError(msg)
    if not isinstance(m, dict):
        bad("manifest phải là đối tượng JSON")
    if m.get("manifest_version") != MANIFEST_VERSION:
        bad(f"manifest_version {m.get('manifest_version')!r} chưa hỗ trợ (biết {MANIFEST_VERSION})")
    if not (isinstance(m.get("generated"), str) and _RE_DATE.match(m["generated"])):
        bad("generated phải dạng YYYY-MM-DD")
    norm_base_url(m.get("base_url"))
    ms = m.get("models")
    if not isinstance(ms, list) or len(ms) > MAX_MODELS:
        bad(f"models phải là mảng tối đa {MAX_MODELS} phần tử")
    ids = set()
    for i, e in enumerate(ms):
        w = f"models[{i}]"
        if not isinstance(e, dict):
            bad(f"{w} phải là đối tượng")
        for k, t in (("id", str), ("name", str), ("file", str), ("url", str), ("sha256", str), ("quant", str), ("arch_id", str),
                     ("tokenizer_sha256", str), ("license", str), ("bytes", int), ("params", int), ("yui_ver", int), ("ctx", int),
                     ("domains", list), ("lang", list)):
            v = e.get(k)
            if not isinstance(v, t) or isinstance(v, bool):
                bad(f"{w}: thiếu hoặc sai kiểu '{k}'")
        if not _RE_ID.match(e["id"]) or e["id"] in ids:
            bad(f"{w}: id không hợp lệ hoặc trùng")
        ids.add(e["id"])
        if not (_RE_SHA.match(e["sha256"]) and _RE_SHA.match(e["tokenizer_sha256"])):
            bad(f"{w}: sha256 phải là 64 ký tự hex thường")
        if e["bytes"] <= 0 or e["params"] <= 0 or e["ctx"] <= 0:
            bad(f"{w}: bytes/params/ctx phải > 0")
        if e["yui_ver"] not in (1, 2, 3):
            bad(f"{w}: yui_ver lạ")
        if not e["url"].startswith("https://") or not e["url"].lower().split("?")[0].endswith(".yui"):
            bad(f"{w}: url phải là https và trỏ tới file .yui")
        if e["file"].startswith("/") or ".." in e["file"].split("/") or "\\" in e["file"]:
            bad(f"{w}: file không được là đường dẫn tuyệt đối hoặc chứa '..'")
        if not all(isinstance(x, str) for x in e["domains"] + e["lang"]):
            bad(f"{w}: domains/lang phải là mảng chuỗi")
    return m


def check_manifest(m, root):
    """So manifest với file thật dưới root -> danh sách vấn đề (rỗng = khớp)."""
    problems = []
    for e in m["models"]:
        p = os.path.join(root, *e["file"].split("/"))
        if not os.path.isfile(p):
            problems.append(f"{e['file']}: không có file")
        elif os.path.getsize(p) != e["bytes"]:
            problems.append(f"{e['file']}: dung lượng {os.path.getsize(p)} ≠ {e['bytes']}")
        elif sha256_file(p) != e["sha256"]:
            problems.append(f"{e['file']}: sha256 không khớp")
    return problems


def main(argv=None):
    ap = argparse.ArgumentParser(description="Sinh/kiểm manifest.json cho Kho model Yui")
    ap.add_argument("dir", help="thư mục chứa các file .yui (cấu trúc giống gốc repo model)")
    ap.add_argument("--out", help="file manifest (mặc định DIR/manifest.json)")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL, help="tiền tố link tải, https, kết thúc bằng /")
    ap.add_argument("--generated", help="ngày YYYY-MM-DD (mặc định hôm nay UTC; đặt cố định để kết quả lặp lại được)")
    ap.add_argument("--strict", action="store_true", help="có file hỏng thì không ghi gì và thoát mã 1")
    ap.add_argument("--check", metavar="MANIFEST", help="chỉ kiểm manifest này còn khớp file trong DIR")
    a = ap.parse_args(argv)
    if not os.path.isdir(a.dir):
        print(f"không thấy thư mục {a.dir}", file=sys.stderr)
        return 2
    try:
        if a.check:
            m = validate_manifest(json.load(open(a.check, encoding="utf-8")))
            pr = check_manifest(m, a.dir)
            for p in pr:
                print("LỆCH:", p)
            print(f"{'KHỚP' if not pr else 'KHÔNG KHỚP'}: {len(m['models'])} model")
            return 1 if pr else 0
        m, skipped = build_manifest(a.dir, a.base_url, a.generated)
        validate_manifest(m)
    except (ManifestError, ValueError, OSError) as ex:
        print(f"LỖI: {ex}", file=sys.stderr)
        return 1
    for rel, why in skipped:
        print(f"BỎ QUA {rel}: {why}", file=sys.stderr)
    if (a.strict and skipped) or not m["models"]:
        print("không ghi manifest: " + ("có file hỏng (--strict)" if skipped else "không có file .yui hợp lệ nào"), file=sys.stderr)
        return 1
    out = a.out or os.path.join(a.dir, "manifest.json")
    with open(out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
        f.write("\n")
    print(f"đã ghi {out}: {len(m['models'])} model" + (f", bỏ qua {len(skipped)} file hỏng" if skipped else ""))
    for e in m["models"]:
        print(f"  {e['id']}: .yui v{e['yui_ver']} {e['quant']}, {e['bytes'] / 1e6:.2f} MB, domains={','.join(e['domains']) or '-'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
