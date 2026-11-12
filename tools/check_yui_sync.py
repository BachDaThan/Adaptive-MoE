#!/usr/bin/env python3
"""Kiểm tra yui.cpp ở Adaptive-MoE và Y-App phải giống hệt nhau (so sha256).

Vị trí file:  Adaptive-MoE/yui.cpp   |   Y-App/native/yui.cpp
Một bản script duy nhất, đặt giống hệt ở cả 2 repo (tools/check_yui_sync.py).

Cách dùng:
  python tools/check_yui_sync.py                 # so với bản trên GitHub (nhánh main) của repo kia
  python tools/check_yui_sync.py --other PATH    # so với file cục bộ (dùng khi có sẵn cả 2 repo)
  python tools/check_yui_sync.py --strict        # mất mạng cũng tính là lỗi

Mã thoát: 0 = khớp (hoặc không có mạng và không --strict) · 1 = LỆCH · 2 = lỗi khác.
Lưu ý: nếu bạn sửa yui.cpp ở cả 2 repo trong cùng một lần push, CI repo đầu tiên có thể
đỏ tạm thời vì repo kia chưa cập nhật; push nốt repo kia rồi chạy lại là xanh.
"""
import argparse
import hashlib
import sys
import urllib.error
import urllib.request
from pathlib import Path

OWNER = "BachDaThan"
# (đường dẫn trong repo hiện tại) -> (repo kia, đường dẫn trong repo kia)
LAYOUTS = {
    "Adaptive-MoE": ("yui.cpp", "Y-App", "native/yui.cpp"),
    "Y-App": ("native/yui.cpp", "Adaptive-MoE", "yui.cpp"),
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def detect_repo(root: Path):
    """Nhận ra đang đứng ở repo nào theo vị trí yui.cpp."""
    for name, (local, _, _) in LAYOUTS.items():
        if (root / local).is_file():
            return name
    return None


def fetch_remote(repo: str, path: str, ref: str, timeout: float = 30.0) -> bytes:
    url = f"https://raw.githubusercontent.com/{OWNER}/{repo}/{ref}/{path}"
    req = urllib.request.Request(url, headers={"User-Agent": "yui-sync-check"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=".", help="gốc repo hiện tại (mặc định: thư mục hiện tại)")
    ap.add_argument("--other", help="đường dẫn file yui.cpp của repo kia (bỏ qua việc tải GitHub)")
    ap.add_argument("--ref", default="main", help="nhánh/tag của repo kia (mặc định: main)")
    ap.add_argument("--strict", action="store_true", help="không có mạng cũng báo lỗi")
    a = ap.parse_args(argv)

    root = Path(a.root).resolve()
    me = detect_repo(root)
    if me is None:
        print("LỖI: không thấy yui.cpp (cần Adaptive-MoE/yui.cpp hoặc Y-App/native/yui.cpp)", file=sys.stderr)
        return 2
    local_rel, other_repo, other_rel = LAYOUTS[me]
    mine = (root / local_rel).read_bytes()
    h_me = sha256_bytes(mine)
    print(f"[{me}] {local_rel}: {len(mine)} byte, sha256 {h_me}")

    if a.other:
        theirs = Path(a.other).read_bytes()
        src = a.other
    else:
        try:
            theirs = fetch_remote(other_repo, other_rel, a.ref)
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            msg = f"không tải được {other_repo}/{other_rel}: {e}"
            if a.strict:
                print("LỖI: " + msg, file=sys.stderr)
                return 2
            print("CẢNH BÁO: " + msg + " (bỏ qua; dùng --strict để coi là lỗi)")
            return 0
        src = f"{other_repo}@{a.ref}"
    h_other = sha256_bytes(theirs)
    print(f"[{src}] {other_rel}: {len(theirs)} byte, sha256 {h_other}")

    if h_me != h_other:
        print(f"LỆCH: yui.cpp ở {me} khác {other_repo}. Hãy chép bản mới nhất sang repo kia.", file=sys.stderr)
        return 1
    print("OK: hai bản yui.cpp giống hệt nhau.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
