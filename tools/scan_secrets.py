#!/usr/bin/env python3
"""Quét bí mật (khoá API, khoá riêng, token) trong cây làm việc và (tuỳ chọn) toàn bộ lịch sử git.
Chỉ dùng thư viện chuẩn. Thoát mã 1 nếu có trúng. KHÔNG in nội dung bí mật, chỉ in 4 ký tự đầu + vị trí.

  python3 tools/scan_secrets.py            # cây làm việc
  python3 tools/scan_secrets.py --history  # thêm toàn bộ lịch sử mọi nhánh

Đây là lưới thô tự viết, không thay gitleaks/trufflehog. Lộ một lần = coi như hỏng, hãy XOAY KHOÁ."""
import os
import re
import subprocess
import sys

PATTERNS = {
    "khoá riêng": r"-----BEGIN (?:RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY",
    "Hugging Face": r"\bhf_[A-Za-z0-9]{30,}",
    "Google API": r"\bAIza[0-9A-Za-z_\-]{35}",
    "sk-": r"\bsk-[A-Za-z0-9_\-]{20,}",
    "GitHub": r"\b(?:ghp|gho|ghs|ghu|github_pat)_[A-Za-z0-9_]{20,}",
    "Tavily": r"\btvly-[A-Za-z0-9\-_]{16,}",
    "Discord": r"\b[MN][A-Za-z\d]{23}\.[\w-]{6}\.[\w-]{27,}",
    "AWS": r"\bAKIA[0-9A-Z]{16}\b",
    "Slack": r"\bxox[baprs]-[A-Za-z0-9-]{10,}",
    "gán bí mật": r"(?i)(?:api[_-]?key|secret|token|password|passwd)\s*[:=]\s*['\"][A-Za-z0-9_\-/+]{24,}['\"]",
}
SENSITIVE_NAMES = re.compile(r"\.(jks|keystore|p12|pfx|pem|key|env)$|(^|/)(\.env|id_rsa|id_ed25519|key\.properties|google-services\.json)$")
SKIP_DIRS = {".git", "node_modules", "__pycache__", ".dart_tool", "build"}
ALLOW = re.compile(r"scan-secrets:\s*allow")          # dòng có chú thích này được bỏ qua (chuỗi giả trong test)
COMPILED = {k: re.compile(v) for k, v in PATTERNS.items()}
IGNORE = []          # chuỗi con từ tệp .scan-secrets-ignore (mỗi dòng một chuỗi giả đã biết, vd mật khẩu giả trong test)


def load_ignore(root):
    p = os.path.join(root, ".scan-secrets-ignore")
    IGNORE[:] = []
    if os.path.exists(p):
        with open(p, encoding="utf-8") as fh:
            IGNORE.extend(x.strip() for x in fh if x.strip() and not x.startswith("#"))


def scan_text(text, where):
    hits = []
    for n, line in enumerate(text.split("\n"), 1):
        if ALLOW.search(line) or any(s in line for s in IGNORE):
            continue
        for k, rx in COMPILED.items():
            m = rx.search(line)
            if m:
                hits.append((where, n, k, m.group(0)[:4] + "…"))
    return hits


def scan_tree(root):
    hits = []
    for d, dirs, files in os.walk(root):
        dirs[:] = [x for x in dirs if x not in SKIP_DIRS]
        for f in files:
            p = os.path.join(d, f)
            rel = os.path.relpath(p, root).replace(os.sep, "/")
            if SENSITIVE_NAMES.search(rel) and not rel.endswith((".env.example", ".env.sample")):
                hits.append((rel, 0, "tên tệp nhạy cảm", f))
            try:
                with open(p, encoding="utf-8") as fh:
                    hits += scan_text(fh.read(), rel)
            except (UnicodeDecodeError, OSError):
                pass
    return hits


def scan_history(root):
    hits = []
    names = subprocess.run(["git", "-C", root, "log", "--all", "--name-only", "--pretty=format:"], capture_output=True, text=True, errors="replace").stdout
    for n in sorted(set(names.split("\n"))):
        if n and SENSITIVE_NAMES.search(n) and not n.endswith((".env.example", ".env.sample")):
            hits.append((n, 0, "tên tệp nhạy cảm (lịch sử)", n))
    log = subprocess.run(["git", "-C", root, "log", "--all", "-p", "--no-color", "-U0"], capture_output=True, text=True, errors="replace").stdout
    return hits + scan_text(log, "lịch sử git")


def main(argv):
    root = os.getcwd()
    load_ignore(root)
    hits = scan_tree(root) + (scan_history(root) if "--history" in argv else [])
    for where, n, kind, preview in hits:
        print(f"TRÚNG {kind}: {where}:{n} ({preview})")
    print(f"{len(hits)} trúng")
    return 1 if hits else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
