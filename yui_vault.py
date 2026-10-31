#!/usr/bin/env python3
"""
yui_vault.py - THƯ MỤC LÀM VIỆC của model Yui: lịch sử chat, ghi chú, skill, file model tạo, tiến độ, nhật ký lỗi. Chỉ dùng thư viện chuẩn.

BỐ CỤC (người dùng chọn/nhập MỘT đường dẫn gốc, vd  .../chat-yui):
  chat/    lịch sử chat (.json .jsonl .txt .md). Người dùng chỉ cần chép/dán file vào đây (vd khi đổi điện thoại); lần mở sau tự nạp lại.
  notes/   ghi chú (.md) do model hoặc người dùng viết
  skills/  skill (.md) - chép skill vào đây là dùng được
  kb/      tài liệu cá nhân (.txt .md .html .docx .csv ...) - chạy `mem kb` để lọc rác, cô đọng và nạp (xem yui_kb.py); kb/_clean + kb/_tomtat do hệ thống tạo
  files/   file model tạo ra khi làm việc (đây là thư mục duy nhất tool read/write của model nhìn thấy)
  index/   chỉ mục tìm kiếm + đồ thị (xoá đi cũng được, tự dựng lại từ chat/ và notes/)
  runs/    tiến độ của từng lượt làm việc nhiều bước (tạm thời, xoá khi xong; giữ lại nếu lỗi)
  logs/    errors.jsonl - nhật ký lỗi (luôn giữ)

AN TOÀN ĐƯỜNG DẪN: mọi truy cập đi qua Workspace.safe(): chỉ trong thư mục gốc, chặn "..", đường tuyệt đối và symlink chui ra ngoài.

MÃ HOÁ (tuỳ chọn, bật bằng mật khẩu/khoá: biến môi trường YUI_KEY hoặc file khoá YUI_KEY_FILE):
  khoá  = scrypt(mật khẩu, salt riêng của kho) -> 64 byte (32 byte mã hoá + 32 byte MAC)
  mã    = dòng khoá SHAKE-256(khoá||nonce ngẫu nhiên) XOR dữ liệu, rồi HMAC-SHA256 (mã hoá-rồi-xác thực: file bị sửa/hỏng thì từ chối đọc)
  File đã mã hoá có đuôi .enc (vd chat/yui-20261030.jsonl.enc); ai không có khoá chỉ thấy dữ liệu ngẫu nhiên.
  LƯU Ý THẬT: (1) mất khoá = mất dữ liệu, không có cách khôi phục; (2) khoá nằm ở chỗ chạy app, model đọc được vì app giải mã trong RAM rồi đưa cho model -
  đây là mã hoá khi lưu trữ/sao lưu (kho bị chép đi thì không đọc được), không phải chặn người đang dùng máy có khoá;
  (3) dùng thư viện chuẩn nên KHÔNG thay cho AES-GCM của thư viện chuyên dụng nếu bạn cần chuẩn tuân thủ.
  File chưa mã hoá do người dùng thả vào vẫn đọc được; lệnh seal() sẽ mã hoá hết rồi xoá bản gốc.
"""
import base64
import hashlib
import hmac
import json
import os
import re
import secrets
import tempfile
import time

MAGIC = b"YUI1"
SUBDIRS = ("chat", "notes", "skills", "kb", "files", "index", "runs", "logs")
MAX_FILE = 25 * 1024 * 1024
TEXT_EXT = (".json", ".jsonl", ".txt", ".md")


class VaultError(ValueError):
    pass


def _xor(data, ks):
    n = len(data)
    return (int.from_bytes(data, "big") ^ int.from_bytes(ks, "big")).to_bytes(n, "big") if n else b""


class Cipher:
    def __init__(self, passphrase, salt):
        if isinstance(passphrase, str):
            passphrase = passphrase.encode("utf-8")
        if len(passphrase) < 6:
            raise VaultError("mật khẩu/khoá quá ngắn (tối thiểu 6 ký tự, nên dài hơn)")
        dk = hashlib.scrypt(passphrase, salt=salt, n=2 ** 15, r=8, p=1, maxmem=96 * 1024 * 1024, dklen=64)
        self.ek, self.mk = dk[:32], dk[32:]

    def encrypt(self, data):
        nonce = os.urandom(16)
        ct = _xor(data, hashlib.shake_256(self.ek + nonce).digest(len(data)))
        body = MAGIC + nonce + ct
        return body + hmac.new(self.mk, body, hashlib.sha256).digest()

    def decrypt(self, blob):
        if len(blob) < 4 + 16 + 32 or blob[:4] != MAGIC:
            raise VaultError("không phải file mã hoá của Yui")
        body, tag = blob[:-32], blob[-32:]
        if not hmac.compare_digest(hmac.new(self.mk, body, hashlib.sha256).digest(), tag):
            raise VaultError("sai khoá hoặc file đã bị sửa/hỏng")
        nonce, ct = body[4:20], body[20:]
        return _xor(ct, hashlib.shake_256(self.ek + nonce).digest(len(ct)))


def load_key(passphrase=None, key_env="YUI_KEY", key_file_env="YUI_KEY_FILE"):
    """Mật khẩu theo thứ tự: tham số -> biến môi trường -> file khoá. Trả None nếu không có (kho không mã hoá)."""
    if passphrase:
        return passphrase
    v = os.environ.get(key_env)
    if v:
        return v
    kf = os.environ.get(key_file_env)
    if kf and os.path.isfile(kf):
        return open(kf, encoding="utf-8").read().strip() or None
    return None


def atomic_write(path, data):
    d = os.path.dirname(path)
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, prefix=".tmp_")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def slug(name, default="ghi-chu", maxlen=48):
    import unicodedata
    s = unicodedata.normalize("NFD", str(name).lower().replace("đ", "d"))
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9]+", "-", s).strip("-")[:maxlen].strip("-")
    return s or default


class Workspace:
    def __init__(self, root, passphrase=None, create=True):
        self.root = os.path.realpath(os.path.expanduser(str(root)))
        if create:
            for d in SUBDIRS:
                os.makedirs(os.path.join(self.root, d), exist_ok=True)
        elif not os.path.isdir(self.root):
            raise VaultError(f"không có thư mục {root}")
        self.cipher = None
        self._open_vault(load_key(passphrase))

    # ---------------- khoá ----------------
    def _open_vault(self, pw):
        meta = os.path.join(self.root, ".vault.json")
        if os.path.exists(meta):
            m = json.load(open(meta, encoding="utf-8"))
            if not pw:
                raise VaultError("kho này đã mã hoá: cần đặt YUI_KEY (hoặc YUI_KEY_FILE) đúng mật khẩu")
            self.cipher = Cipher(pw, base64.b64decode(m["salt"]))
            try:
                ok = self.cipher.decrypt(base64.b64decode(m["check"])) == b"yui-vault-ok"
            except VaultError:
                ok = False
            if not ok:
                self.cipher = None
                raise VaultError("sai mật khẩu/khoá của kho")
        elif pw:
            salt = os.urandom(16)
            self.cipher = Cipher(pw, salt)
            atomic_write(meta, json.dumps({"v": 1, "salt": base64.b64encode(salt).decode(),
                                           "check": base64.b64encode(self.cipher.encrypt(b"yui-vault-ok")).decode()}).encode())
            try:
                os.chmod(meta, 0o600)
            except OSError:
                pass

    @property
    def encrypted(self):
        return self.cipher is not None

    # ---------------- đường dẫn an toàn ----------------
    def safe(self, rel, sub=None):
        """Đường dẫn tuyệt đối đã kiểm tra nằm trong kho (hoặc trong thư mục con `sub`)."""
        rel = str(rel).replace("\\", "/").strip()
        if not rel or rel.startswith("/") or "\x00" in rel or re.match(r"^[A-Za-z]:", rel):
            raise VaultError("đường dẫn không hợp lệ")
        base = os.path.realpath(os.path.join(self.root, sub)) if sub else self.root
        full = os.path.realpath(os.path.join(base, rel))
        if full != base and not full.startswith(base + os.sep):
            raise VaultError("chỉ được truy cập trong thư mục đã thiết lập")
        return full

    def rel(self, full):
        return os.path.relpath(full, self.root).replace(os.sep, "/")

    # ---------------- đọc / ghi ----------------
    def read_bytes(self, full):
        if os.path.getsize(full) > MAX_FILE:
            raise VaultError("file quá lớn")
        data = open(full, "rb").read()
        if full.endswith(".enc"):
            if not self.cipher:
                raise VaultError("file mã hoá nhưng chưa có khoá")
            data = self.cipher.decrypt(data)
        return data

    def read_text(self, full):
        return self.read_bytes(full).decode("utf-8", "replace")

    def write_text(self, rel, text, sub=None, plain=False):
        """Ghi file text (mã hoá nếu kho có khoá, trừ khi plain=True). Trả đường dẫn tương đối thực tế đã ghi."""
        return self.write_bytes(rel, text.encode("utf-8"), sub, plain)

    def write_bytes(self, rel, data, sub=None, plain=False):
        full = self.safe(rel, sub)
        if len(data) > MAX_FILE:
            raise VaultError("nội dung quá lớn")
        if self.cipher and not plain:
            full += ".enc" if not full.endswith(".enc") else ""
            data = self.cipher.encrypt(data)
        atomic_write(full, data)
        return self.rel(full)

    def find(self, rel, sub=None):
        """Tìm file theo tên (có hoặc không có đuôi .enc). Trả đường dẫn tuyệt đối hoặc None."""
        full = self.safe(rel, sub)
        for cand in (full, full + ".enc") if not full.endswith(".enc") else (full, full[:-4]):
            if os.path.isfile(cand):
                return cand
        return None

    def list_files(self, sub, exts=TEXT_EXT, limit=50000):
        """[(đường dẫn tuyệt đối, mtime, size)] của file trong thư mục con (đệ quy), bỏ file ẩn/tạm và symlink chui ra ngoài."""
        out, base = [], os.path.join(self.root, sub)
        for r, ds, fs in os.walk(base):
            ds[:] = [d for d in ds if not d.startswith(".")]
            for f in fs:
                if f.startswith("."):
                    continue
                name = f[:-4] if f.endswith(".enc") else f
                if not name.lower().endswith(exts):
                    continue
                p = os.path.join(r, f)
                rp = os.path.realpath(p)
                if not rp.startswith(self.root + os.sep) or not os.path.isfile(rp):
                    continue
                try:
                    st = os.stat(rp)
                except OSError:
                    continue
                out.append((p, st.st_mtime, st.st_size))
                if len(out) >= limit:
                    return out
        return sorted(out)

    # ---------------- nhật ký lỗi ----------------
    def log_error(self, where, message, **extra):
        """Nhật ký lỗi (luôn giữ). KHÔNG ghi nội dung chat của người dùng, chỉ nơi lỗi + thông điệp ngắn. Tự xoay vòng khi > 1MB."""
        try:
            p = os.path.join(self.root, "logs", "errors.jsonl")
            os.makedirs(os.path.dirname(p), exist_ok=True)
            if os.path.exists(p) and os.path.getsize(p) > 1_000_000:
                os.replace(p, p + ".1")
            rec = {"ts": round(time.time(), 3), "at": time.strftime("%Y-%m-%d %H:%M:%S"), "where": str(where)[:60], "msg": str(message)[:300], **{k: str(v)[:120] for k, v in extra.items()}}
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def recent_errors(self, n=10):
        p = os.path.join(self.root, "logs", "errors.jsonl")
        try:
            return [json.loads(x) for x in open(p, encoding="utf-8").read().splitlines()[-n:] if x.strip()]
        except (OSError, ValueError):
            return []

    # ---------------- mã hoá hàng loạt ----------------
    def seal(self, subs=("chat", "notes", "skills")):
        """Mã hoá mọi file text chưa mã hoá trong chat/notes/skills rồi xoá bản gốc. Trả số file đã mã hoá."""
        if not self.cipher:
            raise VaultError("chưa có khoá: đặt YUI_KEY trước")
        n = 0
        for sub in subs:
            for p, _, _ in self.list_files(sub):
                if p.endswith(".enc"):
                    continue
                atomic_write(p + ".enc", self.cipher.encrypt(open(p, "rb").read()))
                os.unlink(p)
                n += 1
        return n


def new_id(prefix=""):
    return prefix + time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
