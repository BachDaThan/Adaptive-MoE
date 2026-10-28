#!/usr/bin/env python3
"""
yui_agent.py - công cụ (tool), sandbox và MCP cho model của dự án Yui (chỉ dùng thư viện chuẩn của Python).

THIẾT KẾ CHO MODEL NHỎ
  * Model chỉ cần học MỘT cú pháp rất ngắn:   <tool>tên đối_số</tool>   (mỗi công cụ đúng MỘT đối số chuỗi), ví dụ
        <tool>calc 23*17</tool>      <tool>search thủ đô của Nhật Bản</tool>      <tool>time</tool>
    Kết quả được trả lại như một lượt "user":  <result>391</result>  (nên khuôn hội thoại user/assistant của train_sft.py giữ nguyên).
  * Mọi thứ nguy hiểm do KHUNG CHẠY (harness) quyết định, không phải model: danh sách công cụ cho phép, sandbox, danh sách mạng cho phép,
    giới hạn tài nguyên, xác nhận của người dùng, nhật ký (audit). Model 50M hay 7B đều bị ràng buộc như nhau.
  * MCP: kết nối server MCP (stdio hoặc HTTP) rồi "gán" công cụ MCP vào tên công cụ chuẩn (bind) để model nhỏ chỉ cần biết ~6 tên;
    đồng thời có thể chạy chế độ MCP server để các ứng dụng khác dùng công cụ + sandbox của Yui.

SANDBOX (3 bậc, tự dò; công cụ py chỉ chạy khi bậc đủ mạnh theo cấu hình, mặc định "strong")
  strong  : bubblewrap (bwrap): không mạng, không thấy file máy chủ (chỉ /usr, /bin, /lib ở chế độ chỉ-đọc), thư mục làm việc tạm, PID riêng, giới hạn CPU/RAM/thời gian
  medium  : unshare: không mạng + PID riêng + giới hạn tài nguyên, nhưng VẪN đọc được file máy chủ -> không dùng cho mã không tin cậy
  weak    : chỉ giới hạn tài nguyên + thư mục tạm
  Cài bậc strong:  apt install bubblewrap   (Docker/Kaggle: cần cho phép user namespace; nếu không được thì tắt py, các công cụ còn lại không cần sandbox).
  Sandbox này là lớp bảo vệ thực dụng, KHÔNG thay cho container/VM khi chạy mã của người lạ.

DÙNG
  python yui_agent.py status [--config agent.json]
  python yui_agent.py chat --url http://127.0.0.1:11434 [--config agent.json]      # model chạy bằng:  ./yui --model m.yui --serve
  python yui_agent.py mcp-server [--config agent.json]                              # MCP server (stdio)
  python yui_agent.py tool calc "2+3*4"                                             # thử một công cụ
Cấu hình mẫu: agent.example.json. Key Tavily (tuỳ chọn, không bắt buộc): TAVILY_API_KEY1, TAVILY_API_KEY2, ... hoặc TAVILY_API_KEYS.
"""
import argparse
import ast
import datetime
import difflib
import html
import http.client
import ipaddress
import json
import math
import operator
import os
import queue
import re
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

try:
    import resource
except ImportError:  # Windows
    resource = None

# ============================ cấu hình / profile ============================
BASE_LIMITS = dict(cpu_s=5, mem_mb=256, wall_s=10, out_kb=16, file_kb=64, workspace_mb=16, fetch_kb=256, procs=64)
ALL_TOOLS = ["calc", "time", "search", "fetch", "read", "write", "py"]
PRESETS = {
    "minimal": dict(tools=["calc", "time"], net=[], py_min="off", confirm=[]),
    "standard": dict(tools=ALL_TOOLS, net=["vi.wikipedia.org", "api.tavily.com"], py_min="strong", confirm=[]),
    "open": dict(tools=ALL_TOOLS, net=["*"], py_min="medium", confirm=["py", "fetch"]),
}
RANK = {"off": 4, "strong": 3, "medium": 2, "weak": 1}


class Profile:
    """Cấu hình sandbox + công cụ. `scale` co giãn mọi giới hạn tài nguyên (0.5 = nửa, 4 = gấp bốn) mà không đụng tới quyền."""

    def __init__(self, cfg=None):
        cfg = dict(cfg or {})
        base = dict(PRESETS[cfg.get("preset", "standard")])
        self.preset = cfg.get("preset", "standard")
        self.tools = [t for t in cfg.get("tools", base["tools"]) if t in ALL_TOOLS]
        net = cfg.get("net", {})
        self.net_allow = list(net.get("allow", base["net"])) if isinstance(net, dict) else list(net)
        self.allow_private = bool(net.get("allow_private", False)) if isinstance(net, dict) else False   # CHỈ để thử trên máy mình
        self.scale = float(cfg.get("scale", 1.0))
        lim = dict(BASE_LIMITS)
        for k, v in (cfg.get("limits") or {}).items():
            if k in lim:
                lim[k] = v
        self.limits = {k: max(1, int(v * self.scale)) for k, v in lim.items()}
        self.py_min = (cfg.get("python") or {}).get("min_sandbox", base["py_min"])
        self.workspace = os.path.abspath(cfg.get("workspace", "agent_workspace"))
        self.confirm = set(cfg.get("confirm", base["confirm"]))
        self.audit = cfg.get("audit")
        self.max_steps = int(cfg.get("max_steps", 3))
        self.result_chars = int(cfg.get("result_chars", 300))   # kết quả đưa lại cho model: giữ ngắn cho vừa ngữ cảnh (--L) của model
        self.mcp = dict(cfg.get("mcp", {}))
        self.bind = dict(cfg.get("bind", {}))
        self.expose_all_mcp = bool(cfg.get("expose_all_mcp", False))
        self.wiki_base = cfg.get("wiki_base", "https://vi.wikipedia.org")
        self.tavily_base = cfg.get("tavily_base", "https://api.tavily.com")

    @staticmethod
    def load(path=None):
        path = path or os.environ.get("YUI_AGENT_CONFIG")
        if path and os.path.exists(path):
            return Profile(json.load(open(path, encoding="utf-8")))
        return Profile({})


# ============================ sandbox ============================
_SANDBOX = {}


def _probe(cmd):
    try:
        return subprocess.run(cmd, capture_output=True, timeout=10).returncode == 0
    except Exception:  # noqa: BLE001
        return False


def _ro_binds():
    out = []
    for p in ("/usr", "/bin", "/lib", "/lib64", "/sbin"):
        if os.path.exists(p):
            out += ["--ro-bind", p, p]
    return out


def sandbox_tier():
    """'strong' | 'medium' | 'weak' (dò một lần)."""
    if "tier" not in _SANDBOX:
        tier = "weak"
        if sys.platform.startswith("linux"):
            if shutil.which("bwrap") and _probe(["bwrap", "--unshare-all", *_ro_binds(), "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp", "true"]):
                tier = "strong"
            elif shutil.which("unshare") and _probe(["unshare", "-rn", "true"]):
                tier = "medium"
        _SANDBOX["tier"] = tier
    return _SANDBOX["tier"]


def _limits_fn(lim):
    def apply():
        if resource is None:
            return
        os.setsid()
        for res, val in ((resource.RLIMIT_CPU, lim["cpu_s"]), (resource.RLIMIT_AS, lim["mem_mb"] * 1024 * 1024),
                         (resource.RLIMIT_FSIZE, lim["file_kb"] * 1024), (resource.RLIMIT_CORE, 0)):
            try:
                resource.setrlimit(res, (val, val))
            except (ValueError, OSError):
                pass
    return apply


def run_python(code, prof):
    """Chạy mã Python trong sandbox. Trả (ok, văn bản). Từ chối nếu bậc sandbox yếu hơn yêu cầu của profile."""
    tier = sandbox_tier()
    if prof.py_min == "off" or RANK[tier] < RANK[prof.py_min]:
        return False, f"công cụ py bị tắt (sandbox hiện có: {tier}, cấu hình cần: {prof.py_min})"
    lim = prof.limits
    work = tempfile.mkdtemp(prefix="yui_py_")
    try:
        with open(os.path.join(work, "main.py"), "w", encoding="utf-8") as f:
            f.write(code)
        py = "/usr/bin/python3" if os.path.exists("/usr/bin/python3") else sys.executable
        env = {"PATH": "/usr/bin:/bin", "HOME": "/work" if tier == "strong" else work, "PYTHONDONTWRITEBYTECODE": "1", "LANG": "C.UTF-8"}
        if tier == "strong":
            cmd = ["bwrap", "--unshare-all", "--die-with-parent", "--new-session", *_ro_binds(), "--proc", "/proc", "--dev", "/dev",
                   "--tmpfs", "/tmp", "--bind", work, "/work", "--chdir", "/work", "--clearenv"]
            for k, v in env.items():
                cmd += ["--setenv", k, v]
            cmd += [py, "-I", "-B", "main.py"]
            cwd = None
        elif tier == "medium":
            cmd = ["unshare", "-rn", "--fork", "--pid", py, "-I", "-B", "main.py"]
            cwd = work
        else:
            cmd = [py, "-I", "-B", "main.py"]
            cwd = work
        try:
            p = subprocess.Popen(cmd, cwd=cwd, env=env, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                 preexec_fn=_limits_fn(lim) if resource else None, start_new_session=resource is None)
        except OSError as e:
            return False, f"không chạy được sandbox: {e}"
        try:
            out, _ = p.communicate(timeout=lim["wall_s"])
        except subprocess.TimeoutExpired:
            try:
                os.killpg(p.pid, 9)
            except (ProcessLookupError, PermissionError, OSError):
                p.kill()
            p.communicate()
            return False, f"quá {lim['wall_s']}s nên bị dừng"
        text = out[: lim["out_kb"] * 1024].decode("utf-8", "replace")
        return p.returncode == 0, text if p.returncode == 0 else f"lỗi (mã {p.returncode}): {text[-400:]}"
    finally:
        shutil.rmtree(work, ignore_errors=True)


# ============================ công cụ ============================
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod, ast.Pow: operator.pow}
_FUNCS = {"sqrt": math.sqrt, "sin": math.sin, "cos": math.cos, "tan": math.tan, "log": math.log, "log10": math.log10, "exp": math.exp,
          "abs": abs, "round": round, "floor": math.floor, "ceil": math.ceil}
_CONST = {"pi": math.pi, "e": math.e}


def safe_calc(expr):
    """Máy tính an toàn: duyệt cây cú pháp, KHÔNG eval. Chỉ số, + - * / // % ** ( ) và vài hàm toán; chặn luỹ thừa khổng lồ."""
    expr = expr.strip().replace("×", "*").replace("÷", "/").replace("^", "**").replace("−", "-")
    expr = re.sub(r"(?<=\d),(?=\d)", ".", expr)
    if not expr or len(expr) > 200:
        raise ValueError("biểu thức rỗng hoặc quá dài")
    tree = ast.parse(expr, mode="eval")
    count = [0]

    def ev(n):
        count[0] += 1
        if count[0] > 200:
            raise ValueError("biểu thức quá phức tạp")
        if isinstance(n, ast.Expression):
            return ev(n.body)
        if isinstance(n, ast.Constant) and type(n.value) in (int, float):
            return n.value
        if isinstance(n, ast.Name) and n.id in _CONST:
            return _CONST[n.id]
        if isinstance(n, ast.UnaryOp) and isinstance(n.op, (ast.USub, ast.UAdd)):
            v = ev(n.operand)
            return -v if isinstance(n.op, ast.USub) else v
        if isinstance(n, ast.BinOp) and type(n.op) in _OPS:
            a, b = ev(n.left), ev(n.right)
            if isinstance(n.op, ast.Pow):
                if abs(b) > 1000 or (abs(a) > 1 and abs(b) * math.log2(abs(a)) > 4096):
                    raise ValueError("luỹ thừa quá lớn")
            return _OPS[type(n.op)](a, b)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in _FUNCS and not n.keywords and len(n.args) <= 2:
            return _FUNCS[n.func.id](*[ev(a) for a in n.args])
        raise ValueError("biểu thức không hợp lệ")

    v = ev(tree)
    if isinstance(v, float):
        if math.isnan(v) or math.isinf(v):
            raise ValueError("kết quả không xác định")
        if v == int(v) and abs(v) < 1e15:
            return str(int(v))
        return f"{v:.10g}"
    if len(str(v)) > 60:
        raise ValueError("kết quả quá dài")
    return str(v)


_DOW = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]


def tool_time(arg=""):
    m = re.fullmatch(r"\s*(?:utc)?\s*([+-]\d{1,2})?\s*", arg or "", re.I)
    off = int(m.group(1)) if m and m.group(1) else 7   # mặc định giờ Việt Nam
    if not -12 <= off <= 14:
        raise ValueError("múi giờ không hợp lệ")
    now = datetime.datetime.now(datetime.timezone.utc) + datetime.timedelta(hours=off)
    return f"{_DOW[now.weekday()]}, {now:%d/%m/%Y %H:%M} (UTC{off:+d})"


def sanitize(text):
    """Kết quả công cụ là DỮ LIỆU KHÔNG TIN CẬY: bỏ ký tự điều khiển và mọi thẻ <tool>/<result> để nội dung web không thể giả mạo lệnh gọi công cụ."""
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text)
    text = re.sub(r"</?\s*(tool|result)\b[^>]*>?", " ", text, flags=re.I)
    return re.sub(r"\s+", " ", text).strip()


def _host_allowed(host, prof):
    host = (host or "").lower().rstrip(".")
    return any(a == "*" or host == a or host.endswith("." + a.lstrip("*.")) for a in prof.net_allow)


def check_url(url, prof):
    """Chống SSRF: chỉ https (http chỉ khi allow_private để thử), host nằm trong danh sách cho phép, mọi IP phân giải phải là IP công cộng. Trả (host, ip, port, path)."""
    u = urllib.parse.urlsplit(url.strip())
    if u.scheme not in ("https", "http") or (u.scheme == "http" and not prof.allow_private):
        raise ValueError("chỉ cho phép https")
    if not u.hostname or u.username or u.password:
        raise ValueError("địa chỉ không hợp lệ")
    if not _host_allowed(u.hostname, prof):
        raise ValueError(f"máy chủ {u.hostname} không nằm trong danh sách cho phép")
    port = u.port or (443 if u.scheme == "https" else 80)
    try:
        infos = socket.getaddrinfo(u.hostname, port, type=socket.SOCK_STREAM)
    except socket.gaierror as e:
        raise ValueError(f"không phân giải được tên miền ({e})")
    ips = [i[4][0] for i in infos]
    if not ips:
        raise ValueError("không có địa chỉ IP")
    if not prof.allow_private:
        for ip in ips:
            a = ipaddress.ip_address(ip.split("%")[0])
            if not a.is_global or a.is_multicast:
                raise ValueError("địa chỉ nội bộ/riêng bị chặn")
    path = (u.path or "/") + (("?" + u.query) if u.query else "")
    return u.hostname, ips[0], port, path, u.scheme


class _PinnedHTTPS(http.client.HTTPSConnection):
    def __init__(self, host, ip, port, timeout):
        super().__init__(host, port, timeout=timeout, context=ssl.create_default_context())
        self._ip = ip

    def connect(self):   # nối thẳng tới IP đã kiểm tra (chống DNS rebinding), vẫn xác thực chứng chỉ theo tên miền
        sock = socket.create_connection((self._ip, self.port), self.timeout)
        self.sock = self._context.wrap_socket(sock, server_hostname=self.host)


def safe_http(method, url, prof, body=None, headers=None, max_kb=None, timeout=15, redirects=3):
    """GET/POST có kiểm tra SSRF ở MỖI bước chuyển hướng, giới hạn dung lượng. Trả (status, content_type, bytes)."""
    max_kb = max_kb or prof.limits["fetch_kb"]
    for _ in range(redirects + 1):
        host, ip, port, path, scheme = check_url(url, prof)
        conn = _PinnedHTTPS(host, ip, port, timeout) if scheme == "https" else http.client.HTTPConnection(ip, port, timeout=timeout)
        try:
            h = {"Host": host if port in (80, 443) else f"{host}:{port}", "User-Agent": "yui-agent/1.0", "Accept": "*/*", "Accept-Encoding": "identity", **(headers or {})}
            conn.request(method, path, body=body, headers=h)
            r = conn.getresponse()
            if r.status in (301, 302, 303, 307, 308) and r.getheader("Location"):
                url = urllib.parse.urljoin(url, r.getheader("Location"))
                method, body = ("GET", None) if r.status in (301, 302, 303) else (method, body)
                continue
            data = r.read(max_kb * 1024 + 1)[: max_kb * 1024]
            return r.status, r.getheader("Content-Type") or "", data
        finally:
            conn.close()
    raise ValueError("quá nhiều lần chuyển hướng")


def _text_of(ct, data):
    s = data.decode("utf-8", "replace")
    if "html" in ct.lower() or s.lstrip().startswith("<"):
        s = re.sub(r"(?is)<(script|style|noscript)[^>]*>.*?</\1>", " ", s)
        s = re.sub(r"(?s)<[^>]+>", " ", s)
        s = html.unescape(s)
    return s


def tool_fetch(url, prof):
    st, ct, data = safe_http("GET", url, prof)
    if st != 200:
        raise ValueError(f"HTTP {st}")
    if not (ct.lower().startswith("text/") or "json" in ct.lower() or "xml" in ct.lower()):
        raise ValueError(f"loại nội dung không hỗ trợ ({ct or '?'})")
    return _text_of(ct, data)


def _tavily_keys():
    out = []
    for k, v in sorted(os.environ.items()):
        if re.fullmatch(r"TAVILY_API_KEYS?(_?\d+)?", k) and v.strip():
            out += [x for x in re.split(r"[,;\s]+", v) if x]
    return list(dict.fromkeys(out))


def tool_search(q, prof):
    """Tavily nếu có key (xoay vòng), không thì Wikipedia tiếng Việt. Trả đoạn ngắn nhất đủ ý."""
    q = q.strip()[:200]
    if not q:
        raise ValueError("thiếu từ khoá")
    for key in _tavily_keys():
        try:
            st, _, data = safe_http("POST", prof.tavily_base.rstrip("/") + "/search", prof, body=json.dumps({"query": q, "max_results": 3, "search_depth": "basic"}),
                                    headers={"Authorization": "Bearer " + key, "Content-Type": "application/json"}, max_kb=128)
        except ValueError:
            break
        if st == 200:
            res = json.loads(data.decode("utf-8", "replace")).get("results", [])
            txt = " ".join(str(r.get("content", "")) for r in res[:2]).strip()
            if txt:
                return txt
            break
        if st in (401, 403, 432, 433, 429):
            continue   # key hỏng/hết hạn mức -> thử key kế
        break
    url = prof.wiki_base.rstrip("/") + "/w/api.php?" + urllib.parse.urlencode({
        "action": "query", "format": "json", "formatversion": "2", "generator": "search", "gsrsearch": q, "gsrlimit": 1,
        "prop": "extracts", "exintro": 1, "explaintext": 1, "exchars": max(200, prof.result_chars * 2)})
    st, _, data = safe_http("GET", url, prof, max_kb=128)
    if st != 200:
        raise ValueError(f"HTTP {st}")
    pages = json.loads(data.decode("utf-8", "replace")).get("query", {}).get("pages", [])
    if not pages or not pages[0].get("extract"):
        return "không tìm thấy"
    return pages[0]["extract"]


def _safe_path(prof, rel):
    rel = rel.strip().strip("\"'")
    if not rel or os.path.isabs(rel) or "\x00" in rel:
        raise ValueError("đường dẫn không hợp lệ")
    root = os.path.realpath(prof.workspace)
    full = os.path.realpath(os.path.join(root, rel))
    if full != root and not full.startswith(root + os.sep):
        raise ValueError("chỉ được truy cập trong thư mục làm việc")
    return full


def tool_read(arg, prof):
    p = _safe_path(prof, arg)
    if not os.path.isfile(p):
        raise ValueError("không có file này")
    with open(p, "rb") as f:
        return f.read(prof.limits["file_kb"] * 1024).decode("utf-8", "replace")


def tool_write(arg, prof):
    path, sep, text = arg.partition("|")
    if not sep:
        raise ValueError("cú pháp: write đường/dẫn|nội dung")
    if len(text.encode("utf-8")) > prof.limits["file_kb"] * 1024:
        raise ValueError("nội dung quá lớn")
    p = _safe_path(prof, path)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    total = sum(os.path.getsize(os.path.join(r, f)) for r, _, fs in os.walk(prof.workspace) for f in fs)
    if total + len(text) > prof.limits["workspace_mb"] * 1024 * 1024:
        raise ValueError("thư mục làm việc đã đầy")
    with open(p, "w", encoding="utf-8") as f:
        f.write(text)
    return f"đã ghi {len(text)} ký tự"


# ============================ MCP ============================
class McpError(Exception):
    pass


def _jsonrpc_text(result):
    """Gom phần text của kết quả tools/call."""
    parts = [c.get("text", "") for c in (result.get("content") or []) if c.get("type") == "text"]
    text = "\n".join(p for p in parts if p) or json.dumps(result.get("structuredContent", ""), ensure_ascii=False)
    if result.get("isError"):
        raise McpError(text[:200])
    return text


class McpStdio:
    """Client MCP qua stdio (JSON-RPC, mỗi dòng một thông điệp). Tiến trình con chạy với môi trường đã lọc."""

    def __init__(self, cmd, env_pass=(), timeout=30):
        env = {k: os.environ[k] for k in ("PATH", "HOME", "LANG", "SYSTEMROOT", *env_pass) if k in os.environ}
        self.p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, bufsize=1, env=env)
        self.q, self.timeout, self.id = queue.Queue(), timeout, 0
        threading.Thread(target=self._reader, daemon=True).start()
        self.request("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "yui-agent", "version": "1.0"}})
        self._send({"jsonrpc": "2.0", "method": "notifications/initialized"})

    def _reader(self):
        for line in self.p.stdout:
            try:
                self.q.put(json.loads(line))
            except ValueError:
                pass
        self.q.put(None)

    def _send(self, obj):
        self.p.stdin.write(json.dumps(obj) + "\n")
        self.p.stdin.flush()

    def request(self, method, params=None):
        self.id += 1
        rid = self.id
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params or {}})
        end = time.time() + self.timeout
        while True:
            try:
                msg = self.q.get(timeout=max(0.05, end - time.time()))
            except queue.Empty:
                raise McpError(f"{method}: quá thời gian")
            if msg is None:
                raise McpError("server MCP đã thoát")
            if msg.get("id") == rid:
                if "error" in msg:
                    raise McpError(str(msg["error"].get("message", msg["error"]))[:200])
                return msg.get("result", {})

    def list_tools(self):
        return self.request("tools/list").get("tools", [])

    def call(self, name, args):
        return _jsonrpc_text(self.request("tools/call", {"name": name, "arguments": args}))

    def close(self):
        try:
            self.p.stdin.close()
            self.p.terminate()
        except Exception:  # noqa: BLE001
            pass


class McpHttp:
    """Client MCP qua HTTP (Streamable HTTP: POST JSON-RPC, trả JSON hoặc SSE). URL do người dùng tự cấu hình nên được tin cậy."""

    def __init__(self, url, headers=None, timeout=30):
        self.url, self.headers, self.timeout, self.id, self.session = url, dict(headers or {}), timeout, 0, None
        self.request("initialize", {"protocolVersion": "2024-11-05", "capabilities": {}, "clientInfo": {"name": "yui-agent", "version": "1.0"}})
        self._post({"jsonrpc": "2.0", "method": "notifications/initialized"}, expect=False)

    def _post(self, obj, expect=True):
        h = {"Content-Type": "application/json", "Accept": "application/json, text/event-stream", **self.headers}
        if self.session:
            h["Mcp-Session-Id"] = self.session
        req = urllib.request.Request(self.url, data=json.dumps(obj).encode(), headers=h, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                self.session = r.headers.get("Mcp-Session-Id") or self.session
                ct, body = r.headers.get("Content-Type", ""), r.read(4 * 1024 * 1024).decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            raise McpError(f"HTTP {e.code}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            raise McpError(f"không kết nối được: {e}")
        if not expect or not body.strip():
            return None
        msgs = [json.loads(m[5:].strip()) for m in body.splitlines() if m.startswith("data:") and m[5:].strip()] if "event-stream" in ct else [json.loads(body)]
        for m in msgs:
            if isinstance(m, dict) and m.get("id") == obj.get("id"):
                return m
        raise McpError("không có phản hồi khớp")

    def request(self, method, params=None):
        self.id += 1
        m = self._post({"jsonrpc": "2.0", "id": self.id, "method": method, "params": params or {}})
        if "error" in m:
            raise McpError(str(m["error"].get("message", m["error"]))[:200])
        return m.get("result", {})

    def list_tools(self):
        return self.request("tools/list").get("tools", [])

    def call(self, name, args):
        return _jsonrpc_text(self.request("tools/call", {"name": name, "arguments": args}))

    def close(self):
        pass


def _subst(obj, arg):
    if isinstance(obj, str):
        return obj.replace("$arg", arg)
    if isinstance(obj, dict):
        return {k: _subst(v, arg) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_subst(v, arg) for v in obj]
    return obj


# ============================ hộp công cụ ============================
DOCS = {"calc": "calc biểu_thức", "time": "time", "search": "search từ_khoá", "fetch": "fetch https://...", "read": "read tệp", "write": "write tệp|nội_dung", "py": "py mã_python"}


class ToolBox:
    def __init__(self, prof, confirm_cb=None):
        self.prof, self.confirm_cb, self.mcp, self.mcp_tools = prof, confirm_cb, {}, {}
        self.tools = {}
        self._failed = {}
        py_ok = prof.py_min != "off" and RANK[sandbox_tier()] >= RANK[prof.py_min]
        local = {
            "calc": lambda a: safe_calc(a),
            "time": lambda a: tool_time(a),
            "search": lambda a: tool_search(a, prof),
            "fetch": lambda a: tool_fetch(a, prof),
            "read": lambda a: tool_read(a, prof),
            "write": lambda a: tool_write(a, prof),
            "py": lambda a: self._py(a),
        }
        for name in prof.tools:
            if name == "py" and not py_ok:
                self._failed["py"] = f"tắt: sandbox {sandbox_tier()} < {prof.py_min}"
                continue
            self.tools[name] = local[name]
        self._connect_mcp()

    def _py(self, code):
        ok, out = run_python(code, self.prof)
        if not ok:
            raise ValueError(out)
        return out

    def _connect_mcp(self):
        for sname, spec in self.prof.mcp.items():
            try:
                cli = McpHttp(spec["url"], spec.get("headers")) if "url" in spec else McpStdio(spec["command"], spec.get("env_pass", ()))
                self.mcp[sname] = cli
                self.mcp_tools[sname] = {t["name"]: t for t in cli.list_tools()}
            except (McpError, OSError, KeyError, ValueError) as e:
                self._failed[f"mcp:{sname}"] = str(e)[:120]
        for canon, b in self.prof.bind.items():   # gán công cụ MCP vào tên chuẩn mà model nhỏ đã học
            cli, tool = self.mcp.get(b.get("mcp")), b.get("tool")
            if cli is None or tool not in self.mcp_tools.get(b.get("mcp"), {}):
                self._failed[f"bind:{canon}"] = "server/tool MCP không có"
                continue
            tpl = b.get("args", {"arg": "$arg"})
            self.tools[canon] = (lambda a, cli=cli, tool=tool, tpl=tpl: cli.call(tool, _subst(tpl, a)))
        if self.prof.expose_all_mcp:   # cho model lớn: server.tool với đối số JSON
            for sname, tl in self.mcp_tools.items():
                for tname in tl:
                    self.tools[f"{sname}.{tname}"] = (lambda a, c=self.mcp[sname], t=tname: c.call(t, json.loads(a) if a.strip().startswith("{") else {"arg": a}))

    def names(self):
        return sorted(self.tools)

    def menu(self):
        return "Công cụ: " + "; ".join(DOCS.get(n, n) for n in self.names())

    def call(self, name, arg):
        """Chạy một công cụ qua chính sách. Luôn trả chuỗi ngắn đã làm sạch (lỗi cũng là chuỗi), không bao giờ ném lỗi ra cho model."""
        t0 = time.time()
        decision, out = "ok", ""
        if name not in self.tools:
            decision, out = "unknown", f"lỗi: không có công cụ {name[:30]}"
        elif name in self.prof.confirm and not (self.confirm_cb and self.confirm_cb(name, arg)):
            decision, out = "denied", "bị chặn: cần người dùng xác nhận"
        else:
            try:
                out = str(self.tools[name](arg))
            except (ValueError, McpError, OSError, ZeroDivisionError, OverflowError, SyntaxError, json.JSONDecodeError, urllib.error.URLError, ssl.SSLError, http.client.HTTPException, TimeoutError) as e:
                decision, out = "error", f"lỗi: {str(e)[:120]}"
        out = sanitize(out)[: self.prof.result_chars] or "(trống)"
        self._audit(name, arg, decision, out, time.time() - t0)
        return out

    def _audit(self, name, arg, decision, out, dt):
        if not self.prof.audit:
            return
        try:
            with open(self.prof.audit, "a", encoding="utf-8") as f:
                f.write(json.dumps({"ts": round(time.time(), 3), "tool": name, "arg": arg[:200], "decision": decision, "out_len": len(out), "ms": int(dt * 1000)}, ensure_ascii=False) + "\n")
        except OSError:
            pass

    def close(self):
        for c in self.mcp.values():
            c.close()


# ============================ khung chạy agent ============================
CALL_RE = re.compile(r"<tool>(.*?)(?:</tool>|$)", re.S | re.I)


def parse_call(text, names=()):
    """Tách lệnh gọi công cụ. Dễ tính với model nhỏ: chấp nhận thiếu </tool>, dạng JSON, dấu | hoặc : sau tên, tên gõ sai nhẹ (khớp mờ)."""
    m = CALL_RE.search(text)
    if not m:
        return None
    body = m.group(1).strip()
    if not body:
        return None
    name, arg = None, ""
    if body.startswith("{"):
        try:
            o = json.loads(body)
            name = str(o.get("name") or o.get("tool") or "")
            a = o.get("arg", o.get("args", o.get("q", o.get("expr", ""))))
            arg = a if isinstance(a, str) else (next((v for v in a.values() if isinstance(v, str)), "") if isinstance(a, dict) else str(a))
        except ValueError:
            return None
    else:
        mm = re.match(r"^([\w.\-]+)\s*[|:]?\s*(.*)$", body, re.S)
        if not mm:
            return None
        name, arg = mm.group(1), mm.group(2)
    low = name.lower()
    if names and low not in names:
        close = difflib.get_close_matches(low, list(names), n=1, cutoff=0.75)
        low = close[0] if close else low
    return low, arg.strip()


FALLBACK = "Mik chưa trả lời được câu này."


def strip_tool_tags(text):
    return re.sub(r"<tool>.*?(</tool>|$)", "", text, flags=re.S | re.I).strip()


class Agent:
    def __init__(self, llm, toolbox):
        self.llm, self.tb = llm, toolbox

    def run(self, messages):
        """messages: [{'role','content'}...]. Trả (câu trả lời cuối, vết các bước). Tối đa max_steps lần gọi công cụ mỗi lượt."""
        msgs, trace = list(messages), []
        names = set(self.tb.names())
        for step in range(self.tb.prof.max_steps + 1):
            out = self.llm(msgs)
            call = parse_call(out, names) if step < self.tb.prof.max_steps else None
            if call is None:
                return strip_tool_tags(out) or FALLBACK, trace   # không bao giờ đưa thẻ <tool> thô ra cho người dùng
            name, arg = call
            res = self.tb.call(name, arg)
            trace.append({"tool": name, "arg": arg, "result": res})
            msgs += [{"role": "assistant", "content": f"<tool>{name} {arg}</tool>".strip()}, {"role": "user", "content": f"<result>{res}</result>"}]
        return FALLBACK, trace


class OpenAICompat:
    """Gọi model qua API kiểu OpenAI (yui --serve, llama.cpp, Ollama...). Cắt ngay sau </tool> để model nhỏ không nói lan man."""

    def __init__(self, base_url, model="yui", api_key=None, max_tokens=120, temperature=0.7, timeout=120):
        self.url, self.model, self.key, self.mt, self.temp, self.timeout = base_url.rstrip("/"), model, api_key, max_tokens, temperature, timeout

    def __call__(self, msgs):
        body = {"model": self.model, "messages": msgs, "max_tokens": self.mt, "temperature": self.temp, "stop": ["</tool>"]}
        req = urllib.request.Request(self.url + "/v1/chat/completions", data=json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", **({"Authorization": "Bearer " + self.key} if self.key else {})})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read().decode("utf-8", "replace"))["choices"][0]["message"]["content"]


# ============================ MCP server (đưa công cụ + sandbox của Yui cho ứng dụng khác) ============================
def serve_mcp(toolbox, inp=None, out=None):
    inp, out = inp or sys.stdin, out or sys.stdout

    def reply(rid, result=None, error=None):
        m = {"jsonrpc": "2.0", "id": rid}
        m.update({"error": error} if error else {"result": result})
        out.write(json.dumps(m, ensure_ascii=False) + "\n")
        out.flush()

    for line in inp:
        try:
            req = json.loads(line)
        except ValueError:
            continue
        rid, method, params = req.get("id"), req.get("method", ""), req.get("params") or {}
        if rid is None:
            continue   # thông báo (notification): không trả lời
        if method == "initialize":
            reply(rid, {"protocolVersion": params.get("protocolVersion", "2024-11-05"), "capabilities": {"tools": {}}, "serverInfo": {"name": "yui-agent", "version": "1.0"}})
        elif method == "ping":
            reply(rid, {})
        elif method == "tools/list":
            reply(rid, {"tools": [{"name": n, "description": DOCS.get(n, n), "inputSchema": {"type": "object", "properties": {"arg": {"type": "string"}}, "required": ["arg"] if n != "time" else []}}
                                  for n in toolbox.names()]})
        elif method == "tools/call":
            res = toolbox.call(params.get("name", ""), str((params.get("arguments") or {}).get("arg", "")))
            reply(rid, {"content": [{"type": "text", "text": res}], "isError": res.startswith(("lỗi", "bị chặn"))})
        else:
            reply(rid, error={"code": -32601, "message": f"không hỗ trợ {method}"})


# ============================ dòng lệnh ============================
def status(prof, tb):
    tier = sandbox_tier()
    print(f"Sandbox: {tier}" + ("" if tier == "strong" else "  (cài bậc 'strong': apt install bubblewrap; công cụ py cần bậc " + prof.py_min + ")"))
    print(f"Profile: {prof.preset}, scale {prof.scale}, giới hạn {prof.limits}")
    print(f"Mạng cho phép: {prof.net_allow or 'không'}" + (" (cho phép IP nội bộ: CHỈ để thử)" if prof.allow_private else ""))
    print(f"Công cụ model được dùng: {', '.join(tb.names()) or 'không'}")
    for k, v in tb._failed.items():
        print(f"  ⚠ {k}: {v}")
    print(f"Tavily: {len(_tavily_keys())} key" + ("" if _tavily_keys() else " (không có -> search dùng Wikipedia)"))
    for s, tl in tb.mcp_tools.items():
        print(f"MCP {s}: {len(tl)} công cụ ({', '.join(list(tl)[:6])})")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Công cụ + sandbox + MCP cho model Yui")
    ap.add_argument("cmd", choices=["status", "chat", "mcp-server", "tool"])
    ap.add_argument("rest", nargs="*")
    ap.add_argument("--config", default=None)
    ap.add_argument("--url", default="http://127.0.0.1:11434")
    ap.add_argument("--model", default="yui")
    ap.add_argument("--api_key", default=os.environ.get("YUI_API_KEY"))
    a = ap.parse_args(argv)
    prof = Profile.load(a.config)

    def ask(name, arg):
        if not sys.stdin.isatty():
            return False
        return input(f"\n⚠ Model muốn chạy {name}: {arg[:100]!r}. Cho phép? [y/N] ").strip().lower() == "y"

    tb = ToolBox(prof, confirm_cb=ask)
    try:
        if a.cmd == "status":
            status(prof, tb)
        elif a.cmd == "tool":
            print(tb.call(a.rest[0], " ".join(a.rest[1:])))
        elif a.cmd == "mcp-server":
            serve_mcp(tb)
        else:
            llm, agent, hist = OpenAICompat(a.url, a.model, a.api_key), None, []
            agent = Agent(llm, tb)
            print(tb.menu() + "\n(Ctrl+C để thoát)")
            while True:
                q = input("bạn> ").strip()
                if not q:
                    continue
                ans, trace = agent.run(hist + [{"role": "user", "content": q}])
                for t in trace:
                    print(f"  🔧 {t['tool']}({t['arg'][:60]}) -> {t['result'][:80]}")
                print("yui>", ans)
                hist += [{"role": "user", "content": q}, {"role": "assistant", "content": ans}]
                hist = hist[-6:]
    except KeyboardInterrupt:
        pass
    finally:
        tb.close()


if __name__ == "__main__":
    main()
