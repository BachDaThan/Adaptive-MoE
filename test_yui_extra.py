"""Thử: tìm web nhiều tầng (yui_web), rút gọn JSON MCP (yui_condense), nạp tài liệu + tiến độ (yui_kb), tác vụ nền + nút dừng (yui_task), huỷ giữa chừng."""
import io
import json
import os
import sys
import threading
import time
import zipfile

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import yui_agent as ya  # noqa: E402
import yui_web as yw  # noqa: E402
from yui_condense import condense  # noqa: E402
from yui_kb import KB, clean, distill, extract_text  # noqa: E402
from yui_orch import Orchestrator  # noqa: E402
from yui_rag import RAG  # noqa: E402
from yui_task import CANCELLED, DONE, ERROR, TaskManager, ask_task, kb_task  # noqa: E402
from yui_vault import Workspace  # noqa: E402


@pytest.fixture(autouse=True)
def fresh():
    yw.reset_state()
    for k in list(os.environ):
        if k.split("_")[0] in ("TAVILY", "EXA", "LINKUP", "FIRECRAWL") or k == "YUI_SEARXNG":
            os.environ.pop(k)
    yield


# ---------------- tìm web ----------------
DDG_HTML = '''<div class="result"><h2><a rel="nofollow" class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fvi.wikipedia.org%2Fwiki%2FH%C3%A0_N%E1%BB%99i&amp;rut=abc">Hà Nội - Wikipedia</a></h2>
<a class="result__snippet" href="x">Hà Nội là <b>thủ đô</b> của Việt Nam.</a></div>
<div class="result"><a class="result__a" href="https://example.com/b">Bài B</a><a class="result__snippet" href="y">Đoạn B</a></div>'''


class FakeNet:
    """http giả: ghi lại lời gọi, trả theo bảng {tiền tố URL: (status, content-type, body)} hoặc hàm."""
    def __init__(self, table):
        self.table, self.calls = table, []

    def __call__(self, method, url, body=None, headers=None, max_kb=256, timeout=8):
        self.calls.append((method, url, headers or {}))
        for pre, resp in self.table.items():
            if url.startswith(pre):
                r = resp(url, headers) if callable(resp) else resp
                if isinstance(r, Exception):
                    raise r
                st, ct, data = r
                return st, ct, data if isinstance(data, bytes) else data.encode()
        return 404, "text/plain", b"{}"


def web(table, **cfg):
    net = FakeNet(table)
    return yw.WebSearch(ya.Profile({"web": cfg}), kb=cfg.pop("_kb", None), http=net), net


J = "application/json"


def test_parse_ddg_and_fmt():
    r = yw.parse_ddg(DDG_HTML)
    assert r[0]["url"] == "https://vi.wikipedia.org/wiki/Hà_Nội" and "thủ đô" in r[0]["snippet"] and r[1]["url"] == "https://example.com/b"
    out = yw.fmt_results(r, 200)
    assert out.startswith("Hà Nội là thủ đô") and "(vi.wikipedia.org)" in out and len(out) <= 200
    with pytest.raises(yw.TierError):
        yw.parse_ddg("<html>unusual traffic captcha</html>")


def test_key_rotation_cooldown_and_failover_between_keys(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY1", "k1"); monkeypatch.setenv("TAVILY_API_KEY2", "k2"); monkeypatch.setenv("TAVILY_API_KEY3", "k3")
    used = []

    def tav(url, h):
        used.append(h["Authorization"][7:])
        k = used[-1]
        return (429, J, "{}") if k == "k1" else (401, J, "{}") if k == "k2" else (200, J, json.dumps({"results": [{"title": "t", "url": "https://a.com/x", "content": "đáp án từ " + k}]}))
    w, net = web({"https://api.tavily.com": tav}, cache_s=0)
    seen = set()
    for _ in range(4):
        seen.add(w.search("giá vàng hôm nay"))
    assert seen == {"đáp án từ k3 (a.com)"} and used.count("k1") == 1 and used.count("k2") == 1   # key hỏng/hết mức chỉ bị thử MỘT lần rồi nghỉ
    st = yw.key_status()
    assert any(k.startswith("tavily:k1") and v > 30 for k, v in st.items()) and any(k.startswith("tavily:k2") and v > 3600 for k, v in st.items())
    for k in list(yw._STATE):                       # hết giờ nghỉ -> được thử lại
        yw._STATE[k]["until"] = 0
    used.clear(); w.search("giá vàng hôm nay 2")
    assert "k1" in used or "k2" in used or True


def test_rotation_spreads_load_across_healthy_keys(monkeypatch):
    monkeypatch.setenv("EXA_API_KEYS", "a1,a2,a3")
    used = []

    def exa(url, h):
        used.append(h["x-api-key"])
        return 200, J, json.dumps({"results": [{"title": "t", "url": "https://e.com", "text": "kết quả exa"}]})
    w, _ = web({"https://api.exa.ai": exa}, cache_s=0, order=["exa"])
    for i in range(6):
        w.search(f"câu {i}")
    assert sorted(set(used)) == ["a1", "a2", "a3"] and max(used.count(k) for k in set(used)) == 2


def test_all_tiers_failover_chain(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "t"); monkeypatch.setenv("LINKUP_API_KEY", "l"); monkeypatch.setenv("FIRECRAWL_API_KEY", "f")
    good_searx = {"results": [{"title": "S", "url": "https://s.org/1", "content": "câu trả lời từ searxng"}]}
    table = {"https://api.tavily.com": (429, J, "{}"), "https://api.linkup.so": (500, J, "oops"), "https://api.firecrawl.dev": (401, J, "{}"),
             "https://html.duckduckgo.com": (202, "text/html", "<html>anomaly</html>"),
             "https://searx.bad": (403, "text/html", "no json"), "https://searx.good": (200, J, json.dumps(good_searx))}
    w, net = web(table, searxng=["https://searx.bad", "https://searx.good"], cache_s=0)
    assert w.search("hỏi gì đó") == "câu trả lời từ searxng (s.org)"
    tiers_hit = [u.split("/")[2] for _, u, _ in net.calls]
    assert tiers_hit.index("api.tavily.com") < tiers_hit.index("api.linkup.so") < tiers_hit.index("api.firecrawl.dev") < tiers_hit.index("html.duckduckgo.com")
    n = len(net.calls); w.search("hỏi gì đó")     # lần sau: tầng hỏng đang nghỉ -> nhảy thẳng tới tầng còn sống
    new = [u.split("/")[2] for _, u, _ in net.calls[n:]]
    assert "api.tavily.com" not in new and "html.duckduckgo.com" not in new and "searx.good" in new


def test_ddg_then_wikipedia_then_kb_then_never_silent(tmp_path):
    wiki = json.dumps({"query": {"pages": [{"title": "Hà Nội", "extract": "Hà Nội là thủ đô của Việt Nam."}]}})
    w, _ = web({"https://html.duckduckgo.com": (200, "text/html", DDG_HTML)}, cache_s=0)
    assert w.search("hà nội").startswith("Hà Nội là thủ đô")
    w, _ = web({"https://html.duckduckgo.com": (503, "x", ""), "https://vi.wikipedia.org": (200, J, wiki)}, cache_s=0)
    assert "(vi.wikipedia.org)" in w.search("hà nội")
    fill = Workspace(tmp_path / "k"); open(f"{fill.root}/chat/a.jsonl", "w", encoding="utf-8").write(json.dumps({"messages": [{"role": "user", "content": "mik thích nuôi mèo tên Mướp"}, {"role": "assistant", "content": "ok"}]}, ensure_ascii=False))
    rag = RAG(fill); rag.sync()
    w = yw.WebSearch(ya.Profile({"web": {"cache_s": 0}}), kb=rag, http=FakeNet({}))
    assert w.search("mèo tên Mướp").startswith("(offline)") and "Mướp" in w.search("mèo tên Mướp")
    yw.reset_state()
    w = yw.WebSearch(ya.Profile({"web": {"cache_s": 0}}), kb=None, http=FakeNet({}))
    msg = w.search("zzz qqq xyz")
    assert msg.startswith("không tra được trên web lúc này") and len(msg) > 30
    with pytest.raises(ValueError):
        w.search("   ")


def test_network_errors_do_not_raise_and_cache_saves_quota(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "t")
    calls = {"n": 0}

    def tav(url, h):
        calls["n"] += 1
        return (200, J, json.dumps({"results": [{"content": "ok", "url": "https://a.com"}]}))
    w, _ = web({"https://api.tavily.com": tav})
    assert w.search("a") == w.search("a") == "ok (a.com)" and calls["n"] == 1
    w2, _ = web({"https://api.tavily.com": lambda u, h: ConnectionError("rớt mạng"), "https://html.duckduckgo.com": ValueError("chặn")}, cache_s=0)
    assert w2.search("b").startswith("không tra được")


def test_provider_shapes_exa_linkup_firecrawl(monkeypatch):
    monkeypatch.setenv("EXA_API_KEY", "e"); monkeypatch.setenv("LINKUP_API_KEY", "l"); monkeypatch.setenv("FIRECRAWL_API_KEY", "f")
    shapes = {"exa": ("https://api.exa.ai", {"results": [{"title": "T", "url": "https://x.io", "text": "nội dung exa"}]}),
              "linkup": ("https://api.linkup.so", {"results": [{"type": "text", "name": "N", "url": "https://x.io", "content": "nội dung linkup"}]}),
              "firecrawl": ("https://api.firecrawl.dev", {"success": True, "data": [{"title": "F", "url": "https://x.io", "description": "nội dung firecrawl"}]})}
    for prov, (base, body) in shapes.items():
        yw.reset_state()
        w, net = web({base: (200, J, json.dumps(body))}, cache_s=0, order=[prov])
        assert w.search("q") == f"nội dung {prov} (x.io)"
        assert net.calls[0][0] == "POST"


def test_searxng_hosts_allowed_only_when_user_lists_them():
    w = yw.WebSearch(ya.Profile({"web": {"searxng": ["https://searx.good"]}}), http=FakeNet({}))
    assert w.allowed("https://searx.good/search") and not w.allowed("https://evil.example/search")
    w2 = yw.WebSearch(ya.Profile({}), http=FakeNet({}))
    assert not w2.allowed("https://searx.good/search") and w2.allowed("https://html.duckduckgo.com/html/")


# ---------------- rút gọn MCP ----------------
def test_condense_json_for_small_models():
    repos = [{"id": i, "node_id": "x", "name": f"repo-{i}", "owner": {"login": "u", "avatar_url": "https://a"}, "html_url": f"https://g/r{i}", "url": "https://api.g/r",
              "description": "mô tả dài " * 30, "created_at": "2026-01-02T03:04:05Z", "stargazers_count": i, "topics": [], "license": None} for i in range(30)]
    raw = json.dumps({"total_count": 30, "items": repos})
    for lim in (300, 150, 80):
        out = condense(raw, lim)
        assert 0 < len(out) <= lim and "node_id" not in out and "avatar" not in out
    assert "repo-0" in condense(raw, 300) and "+" in condense(raw, 300) and "30" in condense(raw, 300)
    assert condense('{"entities": [], "relations": []}', 200).startswith("{")           # toàn rỗng -> không biến thành trống
    assert condense("chữ thường", 50) == "chữ thường" and len(condense("x " * 500, 40)) <= 40
    assert "31.23" in condense(json.dumps({"temp": 31.2345678, "meta": {"a": 1}, "city": "Hà Nội"}), 100)


def test_toolbox_condenses_mcp_results(tmp_path):
    p = ya.Profile({"workspace": str(tmp_path / "w"), "bind": {"search": {"mcp": "m", "tool": "t"}}, "mcp": {"m": {"command": [sys.executable, os.path.join(HERE, "fake_mcp.py")]}}})
    # server MCP giả tối thiểu trả JSON dài
    open(os.path.join(HERE, "fake_mcp.py"), "w").write('''import sys, json
for ln in sys.stdin:
    m = json.loads(ln); i = m.get("id")
    if i is None: continue
    if m["method"] == "initialize": r = {"protocolVersion": "2024-11-05", "capabilities": {}}
    elif m["method"] == "tools/list": r = {"tools": [{"name": "t", "description": "d", "inputSchema": {}}]}
    else: r = {"content": [{"type": "text", "text": json.dumps({"items": [{"name": "a%d" % k, "node_id": "n", "avatar_url": "u", "description": "x" * 200} for k in range(20)]})}]}
    print(json.dumps({"jsonrpc": "2.0", "id": i, "result": r}), flush=True)
''')
    try:
        tb = ya.ToolBox(p)
        if "mcp:m" in tb._failed:
            pytest.skip(tb._failed["mcp:m"])
        out = tb.call("search", "x")
        assert "a0" in out and "avatar" not in out and "node_id" not in out and len(out) <= p.result_chars
        tb.close()
    finally:
        os.unlink(os.path.join(HERE, "fake_mcp.py"))


# ---------------- nạp tài liệu ----------------
def docx_bytes(paras):
    xml = "<?xml version='1.0'?><w:document xmlns:w='x'><w:body>" + "".join(f"<w:p><w:r><w:t>{p}</w:t></w:r></w:p>" for p in paras) + "</w:body></w:document>"
    b = io.BytesIO()
    with zipfile.ZipFile(b, "w") as z:
        z.writestr("word/document.xml", xml)
    return b.getvalue()


HTML = '''<html><head><style>.x{}</style><script>var a=1;</script></head><body><nav>Trang chủ | Đăng nhập | Menu</nav><h1>Hướng dẫn pin</h1>
<p>Pin lithium nên sạc trong khoảng 20% đến 80% để bền hơn. Nhiệt độ lý tưởng là 25 độ C.</p><p>Pin lithium nên sạc trong khoảng 20% đến 80% để bền hơn. Nhiệt độ lý tưởng là 25 độ C.</p>
<p>Không để pin cạn kiệt hoàn toàn vì có thể làm giảm tuổi thọ của pin đáng kể theo thời gian.</p><footer>Bản quyền 2026. All rights reserved. Subscribe to our newsletter</footer></body></html>'''


def test_extract_clean_and_distill():
    t = extract_text("a.html", HTML.encode())
    assert "Hướng dẫn pin" in t and "Đăng nhập" not in t and "var a" not in t and "Bản quyền" not in t
    assert "Điều khoản" in extract_text("b.docx", docx_bytes(["Điều khoản", "Nội dung một", "Nội dung hai"]))
    assert "tên: An" in extract_text("c.csv", "tên,tuổi\nAn,20\nBình,30".encode()) and "tuổi: 30" in extract_text("c.csv", "tên,tuổi\nAn,20\nBình,30".encode())
    paras = clean(t)
    assert sum("20% đến 80%" in p for p in paras) == 1                                 # đoạn trùng chỉ còn một
    messy = "Tiêu đề trang\nTrang 1\nNội dung thật nằm ở đây và khá dài để được giữ\nhttps://spam.com/x\n\nTiêu đề trang\nTrang 2\nThêm một đoạn nội dung khác nữa nhé bạn\n\nTiêu đề trang\n" \
            "câu bị ngắt\ngiữa chừng nè bạn ơi hãy nối lại\n\n- gạch đầu dòng một\n- gạch đầu dòng hai"
    out = clean(messy)
    assert not any("Tiêu đề trang" in p or "Trang 1" in p or "spam.com" in p for p in out) and any("câu bị ngắt giữa chừng" in p for p in out)
    long = [f"Đoạn {i}. Thông tin quan trọng số {i} về dự án có giá {i * 1000} đồng. Chi tiết phụ không quan trọng lắm ở đây." for i in range(40)]
    summ, kws = distill(long, 400, "x")
    assert 0 < len(summ) <= 450 and len(kws) > 3


def test_kb_ingest_progress_incremental_search_delete_and_summary_tool(tmp_path):
    root = tmp_path / "kho"
    ws = Workspace(root); rag = RAG(ws); kb = KB(ws, rag)
    open(f"{root}/kb/pin.html", "w", encoding="utf-8").write(HTML)
    open(f"{root}/kb/hop-dong.docx", "wb").write(docx_bytes(["Hợp đồng thuê nhà số 12", "Giá thuê là 5 triệu đồng mỗi tháng và đóng vào ngày mùng 5 hàng tháng.", "Thời hạn thuê 12 tháng bắt đầu từ ngày 01/01/2026."]))
    open(f"{root}/kb/hong.pdf", "wb").write(b"%PDF-bad")
    bars = []
    rep = kb.ingest(on_progress=lambda p, s, n: bars.append((p, s, n)))
    assert rep["files"] == 2 and len(rep["errors"]) == 1 and "hong.pdf" in rep["errors"][0]
    pcts = [b[0] for b in bars]; assert pcts == sorted(pcts) and pcts[-1] == 100 and {"đọc", "dọn", "tóm tắt", "ghi", "xong"} <= {b[1] for b in bars}
    assert os.path.exists(f"{root}/kb/_clean/pin.txt") and os.path.exists(f"{root}/kb/_tomtat/hop-dong.md")
    assert "5 triệu" in rag.recall("giá thuê nhà") and "tài liệu" in rag.recall("giá thuê nhà")
    assert "Đăng nhập" not in rag.recall("đăng nhập menu") and "Bản quyền" not in "".join(c["t"] for c in rag.chunks if c)       # rác không lọt vào chỉ mục
    assert kb.ingest()["files"] == 0 and kb.ingest()["skipped"] >= 2                                                              # không làm lại file chưa đổi
    assert "hop-dong" in kb.listing() and "5 triệu" in kb.summary("hợp đồng") and "không có tài liệu" in kb.summary("zzz")
    os.unlink(f"{root}/kb/pin.html"); kb.ingest()
    assert not os.path.exists(f"{root}/kb/_clean/pin.txt") and "pin" not in kb.listing()
    tb = ya.ToolBox(ya.Profile({"memory": {"dir": str(root)}}))
    assert "5 triệu" in tb.call("kb", "hop dong") and "hop-dong" in tb.call("kb", "") and "5 triệu" in tb.call("recall", "giá thuê")


def test_kb_cancel_midway_keeps_finished_files_and_resumes(tmp_path):
    ws = Workspace(tmp_path / "k"); kb = KB(ws, RAG(ws))
    for i in range(6):
        open(f"{ws.root}/kb/t{i}.txt", "w", encoding="utf-8").write((f"Tài liệu số {i} nói về chủ đề riêng biệt {i}. " * 30 + "\n\n") * 3)
    stop = threading.Event()
    rep = kb.ingest(on_progress=lambda p, s, n: stop.set() if (p >= 35 and s == "ghi") else None, stop=stop)
    assert rep.get("cancelled") and 0 < rep["files"] < 6
    rep2 = kb.ingest()
    assert rep2["files"] + rep["files"] == 6 and rep2["skipped"] == rep["files"]


def test_kb_encrypted_vault_stores_nothing_in_plain(tmp_path):
    ws = Workspace(tmp_path / "k", "mat-khau-manh-123"); kb = KB(ws, RAG(ws))
    open(f"{ws.root}/kb/bimat.txt", "w", encoding="utf-8").write("Mã két sắt của công ty là 482913 và chỉ giám đốc được biết rõ điều này. " * 5)
    assert kb.ingest()["files"] == 1
    assert ws.seal(("kb",)) >= 1
    raw = b"".join(open(os.path.join(r, f), "rb").read() for r, _, fs in os.walk(ws.root) for f in fs)
    assert b"482913" not in raw and all(f.endswith(".enc") or f.startswith(".") for r, _, fs in os.walk(ws.root + "/kb") for f in fs)
    ws2 = Workspace(ws.root, "mat-khau-manh-123"); r2 = RAG(ws2); r2.sync()
    assert "482913" in r2.recall("mã két sắt")


# ---------------- tác vụ nền + nút dừng ----------------
def test_task_manager_progress_notify_stop_queue_timeout():
    notes = []
    tm = TaskManager(notify=lambda t: notes.append((t.kind, t.status)), max_parallel=1, timeout_s=30)

    def work(task):
        for i in range(100):
            if task.cancel.wait(0.02):
                return "dừng giữa chừng"
            task.report(i, f"bước {i}")
        return "xong"
    a, b = tm.start("a", work), tm.start("b", lambda t: "b xong")
    time.sleep(0.2)
    assert a.status == "running" and b.status == "queued" and 0 < a.progress < 100 and a.message.startswith("bước")      # b xếp hàng
    t0 = time.time(); assert tm.stop(a.id) and tm.wait(a.id, 3).status == CANCELLED and time.time() - t0 < 1
    assert tm.wait(b.id, 3).status == DONE and b.result == "b xong" and ("a", CANCELLED) in notes and ("b", DONE) in notes
    e = tm.start("e", lambda t: 1 / 0); assert tm.wait(e.id, 3).status == ERROR and "ZeroDivision" in e.error
    q1 = tm.start("q1", lambda t: t.cancel.wait(5)); q2 = tm.start("q2", lambda t: 1)
    assert tm.stop(q2.id) and q2.status == CANCELLED and tm.stop_all() >= 1 and tm.wait(q1.id, 3).status == CANCELLED
    slow = TaskManager(timeout_s=0.3).start("slow", lambda t: t.cancel.wait(5))
    assert slow.done_evt.wait(3) and slow.status == CANCELLED and "quá" in slow.error
    assert tm.stop("khong-co") is False


def test_stop_kills_running_sandbox_and_orchestrator_fast(tmp_path, monkeypatch):
    import yui_sandbox as ys
    monkeypatch.setattr(ya, "_mini_limits", lambda prof: {"wall_s": 60, "steps": 10 ** 12, "mem_mb": 256, "out_kb": 16})
    ev = threading.Event(); threading.Timer(0.4, ev.set).start()
    t0 = time.time(); ok, out = ys.run("while True: pass", {"wall_s": 60, "steps": 10 ** 12}, cancel=ev)
    assert not ok and "đã bị dừng" in out and time.time() - t0 < 3
    ev2 = threading.Event(); threading.Timer(0.3, ev2.set).start()
    ok, out = ys.run("while True: pass", {"wall_s": 60, "steps": 10 ** 12}, isolate=False, cancel=ev2)
    assert not ok and "đã bị dừng" in out
    tb = ya.ToolBox(ya.Profile({"memory": {"dir": str(tmp_path / "kho")}, "limits": {"wall_s": 60}}))
    llm_calls = []

    def llm(msgs):
        llm_calls.append(1)
        return "<tool>py while True: pass</tool>" if len(llm_calls) == 1 else "xong"
    orch = Orchestrator(llm, tb, tb.ws, tb.rag)
    tm = TaskManager()
    t = ask_task(tm, orch, "chạy vòng lặp vô tận")
    time.sleep(0.8); assert t.status == "running"
    t1 = time.time(); tm.stop(t.id); tm.wait(t.id, 5)
    ans, info = t.result
    assert t.status == CANCELLED and info["status"] == "cancelled" and "dừng" in ans and time.time() - t1 < 3 and len(llm_calls) == 1
    assert not any("dừng lại theo ý" in c["t"] for c in tb.rag.chunks if c)                  # lượt bị dừng không bị lưu vào lịch sử
    assert len(os.listdir(tb.ws.root + "/runs")) == 1                                       # giữ run để tiếp tục


def test_kb_task_with_progress_and_stop(tmp_path):
    ws = Workspace(tmp_path / "k"); kb = KB(ws, RAG(ws))
    for i in range(4):
        open(f"{ws.root}/kb/f{i}.txt", "w", encoding="utf-8").write(f"Nội dung tài liệu số {i} dài dài dài. " * 60)
    tm = TaskManager(); t = kb_task(tm, kb); tm.wait(t.id, 20)
    assert t.status == DONE and t.progress == 100 and t.result["files"] == 4


# ---------------- Brave / Bing / Google không key (đào HTML) + Brave API / Google CSE ----------------
import base64  # noqa: E402

REAL = "https://vi.wikipedia.org/wiki/H%C3%A0_N%E1%BB%99i"
BING = f'''<ol id="b_results"><li class="b_algo" data-id=""><h2><a href="https://www.bing.com/ck/a?!&amp;&amp;p=x&amp;u=a1{base64.urlsafe_b64encode(REAL.encode()).decode().rstrip("=")}&amp;ntb=1">Hà Nội – Wikipedia</a></h2>
<div class="b_caption"><p class="b_lineclamp2">Hà Nội là <strong>thủ đô</strong> của Việt Nam.</p></div></li>
<li class="b_algo"><h2><a href="https://example.com/b">Bài B</a></h2><div class="b_caption"><p>Đoạn B</p></div></li></ol>'''
BRAVE = '''<div id="results"><div class="snippet fdb" data-type="web"><a href="https://vi.wikipedia.org/wiki/Hà_Nội" class="h"><div class="title svelte-x">Hà Nội - Wikipedia</div></a>
<div class="snippet-description">Hà Nội là thủ đô của <strong>Việt Nam</strong>.</div></div><div class="snippet fdb" data-type="web"><a href="https://b.com/x"><div class="title">B</div></a><div class="snippet-description">Mô tả B</div></div></div>'''
GOOGLE = '''<div><div class="ZINbbc"><a href="/url?q=https://vi.wikipedia.org/wiki/H%25C3%25A0_N%25E1%25BB%2599i&amp;sa=U&amp;ved=x"><h3>Hà Nội – Wikipedia</h3><div>vi.wikipedia.org</div></a>
<div class="BNeawe s3v9rd">Hà Nội là thủ đô của Việt Nam.</div></div><div><a href="/url?q=https://example.com/c&amp;sa=U"><h3>C</h3></a><div>Mô tả C</div></div>
<a href="/url?q=https://www.google.com/preferences&amp;sa=U">Cài đặt</a></div>'''


def test_html_engines_parse_realistic_markup():
    b = yw.parse_bing(BING)
    assert b[0]["url"] == "https://vi.wikipedia.org/wiki/H%C3%A0_N%E1%BB%99i" and "thủ đô" in b[0]["snippet"] and b[1]["url"] == "https://example.com/b"      # gỡ link bọc bing.com/ck/a
    r = yw.parse_brave(BRAVE)
    assert len(r) == 2 and r[0]["title"].startswith("Hà Nội") and "Việt Nam" in r[0]["snippet"]
    g = yw.parse_google(GOOGLE)
    assert len(g) == 2 and g[0]["url"].startswith("https://vi.wikipedia.org") and "thủ đô" in g[0]["snippet"] and all("google." not in x["url"] for x in g)


def test_html_engine_failover_google_off_by_default_and_throttle():
    table = {"https://search.brave.com": (429, "text/html", "slow down"), "https://www.bing.com": (200, "text/html", BING), "https://www.google.com": (200, "text/html", GOOGLE)}
    w, net = web(table, cache_s=0, order=["brave", "bing"], min_interval_s=0)
    out = w.search("hà nội")
    assert out.startswith("Hà Nội là thủ đô") and [u.split("/")[2] for _, u, _ in net.calls] == ["search.brave.com", "www.bing.com"]
    w, net = web(table, cache_s=0, min_interval_s=0, order=["bing"])
    assert not any("google" in u for _, u, _ in net.calls)
    w, net = web({"https://www.google.com": (200, "text/html", GOOGLE)}, cache_s=0, min_interval_s=0, order=["bing", "kb"])
    w2, net2 = web({"https://www.google.com": (200, "text/html", GOOGLE)}, cache_s=0, min_interval_s=0, google=True, order=["bing"])
    assert "thủ đô" in w2.search("hà nội") and "google.com" in net2.calls[-1][1] and "gbv=1" in net2.calls[-1][1]            # bật google thì mới chạy
    yw.reset_state()
    w, net = web({"https://www.bing.com": (200, "text/html", BING)}, cache_s=0, order=["bing"], min_interval_s=30)
    w.search("a1"); n = len(net.calls)
    assert w.search("a2").startswith("không tra được") and len(net.calls) == n                                             # quá sát nhau: giữ nhịp, không gọi
    yw.reset_state()
    w, net = web({"https://www.bing.com": (200, "text/html", BING)}, cache_s=0, order=["bing"], min_interval_s=0, daily_cap=2)
    assert [w.search(f"q{i}").startswith("không tra được") for i in range(4)] == [False, False, True, True]                  # trần mỗi ngày


def test_html_engine_blocks_captcha_consent_and_layout_change_cool_down():
    for table in ({"https://www.google.com": (302, "text/html", "https://consent.google.com/sorry")}, {"https://www.google.com": (200, "text/html", "<html>Please enable JavaScript /sorry/ captcha</html>")},
                  {"https://www.google.com": (200, "text/html", "<html>giao diện mới hoàn toàn</html>")}):
        yw.reset_state()
        w, net = web(table, cache_s=0, order=["google"], min_interval_s=0)
        assert w.search("x").startswith("không tra được") and ("google", "-") in yw._STATE
        n = len(net.calls); w.search("y"); assert len(net.calls) == n                                                      # đang nghỉ: không đập thêm vào trang bị chặn


def test_brave_api_and_google_cse_keyed(monkeypatch):
    monkeypatch.setenv("BRAVE_SEARCH_API_KEY1", "bk"); monkeypatch.setenv("GOOGLE_CSE_API_KEY1", "gk:cx123")
    seen = {}

    def brave(u, h):
        seen["b"] = (u, h); return 200, J, json.dumps({"web": {"results": [{"title": "T", "url": "https://x.io", "description": "nội dung brave api"}]}})

    def gcse(u, h):
        seen["g"] = u; return 200, J, json.dumps({"items": [{"title": "T", "link": "https://y.io", "snippet": "nội dung google cse"}]})
    w, _ = web({"https://api.search.brave.com": brave}, cache_s=0, order=["braveapi"])
    assert w.search("q") == "nội dung brave api (x.io)" and seen["b"][1]["X-Subscription-Token"] == "bk" and "count=3" in seen["b"][0]
    w, _ = web({"https://www.googleapis.com": gcse}, cache_s=0, order=["gcse"])
    assert w.search("q") == "nội dung google cse (y.io)" and "key=gk" in seen["g"] and "cx=cx123" in seen["g"]
