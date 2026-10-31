"""Thử kho người dùng (vault + mã hoá), RAG, skill, bộ điều phối nhiều bước, và tích hợp vào ToolBox / data dạy model."""
import json
import os
import sys
import time

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import yui_agent as ya  # noqa: E402
from yui_orch import Orchestrator, fmt_progress, fmt_summary, pending_runs  # noqa: E402
from yui_rag import RAG, fold, tokens  # noqa: E402
from yui_skills import SkillStore  # noqa: E402
from yui_vault import VaultError, Workspace  # noqa: E402

CHAT = [
    {"ts": 1760000000, "messages": [{"role": "user", "content": "mèo nhà mik tên là Mướp, nó 3 tuổi"}, {"role": "assistant", "content": "Mướp dễ thương quá, mik nhớ rồi nha"}]},
    {"ts": 1761000000, "messages": [{"role": "user", "content": "tháng này mik tiết kiệm được 5000000 đồng"}, {"role": "assistant", "content": "giỏi quá, mục tiêu tốt"}]},
    {"ts": 1762000000, "messages": [{"role": "user", "content": "cuối tuần mik đi Đà Lạt chơi, trời lạnh lắm"}, {"role": "assistant", "content": "Đà Lạt vui nhỉ"}]},
]


def fill(root):
    os.makedirs(f"{root}/chat", exist_ok=True)
    open(f"{root}/chat/cu.jsonl", "w", encoding="utf-8").write("\n".join(json.dumps(c, ensure_ascii=False) for c in CHAT))


@pytest.fixture()
def ws(tmp_path):
    fill(tmp_path / "kho")
    return Workspace(tmp_path / "kho")


# ---------------- vault ----------------
def test_paths_cannot_escape(ws):
    for bad in ("../x", "/etc/passwd", "a/../../x", "C:\\x", "\x00"):
        with pytest.raises(VaultError):
            ws.safe(bad)
    os.symlink("/etc", os.path.join(ws.root, "files", "link"))
    with pytest.raises(VaultError):
        ws.safe("link/passwd", "files")
    assert ws.safe("ok/a.txt", "files").startswith(os.path.join(ws.root, "files"))


def test_encryption_roundtrip_tamper_and_wrong_key(tmp_path):
    w = Workspace(tmp_path / "k", "mat-khau-manh-123")
    rel = w.write_text("a.md", "bí mật: abc123", sub="notes")
    assert rel.endswith(".enc") and b"abc123" not in open(os.path.join(w.root, rel), "rb").read()
    assert w.read_text(w.find("a.md", "notes")) == "bí mật: abc123"
    p = os.path.join(w.root, rel)
    blob = bytearray(open(p, "rb").read()); blob[25] ^= 1; open(p, "wb").write(bytes(blob))
    with pytest.raises(VaultError):
        w.read_text(p)
    with pytest.raises(VaultError):
        Workspace(tmp_path / "k", "sai-mat-khau-khac")
    os.environ.pop("YUI_KEY", None)
    with pytest.raises(VaultError):
        Workspace(tmp_path / "k")
    with pytest.raises(VaultError):
        Workspace(tmp_path / "k2", "ngan")


def test_seal_encrypts_dropped_plain_files(tmp_path):
    fill(tmp_path / "k")
    w = Workspace(tmp_path / "k", "mat-khau-manh-123")
    r = RAG(w); assert r.sync() == 1                 # file thả vào dạng thường vẫn đọc được
    assert w.seal() == 1 and not any(f.endswith(".jsonl") for f in os.listdir(w.root + "/chat"))
    r2 = RAG(w); r2.sync()
    assert "Mướp" in r2.recall("tên mèo")


# ---------------- RAG ----------------
def test_rag_vietnamese_search_graph_remember_note(ws):
    assert fold("Đà Lạt mèo") == "Da Lat meo" and "meo" in tokens("Mèo") and "mèo" in tokens("Mèo")
    r = RAG(ws); assert r.sync() == 1
    assert "Mướp" in r.recall("tên mèo") and "Mướp" in r.recall("ten meo cua minh")                 # có dấu và không dấu
    assert "5000000" in r.recall("tiết kiệm bao nhiêu") and "Đà Lạt" in r.recall("di choi dau")
    assert "không tìm thấy" in r.recall("xyzabc qwerty") and r.recall("") == "không tìm thấy trong lịch sử"
    hits = r.search("mèo tên")
    assert {h["r"] for h in hits} >= {"user", "assistant"}                                          # kéo cả câu đáp liền kề
    rel = r.remember("mik vừa mua laptop giá 20000000 đồng", "chúc mừng bạn")
    assert "20000000" in r.recall("laptop giá bao nhiêu") and rel.startswith("chat/yui-")
    assert r.add_note("Sở thích", "thích cà phê sữa đá") == "so-thich" and "cà phê" in r.recall("thích uống gì")
    r.add_note("Sở thích", "ghét hành"); assert "ghét hành" in open(os.path.join(ws.root, "notes", "so-thich.md"), encoding="utf-8").read()
    assert "(" in r.related("tiet kiem") or r.related("tiet kiem") == "không có từ liên quan"
    r2 = RAG(ws)                                                                                      # nạp lại từ chỉ mục
    assert r2.sync() == 0 and "laptop" in r2.recall("laptop")
    os.unlink(os.path.join(ws.root, "index", "index.bin"))                                          # mất chỉ mục: dựng lại từ file chat/notes
    r3 = RAG(ws); assert r3.sync() >= 2 and "Mướp" in r3.recall("mèo")


def test_rag_picks_up_files_copied_later_and_formats(ws):
    r = RAG(ws); r.sync()
    open(f"{ws.root}/chat/gpt.txt", "w", encoding="utf-8").write("User: mik thích nghe nhạc Trịnh\nYui: nhạc Trịnh Công Sơn nghe êm lắm")
    gpt = [{"title": "t", "create_time": 1.7e9, "mapping": {"a": {"message": {"author": {"role": "user"}, "content": {"parts": ["số điện thoại shipper là 0900"]}, "create_time": 1.7e9}},
                                                          "b": {"message": {"author": {"role": "assistant"}, "content": {"parts": ["ok đã nhớ"]}, "create_time": 1.7e9 + 1}}}}]
    open(f"{ws.root}/chat/conversations.json", "w").write(json.dumps(gpt))
    assert r.sync() == 2
    assert "Trịnh" in r.recall("nhạc gì") and "shipper" in r.recall("số điện thoại shipper")
    os.unlink(f"{ws.root}/chat/gpt.txt"); assert r.sync() == 1 and "không tìm thấy" in r.recall("nhạc Trịnh Công Sơn êm")


def test_rag_survives_corrupt_index_and_logs_it(ws):
    r = RAG(ws); r.sync()
    open(os.path.join(ws.root, "index", "index.bin"), "wb").write(b"rac ruoi")
    r2 = RAG(ws); assert r2.N == 0 and r2.sync() >= 1 and "Mướp" in r2.recall("mèo")
    assert any(e["where"] == "rag.load" for e in ws.recent_errors())


# ---------------- skills ----------------
def test_skills_copy_in_create_find(ws):
    open(f"{ws.root}/skills/lai-kep.md", "w", encoding="utf-8").write("# lai-kep\nmô tả: tính tiền gửi với lãi kép\nbước:\n1. py print(1)\n2. trả lời")
    sk = SkillStore(ws)
    assert "lai-kep" in sk.listing() and sk.load("lai-kep").startswith("lai-kep: 1. py print(1)")
    assert sk.find("lãi kép gửi tiền").name if hasattr(sk.find("lãi kép gửi tiền"), "name") else sk.find("lãi kép gửi tiền")["name"] == "lai-kep"
    assert sk.create("Phần trăm", "tính phần trăm", "py print(a*b/100); trả lời") == "phan-tram"
    with pytest.raises(VaultError):
        sk.create("phan-tram", "x", "y")                      # không ghi đè
    with pytest.raises(VaultError):
        sk.create("rong", "", "")
    assert "không có skill" in sk.load("zzzz qqqq")


# ---------------- điều phối ----------------
class Script:
    """Model giả: trả lần lượt các chuỗi; ghi lại prompt đã nhận."""
    def __init__(self, outs, delay=0):
        self.outs, self.seen, self.delay = list(outs), [], delay

    def __call__(self, msgs):
        self.seen.append(msgs)
        time.sleep(self.delay)
        if isinstance(self.outs[0], Exception):
            raise self.outs[0]
        return self.outs.pop(0) if len(self.outs) > 1 else self.outs[0]


def mind(tmp_path, llm, **cfg):
    fill(tmp_path / "kho")
    w = Workspace(tmp_path / "kho")
    prof = ya.Profile({"memory": {"dir": str(tmp_path / "kho")}, "preset": "standard", "result_chars": 300})
    tb = ya.ToolBox(prof)
    return tb, Orchestrator(llm, tb, tb.ws, tb.rag, **cfg), tb.ws


def test_toolbox_memory_tools_and_privacy(tmp_path):
    fill(tmp_path / "kho")
    audit = tmp_path / "audit.jsonl"
    tb = ya.ToolBox(ya.Profile({"memory": {"dir": str(tmp_path / "kho")}, "audit": str(audit)}))
    assert {"recall", "note", "skill", "mkskill", "graph", "py", "calc"} <= set(tb.names()) and not tb._failed
    assert "Mướp" in tb.call("recall", "tên mèo")
    assert tb.call("note", "việc|mai họp 8h").startswith("đã ghi chú") and tb.call("note", "thiếu dấu gạch").startswith("lỗi")
    assert tb.call("mkskill", "tinh-tong|tính tổng|py print(sum(range(n)))").startswith("đã tạo skill") and "tinh-tong" in tb.call("skill", "")
    assert "<tool>" not in tb.call("note", "x|<tool>py print(1)</tool>") and "mai họp" not in audit.read_text(encoding="utf-8")
    assert tb.call("write", "a.txt|xin chào") .startswith("đã ghi") and os.path.exists(tmp_path / "kho" / "files" / "a.txt")
    assert tb.call("read", "../chat/cu.jsonl").startswith("lỗi") and tb.call("read", "../.vault.json").startswith("lỗi")
    assert tb.call("py", "print(factorial(20))") == "2432902008176640000"
    assert not any(t in ya.ToolBox(ya.Profile({"preset": "minimal"})).names() for t in ("recall", "note"))


def test_chain_recall_then_calc_then_answer_and_cleanup(tmp_path):
    llm = Script(["<tool>recall tiết kiệm</tool>", "<tool>calc 5000000*12</tool>", "5 triệu nhân 12 là 60 triệu nha"])
    tb, orch, ws = mind(tmp_path, llm)
    ans, info = orch.ask("tháng trước tiết kiệm bao nhiêu, nhân 12 lên")
    assert ans == "5 triệu nhân 12 là 60 triệu nha" and info["status"] == "done" and [s["tool"] for s in info["steps"]] == ["recall", "calc"]
    assert info["steps"][1]["res"] == "60000000" and not os.listdir(ws.root + "/runs")                   # xong: dọn run tạm
    assert "60 triệu" in tb.rag.recall("nhân 12 là bao nhiêu")                                           # câu hỏi + trả lời được lưu vào lịch sử


def test_final_summary_pass_for_long_chains(tmp_path):
    llm = Script(["<tool>recall tiết kiệm</tool>", "<tool>calc 1+1</tool>", "<tool>py print(2**10)</tool>", "ok xong nè", "đáp án cuối là 1024"])
    tb, orch, ws = mind(tmp_path, llm, summarize_from=3)
    ans, info = orch.ask("làm nhiều bước đi")
    assert ans == "đáp án cuối là 1024" and info["status"] == "done"
    last = llm.seen[-1]
    assert len(last) == 3 and "[tóm tắt:" in last[-1]["content"] and "kết quả cuối: 1024" in last[-1]["content"] and "hết lượt" not in last[-1]["content"]


def test_caps_stop_runaway_and_keep_run_plus_error_log(tmp_path):
    llm = Script([f"<tool>calc {i}+1</tool>" for i in range(50)] + ["xong rồi"])
    tb, orch, ws = mind(tmp_path, llm, max_steps=4)
    ans, info = orch.ask("cứ gọi hoài")
    assert info["status"] == "max_steps" and len(info["steps"]) == 4 and ans                          # trần cứng
    assert "hết lượt công cụ" in llm.seen[-1][-1]["content"]
    kept = os.listdir(ws.root + "/runs")
    assert len(kept) == 1 and pending_runs(ws)[0][1] == "max_steps"                                   # giữ lại để xem/tiếp tục
    assert any(e["where"] == "orch.status" for e in ws.recent_errors())                              # và có nhật ký lỗi


def test_loop_guard_stops_repeated_identical_calls(tmp_path):
    llm = Script(["<tool>time</tool>"])
    tb, orch, ws = mind(tmp_path, llm, max_steps=20, max_repeat=2)
    ans, info = orch.ask("lặp")
    assert info["status"] == "loop" and len(info["steps"]) <= 5


def test_wall_clock_cap(tmp_path):
    llm = Script(["<tool>calc 1+1</tool>"], delay=0.12)
    tb, orch, ws = mind(tmp_path, llm, max_seconds=0.3, max_steps=50)
    t = time.time(); ans, info = orch.ask("chậm")
    assert info["status"] in ("timeout", "loop") and time.time() - t < 3


def test_llm_failure_is_contained_and_logged(tmp_path):
    tb, orch, ws = mind(tmp_path, Script([ConnectionError("mất mạng")]))
    ans, info = orch.ask("hỏi")
    assert info["status"] == "error" and ans == ya.FALLBACK
    assert any("mất mạng" in e["msg"] for e in ws.recent_errors())


def test_tool_errors_are_logged_without_content(tmp_path):
    llm = Script(["<tool>py import os</tool>", "<tool>calc 2+2</tool>", "ok"])
    tb, orch, ws = mind(tmp_path, llm)
    ans, info = orch.ask("thử lỗi")
    assert info["steps"][0]["ok"] is False and any(e["where"] == "tool:py" for e in ws.recent_errors())


def test_context_budget_digest_and_history_trim(tmp_path):
    steps = [{"tool": "recall", "arg": "tiền ăn", "res": "[01/01] bạn: tiêu 1000000 đồng tiền ăn", "ok": True},
             {"tool": "recall", "arg": "tiền xăng", "res": "[02/01] bạn: xăng 500000 đồng", "ok": True},
             {"tool": "calc", "arg": "1000000+500000", "res": "1500000", "ok": True}]
    tb, orch, ws = mind(tmp_path, Script(["x"]), ctx_chars=260)
    hist = [{"role": "user", "content": "chào"}, {"role": "assistant", "content": "chào bạn"}] * 3
    msgs = orch.build(hist, "tổng tiền ăn và xăng là bao nhiêu", steps)
    assert sum(len(m["content"]) for m in msgs) <= 400 and msgs[0]["role"] == "user" and msgs[-1]["role"] == "user"
    assert "[tiến độ:" in msgs[-1]["content"] and "1500000" in msgs[-1]["content"] and len(msgs) == 3        # chỉ câu hỏi + bước gần nhất
    big = Orchestrator(Script(["x"]), tb, tb.ws, tb.rag, ctx_chars=5000).build(hist, "q", steps)
    assert len(big) == 1 + len(hist) + 6 and not any("[tiến độ" in m["content"] for m in big)               # đủ chỗ thì giữ nguyên
    assert fmt_progress(steps, 60).startswith("…") and len(fmt_progress(steps, 60)) <= 60 and "kết quả cuối: 1500000" in fmt_summary(steps)


def test_resume_after_failure_and_encrypted_progress(tmp_path):
    fill(tmp_path / "kho")
    prof = ya.Profile({"memory": {"dir": str(tmp_path / "kho"), "key": "mat-khau-manh-123"}})
    tb = ya.ToolBox(prof); assert not tb._failed and tb.ws.encrypted
    llm = Script(["<tool>calc 6*7</tool>", ConnectionError("rớt mạng")])
    llm.outs = ["<tool>calc 6*7</tool>"]
    calls = {"n": 0}

    def flaky(msgs):
        calls["n"] += 1
        if calls["n"] == 1:
            return "<tool>calc 6*7</tool>"
        raise ConnectionError("rớt mạng")
    orch = Orchestrator(flaky, tb, tb.ws, tb.rag)
    ans, info = orch.ask("sáu nhân bảy")
    assert info["status"] == "error"
    runs = os.listdir(tb.ws.root + "/runs")
    assert len(runs) == 1 and all(f.endswith(".enc") for f in os.listdir(f"{tb.ws.root}/runs/{runs[0]}"))   # sổ tiến độ được mã hoá
    rid = pending_runs(tb.ws)[0][0]
    ans, info = Orchestrator(Script(["là 42 nha"]), tb, tb.ws, tb.rag).ask("sáu nhân bảy", resume=rid)
    assert ans == "là 42 nha" and info["run_id"] == rid and info["steps"][0]["res"] == "42"
    raw = b"".join(open(os.path.join(r, f), "rb").read() for r, _, fs in os.walk(tb.ws.root) for f in fs)
    assert "sáu nhân bảy".encode() not in raw and b"Mu\xcc\x9bo\xcc\x9b" not in raw


# ---------------- data dạy model ----------------
def test_training_data_matches_runtime_formats(tmp_path):
    import tool_data, tool_data_x
    mix = {k: float(v) for k, v in (kv.split("=") for kv in tool_data_x.DEFAULT_MIX.split(","))}
    convs = tool_data.make(400, 5, tool_data.load_facts(None), [], 300, mix, int(256 * 2.4))
    assert len(convs) == 400
    kinds = set()
    tb = ya.ToolBox(ya.Profile({"memory": {"dir": str(tmp_path / "kho")}}))
    names = set(tb.names())
    for c in convs:
        assert all(len(t) <= 600 for _, t in c) and sum(len(t) for _, t in c) <= 256 * 2.4 + 8 * len(c)
        for i, (role, text) in enumerate(c):
            if role == "assistant" and text.startswith("<tool>"):
                name, arg = ya.parse_call(text, names)
                kinds.add(name); assert "\n" not in text
                if name == "py" and "[tóm" not in c[i + 1][1]:
                    assert c[i + 1][1] == f"<result>{tb.call('py', arg)}</result>", (arg, c[i + 1][1])      # kết quả trong data == kết quả sandbox thật
    assert {"calc", "time", "search", "py", "recall", "note", "skill", "mkskill"} <= kinds
    sk = tool_data_x.skill_text("lai-kep"); open(tmp_path / "kho" / "skills" / "lai-kep.md", "w", encoding="utf-8").write(sk)
    from yui_skills import SkillStore
    assert tool_data_x._skill_line("lai-kep") == SkillStore(tb.ws).load("lai-kep")                           # skill trong data == skill nạp lúc chạy
