"""Thử make_data.py không cần mạng: Gemini giả + Wikipedia giả chạy trên localhost, 'repo' là một thư mục."""
import json, os, re, subprocess, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_layout import size_tag, params_from_arch, pick_prefix  # noqa: E402

COUNTER = {"n": 0, "search_calls": 0, "wiki_calls": 0}
STATE = {"search_supported": True}
VI = "Hà Nội là thủ đô của nước Việt Nam, nơi có nhiều hồ nước và những con phố cổ rất đặc sắc, thu hút đông đảo du khách mỗi năm. "


def vi_paragraph(i):
    return (f"Đoạn số {i} kể về một buổi sáng yên bình ở thành phố ven sông, nơi người dân dậy sớm để chuẩn bị bữa điểm tâm quen thuộc. " * 3 +
            f"Mỗi người một việc, tiếng cười nói rộn ràng khắp ngõ nhỏ và mùi cà phê thơm lừng lan tỏa trong không khí buổi mai {i}.")


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path.endswith("/models"):
            return self._send(200, {"models": [{"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]}]})
        if u.path == "/w/api.php":   # Wikipedia giả
            q = parse_qs(u.query); COUNTER["wiki_calls"] += 1
            if q.get("list") == ["search"] or q.get("list") == ["random"]:
                key = "search" if q.get("list") == ["search"] else "random"
                return self._send(200, {"query": {key: [{"title": f"Bài {COUNTER['wiki_calls']}"}]}})
            text = (VI * 6) + "\n== Tham khảo ==\nkhông được lấy"
            return self._send(200, {"query": {"pages": [{"title": q["titles"][0], "extract": text}]}})
        self._send(404, {})

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        prompt = body["contents"][0]["parts"][0]["text"]
        has_tool = "tools" in body
        if has_tool and not STATE["search_supported"]:
            return self._send(400, {"error": {"message": "google_search tool is not supported for this model"}})
        if has_tool:
            COUNTER["search_calls"] += 1
        COUNTER["n"] += 1; n = COUNTER["n"]
        if "hội thoại" in prompt:
            convs = [{"messages": [{"role": "user", "content": f"Câu hỏi số {n}-{k} về chủ đề này là gì vậy?"}, {"role": "assistant", "content": f"Yui trả lời ngắn nè {n}-{k} ùi."}]} for k in range(10)]
            text = json.dumps(convs, ensure_ascii=False)
            if has_tool:
                text = "Đây là kết quả:\n```json\n" + text + "\n```"
        else:
            text = "\n===\n".join(vi_paragraph(n * 100 + k) for k in range(5))
        self._send(200, {"candidates": [{"content": {"parts": [{"text": text}]}}]})


@pytest.fixture(scope="module")
def server():
    srv = HTTPServer(("127.0.0.1", 0), Fake)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_port}"
    srv.shutdown()


def run(args, env, repo):
    e = {**os.environ, "GEMINI_API_KEYS": "k1", "HF_TOKEN": "x", "DATA_REPO_DIR": str(repo), **env}
    r = subprocess.run([sys.executable, os.path.join(HERE, "make_data.py"), *args], env=e, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stdout + r.stderr
    return r.stdout


def files(repo):
    return sorted(os.path.relpath(os.path.join(r, f), repo).replace(os.sep, "/") for r, _, fs in os.walk(repo) for f in fs)


def test_tags():
    assert size_tag(params_from_arch(16000, 256, 6, 16)) == "50M"
    assert size_tag(123e6) == "120M" and size_tag(1.46e9) == "1.5B"


def test_auto_creates_size_folders_and_tops_up(server, tmp_path):
    env = dict(MODEL_PARAMS="50M", FT_PAIRS="30", FT_PER_FILE="10", PRETRAIN_TOKENS="40k", PRETRAIN_PER_FILE="5", GEMINI_RPM="600", WEB_SEARCH="off")
    out = run(["auto", "--api_base", server, "--wiki_base", server, "--out_dir", str(tmp_path / "o")], env, tmp_path / "repo")
    fs = files(tmp_path / "repo")
    assert [f for f in fs if f.startswith("Fine-tune/50M/")] == [f"Fine-tune/50M/{i}.txt" for i in (1, 2, 3)], out
    assert any(f.startswith("data/50M/") for f in fs), fs
    # chạy lại: fine-tune đã đủ -> không tạo thêm (data train tính theo TOKEN thật đang có nên có thể bù thêm nếu lần đầu ước lượng thiếu)
    out2 = run(["auto", "--api_base", server, "--wiki_base", server, "--out_dir", str(tmp_path / "o")], {**env, "DATA_KINDS": "finetune"}, tmp_path / "repo")
    assert files(tmp_path / "repo") == fs and "đã đủ" in out2
    # nâng mục tiêu: chỉ tạo phần thiếu, đánh số tiếp 4.txt...
    run(["auto", "--api_base", server, "--wiki_base", server, "--out_dir", str(tmp_path / "o")], {**env, "FT_PAIRS": "50", "DATA_KINDS": "finetune"}, tmp_path / "repo")
    assert "Fine-tune/50M/4.txt" in files(tmp_path / "repo") and "Fine-tune/50M/6.txt" not in files(tmp_path / "repo")


def test_other_size_gets_own_folder_and_legacy_untouched(server, tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    (repo / "1.txt").write_text('{"messages":[{"role":"user","content":"cũ"},{"role":"assistant","content":"ok"}]}\n', encoding="utf-8")
    env = dict(MODEL_PARAMS="120M", FT_PAIRS="10", FT_PER_FILE="10", DATA_KINDS="finetune", GEMINI_RPM="600", WEB_SEARCH="off")
    run(["auto", "--api_base", server, "--wiki_base", server, "--out_dir", str(tmp_path / "o")], env, repo)
    assert "Fine-tune/120M/1.txt" in files(repo) and (repo / "1.txt").read_text(encoding="utf-8").count("cũ") == 1


def test_web_wiki_and_search_and_fallback(server, tmp_path):
    base = dict(MODEL_PARAMS="50M", FT_PAIRS="20", FT_PER_FILE="10", DATA_KINDS="finetune", GEMINI_RPM="600")
    COUNTER.update(wiki_calls=0, search_calls=0)
    STATE["search_supported"] = True
    out = run(["auto", "--api_base", server, "--wiki_base", server, "--out_dir", str(tmp_path / "o1")], {**base, "WEB_SEARCH": "wiki"}, tmp_path / "r1")
    assert COUNTER["wiki_calls"] > 0 and "Wikipedia" in out and "Fine-tune/50M/1.txt" in files(tmp_path / "r1")
    COUNTER["search_calls"] = 0
    run(["auto", "--api_base", server, "--wiki_base", server, "--out_dir", str(tmp_path / "o2")], {**base, "WEB_SEARCH": "search"}, tmp_path / "r2")
    assert COUNTER["search_calls"] > 0 and "Fine-tune/50M/2.txt" in files(tmp_path / "r2")   # câu trả lời có kèm ```json ... ``` vẫn đọc được
    STATE["search_supported"] = False   # model không có google_search -> tự chuyển sang Wikipedia
    out3 = run(["auto", "--api_base", server, "--wiki_base", server, "--out_dir", str(tmp_path / "o3")], {**base, "WEB_SEARCH": "both"}, tmp_path / "r3")
    assert "Fine-tune/50M/2.txt" in files(tmp_path / "r3"), out3
    STATE["search_supported"] = True


def test_pretrain_never_copies_wiki_verbatim(server, tmp_path):
    run(["pretrain", "--web", "wiki", "--tag", "50M", "--files", "1", "--per_file", "5", "--rpm", "600", "--api_base", server, "--wiki_base", server,
         "--repo_dir", str(tmp_path / "repo"), "--out_dir", str(tmp_path / "o")], {}, tmp_path / "repo")
    assert "data/50M/1.txt" in files(tmp_path / "repo")


def test_pick_prefix_prefers_size_folder_then_legacy():
    assert pick_prefix(["Fine-tune/50M/1.txt", "1.txt"], "finetune", "50M") == "Fine-tune/50M/"
    assert pick_prefix(["1.txt", "2.txt"], "finetune", "50M") == ""
    assert pick_prefix(["pretrain/1.txt"], "pretrain", "50M") == "pretrain/"
    assert pick_prefix([], "pretrain", "50M") == "data/50M/"
