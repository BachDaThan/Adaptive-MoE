"""Thử make_data.py không cần mạng: Gemini giả + Wikipedia giả chạy trên localhost, 'repo' là một thư mục."""
import json, os, re, subprocess, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_layout import size_tag, params_from_arch, pick_prefix  # noqa: E402

COUNTER = {"n": 0, "search_calls": 0, "wiki_calls": 0, "tavily_ok": 0, "tavily_bad": 0}
STATE = {"search_supported": True, "daily_block": None, "overload": None}
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
            return self._send(200, {"models": [{"name": "models/gemini-2.5-flash", "supportedGenerationMethods": ["generateContent"]}, {"name": "models/gemini-3.0-flash", "supportedGenerationMethods": ["generateContent"]}]})
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
        if self.path.endswith("/search"):   # Tavily giả
            if self.headers.get("Authorization") == "Bearer bad":
                COUNTER["tavily_bad"] += 1
                return self._send(401, {"detail": "invalid key"})
            COUNTER["tavily_ok"] += 1
            return self._send(200, {"results": [{"title": f"KQ {i}", "content": VI * 2} for i in range(3)]})
        prompt = body["contents"][0]["parts"][0]["text"]
        if STATE["overload"] and STATE["overload"] in self.path:
            return self._send(503, {"error": {"message": "The model is overloaded. Please try again later."}})
        if STATE["daily_block"] and STATE["daily_block"] in self.path:
            COUNTER["blocked"] = COUNTER.get("blocked", 0) + 1
            return self._send(429, {"error": {"message": "Quota exceeded for metric: generativelanguage.googleapis.com/generate_content_free_tier_requests, limit: 250, GenerateRequestsPerDayPerProjectPerModel-FreeTier"}})
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


def test_rank_models_high_to_low_aliases_last():
    import make_data
    names = "gemini-flash-latest gemini-pro-latest gemini-4-argon gemini-3.8-flash gemini-3.10-flash gemini-2.5-flash gemini-embedding-001".split()
    assert make_data.rank_models(names) == ["gemini-4-argon", "gemini-3.10-flash", "gemini-3.8-flash", "gemini-2.5-flash", "gemini-flash-latest", "gemini-pro-latest"]


def test_numbered_gemini_keys_and_tavily_rotation(server, tmp_path):
    env = dict(GEMINI_API_KEYS="", GEMINI_API_KEY1="g1", GEMINI_API_KEY2="g2", TAVILY_API_KEY1="bad", TAVILY_API_KEY2="good",
               MODEL_PARAMS="50M", FT_PAIRS="10", FT_PER_FILE="10", DATA_KINDS="finetune", GEMINI_RPM="600", WEB_SEARCH="tavily")
    COUNTER.update(tavily_ok=0, tavily_bad=0)
    out = run(["auto", "--api_base", server, "--wiki_base", server, "--tavily_base", server, "--out_dir", str(tmp_path / "o")], env, tmp_path / "repo")
    assert "2 key Gemini, 2 key Tavily" in out, out
    assert COUNTER["tavily_ok"] > 0 and "Fine-tune/50M/1.txt" in files(tmp_path / "repo"), out


def test_works_without_tavily_key(server, tmp_path):
    env = dict(MODEL_PARAMS="50M", FT_PAIRS="10", FT_PER_FILE="10", DATA_KINDS="finetune", GEMINI_RPM="600", WEB_SEARCH="tavily", TAVILY_API_KEY="", TAVILY_API_KEYS="")
    out = run(["auto", "--api_base", server, "--wiki_base", server, "--out_dir", str(tmp_path / "o")], env, tmp_path / "repo")
    assert "không có key Tavily" in out and "Fine-tune/50M/1.txt" in files(tmp_path / "repo")


def test_auto_detects_params_from_config_json_in_repo(server, tmp_path):
    repo = tmp_path / "repo"; repo.mkdir()
    (repo / "config.json").write_text(json.dumps({"vocab": 16000, "d": 512, "layers": 8, "experts": 32}), encoding="utf-8")
    out = run(["auto", "--plan"], dict(DATA_KINDS="finetune"), repo)
    assert "tự nhận" in out and "Fine-tune/440M/" in out, out


def test_make_for_training_creates_data_before_training(server, tmp_path):
    """train_lm.py --size 50M gọi make_for_training: tạo Fine-tune/50M + data/50M rồi mới train."""
    code = ("import make_data; ok = make_data.make_for_training(49_900_000, '50M', 'x/y', 'dataset', 'finetune,pretrain', 5); print('RESULT', ok)")
    e = {**os.environ, "GEMINI_API_KEY1": "g1", "GEMINI_API_KEYS": "", "GEMINI_API_BASE": server, "WIKI_BASE": server, "TAVILY_BASE": server, "WEB_SEARCH": "wiki",
         "DATA_REPO_DIR": str(tmp_path / "repo"), "FT_PAIRS": "10", "FT_PER_FILE": "10", "PRETRAIN_TOKENS": "3k", "PRETRAIN_PER_FILE": "5", "GEMINI_RPM": "600",
         "HF_TOKEN": "x", "PYTHONPATH": HERE}
    r = subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True, timeout=120, cwd=tmp_path)   # cwd riêng: thư mục data_out không dính lần chạy khác
    assert "RESULT True" in r.stdout, r.stdout + r.stderr
    fs = files(tmp_path / "repo")
    assert "Fine-tune/50M/1.txt" in fs and any(f.startswith("data/50M/") for f in fs), fs


def test_make_for_training_skips_without_keys(tmp_path):
    code = "import make_data; print('RESULT', make_data.make_for_training(1, '50M', 'x/y'))"
    e = {k: v for k, v in os.environ.items() if not k.startswith("GEMINI")}
    r = subprocess.run([sys.executable, "-c", code], env={**e, "PYTHONPATH": HERE}, capture_output=True, text=True, timeout=60, cwd=HERE)
    assert "RESULT False" in r.stdout and "bỏ qua" in r.stdout


def test_arch_for_params():
    from data_layout import arch_for_params, parse_params, lr_for_params
    assert arch_for_params(parse_params("50M")) == (256, 6, 16)        # đúng preset small
    assert arch_for_params(parse_params("250M")) == (384, 8, 32)       # đúng preset medium
    for t in ("10M", "120M", "1B"):
        d, l, e = arch_for_params(parse_params(t))
        assert abs(params_from_arch(16000, d, l, e) / parse_params(t) - 1) < 0.05 and d % 8 == 0 and e <= 256
    assert lr_for_params(50e6) == pytest.approx(1e-3)


def test_daily_quota_of_top_model_switches_to_next_model_not_stall(server, tmp_path):
    STATE["daily_block"] = "gemini-3.0-flash"   # model xếp đầu hết quota ngày trên mọi key
    try:
        COUNTER["blocked"] = 0
        env = dict(GEMINI_API_KEYS="", GEMINI_API_KEY1="g1", GEMINI_API_KEY2="g2", MODEL_PARAMS="50M", FT_PAIRS="10", FT_PER_FILE="10", DATA_KINDS="finetune", GEMINI_RPM="600", WEB_SEARCH="off")
        out = run(["auto", "--api_base", server, "--wiki_base", server, "--out_dir", str(tmp_path / "o")], env, tmp_path / "repo")
    finally:
        STATE["daily_block"] = None
    assert COUNTER["blocked"] >= 1 and "Fine-tune/50M/1.txt" in files(tmp_path / "repo"), out
    assert "hết quota NGÀY" in out and "gemini-2.5-flash" in out


def test_tavily_is_primary_source_and_search_only_fallback(server, tmp_path):
    env = dict(GEMINI_API_KEYS="", GEMINI_API_KEY1="g1", TAVILY_API_KEY1="good", MODEL_PARAMS="50M", FT_PAIRS="30", FT_PER_FILE="10", DATA_KINDS="finetune", GEMINI_RPM="600", WEB_SEARCH="both")
    COUNTER.update(tavily_ok=0, search_calls=0)
    run(["auto", "--api_base", server, "--wiki_base", server, "--tavily_base", server, "--out_dir", str(tmp_path / "o")], env, tmp_path / "repo")
    assert COUNTER["tavily_ok"] > 0 and COUNTER["search_calls"] == 0   # có Tavily thì không tốn quota google_search của Gemini


def test_auto_tools_kind_is_free_and_opt_in(server, tmp_path):
    env = dict(MODEL_PARAMS="50M", FT_PAIRS="10", FT_PER_FILE="10", TOOL_PAIRS="25", DATA_KINDS="tools", GEMINI_API_KEYS="", GEMINI_API_KEY1="")
    calls_before = COUNTER["n"]
    out = run(["auto", "--api_base", server, "--out_dir", str(tmp_path / "o")], env, tmp_path / "repo")
    fs = [f for f in files(tmp_path / "repo") if f.startswith("Fine-tune/50M/")]
    assert fs == ["Fine-tune/50M/1.txt", "Fine-tune/50M/2.txt", "Fine-tune/50M/3.txt"] and COUNTER["n"] == calls_before, out   # không gọi Gemini
    again = run(["auto", "--api_base", server, "--out_dir", str(tmp_path / "o")], env, tmp_path / "repo")
    assert "đủ mục tiêu" in again and len(files(tmp_path / "repo")) == 3


def test_overloaded_503_switches_model_reuses_document_and_labels_source(server, tmp_path):
    STATE["overload"] = "gemini-3.0-flash"
    try:
        COUNTER.update(tavily_ok=0)
        env = dict(GEMINI_API_KEYS="", GEMINI_API_KEY1="g1", GEMINI_API_KEY2="g2", TAVILY_API_KEY1="good", MODEL_PARAMS="50M", FT_PAIRS="10", FT_PER_FILE="10",
                   DATA_KINDS="finetune", GEMINI_RPM="600", WEB_SEARCH="tavily")
        out = run(["auto", "--api_base", server, "--wiki_base", server, "--tavily_base", server, "--out_dir", str(tmp_path / "o")], env, tmp_path / "repo")
    finally:
        STATE["overload"] = None
    assert "quá tải" in out and "nguồn: Tavily" in out and "Wikipedia" not in out, out          # nhãn nguồn đúng, không còn ghi nhầm Wikipedia
    assert "Fine-tune/50M/1.txt" in files(tmp_path / "repo")
    assert out.count("nguồn: Tavily") == COUNTER["tavily_ok"]                                    # thử lại sau 503 dùng lại tài liệu cũ, không gọi Tavily thêm


def test_second_make_data_process_is_locked_out(server, tmp_path):
    import fcntl
    od = tmp_path / "o"; od.mkdir()
    f = open(od / ".lock_finetune_50M", "w"); fcntl.flock(f, fcntl.LOCK_EX)
    r = subprocess.run([sys.executable, os.path.join(HERE, "make_data.py"), "finetune", "--tag", "50M", "--files", "1", "--api_base", server, "--out_dir", str(od),
                        "--repo_dir", str(tmp_path / "repo")], env={**os.environ, "GEMINI_API_KEY1": "g"}, capture_output=True, text=True, timeout=60)
    assert "đang có một tiến trình make_data" in r.stdout and files(tmp_path / "repo") == [] if os.path.exists(tmp_path / "repo") else True


def test_estimate_minutes_scales_with_model_size():
    import make_data
    a, b, c = (make_data.estimate_minutes(make_data.parse_count(x), 9) for x in ("50M", "1B", "10B"))
    assert a <= b <= c and make_data.estimate_minutes(1e9, 1) >= make_data.estimate_minutes(1e9, 9)
