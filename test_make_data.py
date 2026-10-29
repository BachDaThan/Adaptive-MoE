"""Thử make_data.py không cần mạng: Gemini giả + Wikipedia giả chạy trên localhost, 'repo' là một thư mục."""
import json, os, re, subprocess, sys, threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from urllib.parse import parse_qs, urlparse

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from data_layout import size_tag, params_from_arch, pick_prefix  # noqa: E402

COUNTER = {"n": 0, "search_calls": 0, "wiki_calls": 0, "tavily_ok": 0, "tavily_bad": 0}
STATE = {"search_supported": True, "daily_block": None, "overload": None, "fail": {}}   # fail: {"cohere": (429, {...}), ...} ép provider/nguồn trả lỗi
VI = "Hà Nội là thủ đô của nước Việt Nam, nơi có nhiều hồ nước và những con phố cổ rất đặc sắc, thu hút đông đảo du khách mỗi năm. "


def vi_paragraph(i):
    return (f"Đoạn số {i} kể về một buổi sáng yên bình ở thành phố ven sông, nơi người dân dậy sớm để chuẩn bị bữa điểm tâm quen thuộc. " * 3 +
            f"Mỗi người một việc, tiếng cười nói rộn ràng khắp ngõ nhỏ và mùi cà phê thơm lừng lan tỏa trong không khí buổi mai {i}.")


def llm_text(prompt, n):
    """Nội dung 'mô hình' giả: hội thoại JSON / đoạn văn pretrain / câu trả lời restyle."""
    if '"question"' in prompt and "old_answer" in prompt:   # restyle
        items = [json.loads(ln) for ln in prompt.split("Các câu:\n")[1].split("\n\nTrả về")[0].splitlines() if ln.strip().startswith("{")]
        return json.dumps([{"i": it["i"], "answer": f"Yui đáp lại {it['i']} nè ùi."} for it in items], ensure_ascii=False)
    if "hội thoại" in prompt:
        convs = [{"messages": [{"role": "user", "content": f"Câu hỏi số {n}-{k} về chủ đề này là gì vậy?"}, {"role": "assistant", "content": f"Yui trả lời ngắn nè {n}-{k} ùi."}]} for k in range(10)]
        return json.dumps(convs, ensure_ascii=False)
    return "\n===\n".join(vi_paragraph(n * 100 + k) for k in range(5))


class Fake(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, obj):
        b = json.dumps(obj).encode()
        self.send_response(code); self.send_header("Content-Type", "application/json"); self.send_header("Content-Length", str(len(b))); self.end_headers(); self.wfile.write(b)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/cohere/v2/models":
            return self._send(200, {"models": [{"name": "command-a-03-2025"}, {"name": "command-r-08-2024"}]})
        if u.path == "/or/models":
            free = {"prompt": "0", "completion": "0"}
            return self._send(200, {"data": [
                {"id": "google/gemma-3-27b-it:free", "pricing": free, "context_length": 8000, "architecture": {"output_modalities": ["text"]}},
                {"id": "meta-llama/llama-3.3-70b-instruct:free", "pricing": free, "context_length": 131072, "architecture": {"output_modalities": ["text"]}},
                {"id": "vendor/paid-model", "pricing": {"prompt": "0.001", "completion": "0.002"}, "architecture": {"output_modalities": ["text"]}},
                {"id": "vendor/image-gen:free", "pricing": free, "architecture": {"output_modalities": ["text", "image"]}}]})
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
        for pre, name in (("/cohere/", "cohere"), ("/or/", "openrouter"), ("/mistral/", "mistral")):   # LLM giả của 3 provider mới
            if self.path.startswith(pre):
                COUNTER[name] = COUNTER.get(name, 0) + 1
                if name in STATE["fail"]:
                    return self._send(*STATE["fail"][name])
                prompt = body["messages"][-1]["content"]
                COUNTER["n"] += 1
                text = llm_text(prompt, COUNTER["n"])
                if name == "cohere":   # Cohere giả bọc ```json để thử json_slice
                    text = "Đây nè:\n```json\n" + text + "\n```" if text.lstrip().startswith("[") else text
                    return self._send(200, {"message": {"content": [{"type": "text", "text": text}]}})
                return self._send(200, {"choices": [{"message": {"content": text}}]})
        for pre, name in (("/exa/", "exa"), ("/linkup/", "linkup"), ("/fire/", "firecrawl")):   # nguồn cào web giả
            if self.path.startswith(pre):
                COUNTER[name] = COUNTER.get(name, 0) + 1
                if name in STATE["fail"]:
                    return self._send(*STATE["fail"][name])
                if name == "exa":
                    return self._send(200, {"results": [{"title": f"Exa {i}", "text": VI * 2} for i in range(3)]})
                if name == "linkup":
                    return self._send(200, {"results": [{"type": "text", "name": f"Linkup {i}", "url": "u", "content": VI * 2} for i in range(3)]})
                return self._send(200, {"success": True, "data": [{"title": f"FC {i}", "markdown": "# Tiêu đề\n![ảnh](http://x/y.png)\n[liên kết](http://x) " + VI * 2} for i in range(3)]})
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
        COUNTER["n"] += 1; n = COUNTER["n"]; COUNTER["gemini"] = COUNTER.get("gemini", 0) + 1
        text = llm_text(prompt, n)
        if has_tool and text.lstrip().startswith("["):
            text = "Đây là kết quả:\n```json\n" + text + "\n```"
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


# ============================ chuỗi LLM: Cohere -> OpenRouter -> Mistral -> Gemini ============================
def chain_env(server, **extra):
    base = dict(GEMINI_API_KEYS="", GEMINI_API_KEY1="g1", COHERE_API_KEY1="c1", OPENROUTER_API_KEY1="o1", MISTRAL_API_KEY1="m1",
                COHERE_BASE=server + "/cohere", OPENROUTER_BASE=server + "/or", MISTRAL_BASE=server + "/mistral",
                COHERE_RPM="600", OPENROUTER_RPM="600", MISTRAL_RPM="600", GEMINI_RPM="600", MODEL_PARAMS="50M", FT_PAIRS="10", FT_PER_FILE="10",
                DATA_KINDS="finetune", WEB_SEARCH="off", LLM_ORDER="", SCRAPE_ORDER="")
    base.update(extra)
    return base


def reset(**kw):
    for k in ("cohere", "openrouter", "mistral", "gemini", "exa", "linkup", "firecrawl", "tavily_ok"):
        COUNTER[k] = 0
    STATE["fail"] = dict(kw)


def auto_run(server, tmp_path, env, extra=()):
    return run(["auto", "--api_base", server, "--wiki_base", server, "--tavily_base", server, "--out_dir", str(tmp_path / "o"), *extra], env, tmp_path / "repo")


def test_cohere_is_first_and_nothing_else_is_called(server, tmp_path):
    reset()
    out = auto_run(server, tmp_path, chain_env(server, DATA_KINDS="finetune,pretrain", PRETRAIN_TOKENS="3k", PRETRAIN_PER_FILE="5"))
    fs = files(tmp_path / "repo")
    assert COUNTER["cohere"] > 0 and COUNTER["openrouter"] == 0 and COUNTER["mistral"] == 0 and COUNTER["gemini"] == 0, out
    assert "Fine-tune/50M/1.txt" in fs and any(f.startswith("data/50M/") for f in fs), (out, fs)   # cả fine-tune lẫn data train đều qua chuỗi mới
    assert "Cohere -> OpenRouter -> Mistral -> Gemini" in out and "1 key Cohere, 1 key OpenRouter, 1 key Mistral, 1 key Gemini" in out, out


def test_model_ranking_dry_run_lists_every_provider_in_order(server, tmp_path):
    reset()
    out = run(["finetune", "--dry_models", "--api_base", server], chain_env(server), tmp_path / "repo")
    line = next(ln for ln in out.splitlines() if "Thứ hạng" in ln)
    order = [line.index(x) for x in ("cohere::command-a-03-2025", "openrouter::meta-llama/llama-3.3-70b-instruct:free", "mistral::mistral-large-latest", "gemini::gemini-3.0-flash")]
    assert order == sorted(order), line
    assert "command-r-plus" not in line and "paid-model" not in line and "image-gen" not in line, line   # model không có trong danh sách / không miễn phí / sinh ảnh bị loại
    assert line.index("llama-3.3-70b") < line.index("gemma-3-27b"), line                                   # họ model ưa thích đứng trước dù context ngắn hơn


def test_falls_through_every_provider_down_to_gemini(server, tmp_path):
    reset(cohere=(429, {"message": "You are using a Trial key, which is limited to 1000 API calls / month."}),
          openrouter=(429, {"error": {"message": "Rate limit exceeded: free-models-per-day"}}),
          mistral=(401, {"message": "Unauthorized"}))
    out = auto_run(server, tmp_path, chain_env(server))
    assert COUNTER["cohere"] == 1 and COUNTER["openrouter"] == 1 and COUNTER["mistral"] == 1 and COUNTER["gemini"] > 0, out   # mỗi provider hỏng chỉ bị thử đúng 1 lần rồi bỏ cả provider
    assert "Fine-tune/50M/1.txt" in files(tmp_path / "repo") and "hết hạn mức dài hạn" in out, out


def test_rate_limited_provider_spills_over_instead_of_waiting(server, tmp_path):
    reset(cohere=(429, {"message": "Too many requests, please wait before trying again."}))
    out = auto_run(server, tmp_path, chain_env(server))
    assert COUNTER["openrouter"] > 0 and "Fine-tune/50M/1.txt" in files(tmp_path / "repo"), out
    assert "⏳ cohere" not in out, out   # không đứng chờ 30s rồi mới sang provider kế


def test_llm_order_env_reorders_and_unknown_names_are_ignored(server, tmp_path):
    reset()
    out = auto_run(server, tmp_path, chain_env(server, LLM_ORDER="gemini,bogus,cohere"))
    assert COUNTER["gemini"] > 0 and COUNTER["cohere"] == 0 and "Gemini -> Cohere" in out and "bogus" in out, out


def test_works_with_only_cohere_key_and_without_gemini(server, tmp_path):
    reset()
    out = auto_run(server, tmp_path, chain_env(server, GEMINI_API_KEY1="", OPENROUTER_API_KEY1="", MISTRAL_API_KEY1=""))
    assert COUNTER["cohere"] > 0 and COUNTER["gemini"] == 0 and "Fine-tune/50M/1.txt" in files(tmp_path / "repo"), out


def test_no_llm_key_at_all_exits_with_clear_message(server, tmp_path):
    e = {**os.environ, **chain_env(server, GEMINI_API_KEY1="", OPENROUTER_API_KEY1="", MISTRAL_API_KEY1="", COHERE_API_KEY1=""), "HF_TOKEN": "x", "DATA_REPO_DIR": str(tmp_path / "repo")}
    r = subprocess.run([sys.executable, os.path.join(HERE, "make_data.py"), "auto", "--out_dir", str(tmp_path / "o")], env=e, capture_output=True, text=True, timeout=60)
    assert r.returncode != 0 and "COHERE_API_KEY1" in (r.stdout + r.stderr)


def test_restyle_goes_through_the_chain(server, tmp_path):
    reset()
    src = tmp_path / "src.jsonl"
    src.write_text("\n".join(json.dumps({"messages": [{"role": "user", "content": f"Câu hỏi cũ số {i} là gì thế nhỉ"}, {"role": "assistant", "content": "Dạ em là siêu trợ lý."}]}, ensure_ascii=False) for i in range(25)), encoding="utf-8")
    out = run(["restyle", "--source_file", str(src), "--files", "1", "--per_file", "10", "--batch", "12", "--tag", "50M", "--api_base", server, "--out_dir", str(tmp_path / "o"),
               "--repo_dir", str(tmp_path / "repo")], chain_env(server), tmp_path / "repo")
    assert COUNTER["cohere"] >= 1 and COUNTER["gemini"] == 0 and "Fine-tune/50M/1.txt" in files(tmp_path / "repo"), out
    assert "Yui đáp lại" in open(tmp_path / "repo" / "Fine-tune" / "50M" / "1.txt", encoding="utf-8").read()


def test_make_for_training_runs_with_only_a_non_gemini_key(server, tmp_path):
    reset()
    code = "import make_data; print('RESULT', make_data.make_for_training(49_900_000, '50M', 'x/y', 'dataset', 'finetune', 5))"
    e = {**os.environ, **chain_env(server, GEMINI_API_KEY1="", OPENROUTER_API_KEY1="", MISTRAL_API_KEY1=""), "DATA_REPO_DIR": str(tmp_path / "repo"), "HF_TOKEN": "x", "PYTHONPATH": HERE}
    r = subprocess.run([sys.executable, "-c", code], env=e, capture_output=True, text=True, timeout=120, cwd=tmp_path)
    assert "RESULT True" in r.stdout and "Fine-tune/50M/1.txt" in files(tmp_path / "repo"), r.stdout + r.stderr


# ============================ chuỗi cào web: Tavily -> Exa -> Linkup -> Firecrawl ============================
def scrape_env(server, **extra):
    e = dict(WEB_SEARCH="tavily", TAVILY_API_KEY1="bad", EXA_API_KEY1="e1", LINKUP_API_KEY1="l1", FIRECRAWL_API_KEY1="f1",
             EXA_BASE=server + "/exa", LINKUP_BASE=server + "/linkup", FIRECRAWL_BASE=server + "/fire",
             GEMINI_API_KEY1="g1", COHERE_API_KEY1="", OPENROUTER_API_KEY1="", MISTRAL_API_KEY1="")
    e.update(extra)
    return chain_env(server, **e)


def test_scrape_chain_tavily_dead_falls_to_exa(server, tmp_path):
    reset()
    COUNTER["tavily_bad"] = 0
    out = auto_run(server, tmp_path, scrape_env(server))
    assert COUNTER["tavily_bad"] >= 1 and COUNTER["exa"] >= 1 and COUNTER["linkup"] == 0 and COUNTER["firecrawl"] == 0, out
    assert "nguồn: Exa" in out and "1 key Tavily, 1 key Exa, 1 key Linkup, 1 key Firecrawl" in out, out


def test_scrape_chain_reaches_linkup_then_firecrawl(server, tmp_path):
    reset(exa=(402, {"error": "no credits"}))
    out = auto_run(server, tmp_path, scrape_env(server))
    assert COUNTER["exa"] >= 1 and COUNTER["linkup"] >= 1 and COUNTER["firecrawl"] == 0 and "nguồn: Linkup" in out, out
    reset(exa=(402, {"error": "no credits"}), linkup=(401, {"error": "bad key"}))
    out2 = auto_run(server, tmp_path / "b", scrape_env(server))
    assert COUNTER["firecrawl"] >= 1 and "nguồn: Firecrawl" in out2, out2
    assert "Fine-tune/50M/1.txt" in files(tmp_path / "b" / "repo")


def test_scrape_order_env_and_all_sources_down_falls_back_to_wikipedia(server, tmp_path):
    reset()
    out = auto_run(server, tmp_path, scrape_env(server, SCRAPE_ORDER="firecrawl,linkup"))
    assert COUNTER["firecrawl"] >= 1 and COUNTER["exa"] == 0 and "nguồn: Firecrawl" in out, out
    reset(exa=(402, {}), linkup=(401, {}), firecrawl=(402, {}))
    out2 = auto_run(server, tmp_path / "c", scrape_env(server, WEB_SEARCH="both"))
    assert "Wikipedia" in out2 and "Fine-tune/50M/1.txt" in files(tmp_path / "c" / "repo"), out2


def test_firecrawl_markdown_is_cleaned_and_v2_shape_is_read():
    import providers
    fc = providers.Firecrawl(["k"])
    items = fc._parse(json.dumps({"data": {"web": [{"title": "T", "markdown": "# Tiêu đề\n![a](http://x/y.png)\n[liên kết](http://x) nội dung **đậm**"}]}}))
    assert items == [("T", "Tiêu đề liên kết nội dung đậm")]


# ============================ router (không cần mạng) ============================
def test_router_drops_whole_provider_on_quota_and_model_on_repeated_empty():
    import providers as P

    class Dummy(P.ChatProvider):
        name = "dummy"
        def __init__(self, behaviour):
            super().__init__(["k"], "http://x", ["m1", "m2"]); self.behaviour = behaviour
        def generate(self, model, system, prompt, temperature=1.0, max_tokens=4096):
            raise self.behaviour

    r = P.LLMRouter(["dummy"], {"dummy": Dummy(P.QuotaOut("hết"))})
    with pytest.raises(P.ModelGone):
        r.generate("dummy::m1", "s", "p")
    assert "dummy" in r.dead
    with pytest.raises(P.ModelGone):      # provider đã chết: loại ngay, không gọi lại
        r.generate("dummy::m2", "s", "p")
    r2 = P.LLMRouter(["dummy"], {"dummy": Dummy(P.Empty("rỗng"))})
    for _ in range(2):
        with pytest.raises(P.Empty):
            r2.generate("dummy::m1", "s", "p")
    with pytest.raises(P.ModelGone):      # lần thứ 3 rỗng liên tiếp -> bỏ model
        r2.generate("dummy::m1", "s", "p")
    assert not r2.can_search("dummy::m1")  # google_search chỉ có ở Gemini
