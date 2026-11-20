"""Thử web_corpus.py (DATA_MODE=web_only) và tool_data_os.py không cần mạng: Tavily giả (hàm post), 'repo' là một thư mục."""
import json
import os
import random
import re
import subprocess
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import web_corpus as wc  # noqa: E402
import tool_data_os as tdo  # noqa: E402

SENT = [
    "Trước khi cài lại máy, bạn nên sao lưu toàn bộ tài liệu quan trọng sang một ổ đĩa ngoài để tránh mất dữ liệu ngoài ý muốn.",
    "Nước sôi ở một trăm độ C trong điều kiện áp suất bình thường, nhưng trên núi cao nhiệt độ sôi thấp hơn một chút.",
    "Người Việt có truyền thống gói bánh chưng vào dịp Tết để tưởng nhớ tổ tiên và cầu mong một năm mới an lành.",
    "Muốn học tốt ngoại ngữ, mỗi ngày bạn nên dành ít nhất ba mươi phút để nghe, nói và ghi chép những từ mới gặp.",
    "Pin lithium sẽ bền hơn nếu bạn tránh để nó cạn kiệt hoàn toàn và không sạc liên tục khi máy đã đầy.",
    "Thuật toán sắp xếp nổi bọt so sánh từng cặp phần tử liền kề rồi đổi chỗ chúng nếu thứ tự chưa đúng.",
    "Khi đi du lịch bụi, hãy mang theo giấy tờ tùy thân, một ít tiền mặt và bản sao các thông tin liên lạc khẩn cấp.",
    "Một bữa sáng đủ chất giúp cơ thể tỉnh táo hơn, đặc biệt khi có thêm trái cây tươi và một ít đạm từ trứng hoặc sữa.",
    "Hồ Gươm nằm giữa trung tâm thủ đô, quanh năm có đông người đến đi dạo, tập thể dục và chụp ảnh lưu niệm.",
    "Nếu mạng Wi-Fi chập chờn, hãy thử khởi động lại bộ phát, đổi kênh sóng hoặc đặt thiết bị ở vị trí thoáng hơn.",
    "Lãi suất tiết kiệm cao hơn lạm phát thì tiền gửi mới thật sự tăng giá trị theo thời gian dài.",
    "Cây xương rồng chỉ cần tưới rất ít nước, nhưng cần nhiều ánh nắng để phát triển khỏe mạnh và ra hoa.",
    "Trong lập trình, việc đặt tên biến rõ ràng giúp người đọc hiểu mã nguồn nhanh hơn và giảm bớt lỗi khi bảo trì.",
    "Đà Lạt có khí hậu mát mẻ quanh năm, nổi tiếng với những vườn hoa, đồi thông và các quán cà phê nhỏ xinh.",
    "Tập chạy bộ nên bắt đầu bằng những quãng ngắn, tăng dần quãng đường mỗi tuần để cơ thể kịp thích nghi.",
    "Dung lượng bộ nhớ trong của điện thoại thường bị chiếm nhiều bởi ảnh, video và các tệp tạm của ứng dụng.",
    "Làng gốm Bát Tràng lưu giữ nghề thủ công hàng trăm năm với những chiếc bình men rạn nổi tiếng khắp nơi.",
    "Để viết luận thuyết phục, bạn cần nêu luận điểm rõ ràng, dẫn chứng cụ thể và kết luận ngắn gọn dễ nhớ.",
    "Tủ lạnh nên đặt cách tường vài centimet để hơi nóng thoát ra dễ dàng và tiết kiệm điện hơn mỗi tháng.",
    "Mạng cục bộ cho phép các máy trong nhà chia sẻ tệp và máy in mà không cần đi qua Internet bên ngoài.",
    "Cờ vua rèn luyện khả năng suy nghĩ trước nhiều nước đi và kiên nhẫn trước những tình huống bất lợi.",
    "Quả địa cầu quay quanh trục mỗi ngày một vòng và quay quanh Mặt Trời mỗi năm một vòng đầy đủ.",
    "Ngủ đủ tám tiếng mỗi đêm giúp trí nhớ ổn định, tâm trạng tốt hơn và cơ thể phục hồi sau một ngày dài.",
    "Cơ sở dữ liệu quan hệ lưu thông tin trong các bảng liên kết với nhau thông qua khóa chính và khóa ngoại.",
    "Phở bò Hà Nội thường có nước dùng trong, bánh phở mềm và vài lát hành tây ngâm giấm chua nhẹ.",
]


def vi_md(i, n_sec=3):
    """Bài Markdown tiếng Việt giả: tiêu đề + n_sec mục, câu lấy khác nhau theo i (để MinHash/lọc không coi là trùng)."""
    r = random.Random(1000 + i)
    s = SENT[:]
    r.shuffle(s)
    out = [f"# Bài thử số {i} về cuộc sống hằng ngày", "", s[0] + " " + s[1], ""]
    heads = ["Chuẩn bị trước khi bắt đầu", "Những điều cần lưu ý", "Mẹo nhỏ hữu ích", "Lỗi thường gặp"]
    for k in range(n_sec):
        out += [f"## {heads[k]} (phần {i}.{k})", "", s[2 + 2 * k] + " " + s[3 + 2 * k], ""]
    out += ["## Các bước thực hiện", "", f"- Bước một của bài {i}: chuẩn bị đầy đủ dụng cụ cần dùng", f"- Bước hai của bài {i}: làm theo hướng dẫn từng phần",
            f"- Bước ba của bài {i}: kiểm tra lại kết quả", "", "[Đăng nhập](https://x.vn/login) | [Đăng ký](https://x.vn/r) | [Liên hệ](https://x.vn/c)",
            "![ảnh](https://x.vn/a.png)", "Xem thêm"]
    return "\n".join(out)


class FakeTavily:
    """post(path, key, body) -> (status, text). Ghi lại các lượt gọi; có thể ép lỗi theo key."""

    def __init__(self, n_docs=60, bad_keys=None):
        self.calls, self.n_docs, self.bad = [], n_docs, bad_keys or {}
        self.pages = {f"https://site{i}.vn/bai-{i}": vi_md(i) for i in range(n_docs)}
        self.pages["https://vi.wikipedia.org/wiki/Ha_Noi"] = vi_md(999)
        self.pages["https://huggingface.co/datasets/x"] = vi_md(998)

    def __call__(self, path, key, body):
        self.calls.append((path, key, body))
        if key in self.bad:
            return self.bad[key], "{}"
        if path == "/search":
            h = abs(hash(body["query"])) % self.n_docs
            urls = [list(self.pages)[(h + j) % len(self.pages)] for j in range(body["max_results"])]
            return 200, json.dumps({"results": [{"url": u, "title": "t", "content": "c"} for u in urls]})
        if path == "/extract":
            assert len(body["urls"]) <= wc.EXTRACT_BATCH and body["format"] == "markdown"
            return 200, json.dumps({"results": [{"url": u, "raw_content": self.pages.get(u, "")} for u in body["urls"]], "failed_results": []})
        if path == "/map":
            return 200, json.dumps({"results": list(self.pages)[:body["limit"]]})
        return 404, "{}"


# ------------------------------------------------------------------ từ khoá, miền chặn
def test_seed_queries_deterministic_unique():
    a = wc.gen_seed_queries(300, random.Random(7))
    b = wc.gen_seed_queries(300, random.Random(7))
    assert a == b and len(set(a)) == 300
    assert wc.gen_seed_queries(5, random.Random(8)) != wc.gen_seed_queries(5, random.Random(9))
    assert any("quản lý thời gian" in q or "Python cơ bản" in q for q in wc.gen_seed_queries(5000, random.Random(1)))
    assert "chủ đề riêng của tôi là gì" in wc.gen_seed_queries(5000, random.Random(1), ["chủ đề riêng của tôi"])


@pytest.mark.parametrize("url,expect", [
    ("https://vi.wikipedia.org/wiki/A", True), ("https://en.wikipedia.org/x", True), ("https://huggingface.co/datasets/a/b", True),
    ("https://www.wikidata.org/wiki/Q1", True), ("https://vnexpress.net/bai", False), ("https://notwikipedia.org.vn/x", False)])
def test_blocked_domains(url, expect):
    assert wc.blocked(url) is expect
    assert wc.blocked("https://abc.vn/x", ["abc.vn"]) is True


# ------------------------------------------------------------------ MinHash
def test_minhash_near_duplicates():
    mh = wc.MinHashLSH()
    base = " ".join(SENT[:10])
    assert mh.add(base) is True
    assert mh.add(base) is False                                      # trùng nguyên văn
    near = base.replace("sao lưu", "sao chép", 1) + " Thêm một câu nhỏ ở cuối bài."
    assert mh.add(near) is False                                      # gần giống
    assert mh.add(" ".join(SENT[10:20])) is True                      # khác hẳn
    assert mh.add("") is False


def test_minhash_accent_insensitive_exact():
    mh = wc.MinHashLSH()
    t = " ".join(SENT[:8])
    assert mh.add(t) and not mh.add(wc.strip_accents(t))


# ------------------------------------------------------------------ Markdown -> sạch / SFT / văn xuôi
def test_clean_markdown_strips_junk():
    c = wc.clean_markdown(vi_md(3))
    assert "![" not in c and "https://" not in c and "Đăng nhập" not in c and "Xem thêm" not in c
    assert "## Mẹo nhỏ hữu ích" in c and "- Bước một" in c


def test_md_to_sft_pairs():
    md = wc.clean_markdown(vi_md(5))
    pairs = wc.md_to_sft(md, wc.title_of(md), random.Random(1))
    assert len(pairs) >= 3
    for q, a in pairs:
        assert 4 <= len(q) <= 160 and 60 <= len(a) <= 600
        assert "http" not in a and "@" not in a and a.lower().rstrip(" .") != q.lower().rstrip(" ?.")
    assert any(a.startswith("1) ") for _, a in pairs)                 # mục danh sách -> "1) …; 2) …"
    # tiêu đề chung chung bị bỏ
    g = "# Một bài\n\n## Kết luận\n\n" + SENT[0] + " " + SENT[1] + "\n\n## Giới thiệu\n\n" + SENT[2] + " " + SENT[3] + "\n"
    assert wc.md_to_sft(g, "", random.Random(1)) == []


def test_prose_paragraphs_filters():
    md = wc.clean_markdown(vi_md(2)) + "\n\n" + ("This is an English paragraph that should be rejected by the Vietnamese check. " * 6)
    ps = wc.prose_paragraphs(md)
    assert ps and all(150 <= len(p) <= 1500 for p in ps)
    assert not any("English paragraph" in p for p in ps)


# ------------------------------------------------------------------ TavilyWeb
def test_tavily_search_extract_credits_and_batch():
    fake = FakeTavily(30)
    t = wc.TavilyWeb(["k1"], delay=0, post=fake)
    urls = [u for u, _ in t.search_urls("máy tính chạy chậm là gì")]
    assert len(urls) == 10
    got = t.extract(urls)
    assert set(got) == set(urls)
    assert abs(t.credits - (1 + 10 / 5)) < 1e-9                         # 1 search + 10 URL / 5
    assert len(t.extract(list(fake.pages)[:25])) == wc.EXTRACT_BATCH     # cắt còn 20 URL/lần
    assert len(t.map_site("https://site1.vn", 7)) == 7


def test_tavily_key_rotation(monkeypatch):
    fake = FakeTavily(10, bad_keys={"dead": 432, "slow": 429})
    t = wc.TavilyWeb(["dead", "slow", "ok"], delay=0, post=fake)
    for _ in range(4):
        assert t.search_urls("du lịch Đà Lạt là gì")
    st = {k["k"]: k for k in t.keys}
    assert st["dead"]["dead"] is True and st["slow"]["cool"] > 0 and not st["ok"]["dead"] and t.available
    only_dead = wc.TavilyWeb(["dead"], delay=0, post=fake)
    assert only_dead.search_urls("x") == [] and not only_dead.available


# ------------------------------------------------------------------ thu thập + nhật ký nguồn
def test_collect_and_build_items(tmp_path):
    import vi_filter
    fake = FakeTavily(60)
    tav = wc.TavilyWeb(["k"], delay=0, post=fake)
    log = wc.SourceLog(str(tmp_path))
    qs = wc.gen_seed_queries(40, random.Random(2))
    wc.collect(tav, None, qs, log, 20, vi_keep=vi_filter.keep, say=lambda *a, **k: None)
    assert len(log.unused()) >= 20
    assert not any(wc.blocked(u) for u in log.rows)                      # không có Wikipedia/Hugging Face
    n_calls = len(fake.calls)
    wc.collect(tav, None, qs, log, 20, vi_keep=vi_filter.keep, say=lambda *a, **k: None)
    assert len(fake.calls) == n_calls                                    # đã đủ tài liệu -> không gọi thêm
    chunks = wc.build_items("finetune", log, wc.MinHashLSH(), set(), random.Random(3), per_file=10, files=2)
    assert len(chunks) == 2
    for c in chunks:
        for ln in c.strip().split("\n"):
            m = json.loads(ln)["messages"]
            assert [x["role"] for x in m] == ["user", "assistant"]
    used = [r for r in log.rows.values() if r["used"] == "finetune"]
    assert used
    log.save()
    assert os.path.exists(tmp_path / "web_sources.jsonl")
    ds = (tmp_path / "DATA_SOURCES.md").read_text(encoding="utf-8")
    assert "wikipedia" not in ds and "huggingface" not in ds and "Chưa kiểm giấy phép" in ds
    # nạp lại nhật ký: URL cũ không bị cào lại
    log2 = wc.SourceLog(str(tmp_path))
    assert all(log2.has(u) for u in log.rows)
    # pretrain dùng các tài liệu CHƯA dùng (mỗi tài liệu chỉ thuộc một loại)
    pre = wc.build_items("pretrain", log2, wc.MinHashLSH(), set(), random.Random(3), per_file=10, files=1)
    assert pre and all(len(p) >= 150 for p in pre[0].strip().split("\n\n"))


def test_dedup_removes_near_identical_docs(tmp_path):
    log = wc.SourceLog(str(tmp_path))
    log.add("https://a.vn/1", "A", vi_md(1))
    log.add("https://a.vn/2", "A2", vi_md(1).replace("số 1", "số 1 ").replace("một", "một"))   # gần như cùng bài
    log.add("https://a.vn/3", "B", vi_md(40))
    wc.build_items("pretrain", log, wc.MinHashLSH(), set(), random.Random(1), per_file=5, files=3)
    assert log.rows["https://a.vn/2"]["used"].startswith("rejected")


# ------------------------------------------------------------------ chạy trọn vẹn qua make_data (repo = thư mục)
def _args(md, mode, repo_dir, out_dir, files=1, per_file=10, tag="50M"):
    return md.build_parser().parse_args([mode, "--data_mode", "web_only", "--tag", tag, "--files", str(files), "--per_file", str(per_file),
                                         "--repo_dir", repo_dir, "--out_dir", out_dir, "--seed", "5"])


def test_run_web_only_end_to_end(tmp_path):
    import make_data as md
    fake = FakeTavily(80)
    repo, out = str(tmp_path / "repo"), str(tmp_path / "out")
    n1 = wc.run_web_only(_args(md, "finetune", repo, out, files=2), md, tav=wc.TavilyWeb(["k"], delay=0, post=fake), say=lambda *a, **k: None)
    n2 = wc.run_web_only(_args(md, "pretrain", repo, out, files=1), md, tav=wc.TavilyWeb(["k"], delay=0, post=fake), say=lambda *a, **k: None)
    assert n1 == 2 and n2 == 1
    names = sorted(os.listdir(os.path.join(repo, "Fine-tune", "50M"))) + sorted(os.listdir(os.path.join(repo, "data", "50M")))
    assert names == ["1.txt", "2.txt", "1.txt"]
    from sft_format import load_convs
    texts = {n: open(os.path.join(repo, "Fine-tune", "50M", n), encoding="utf-8").read() for n in ("1.txt", "2.txt")}
    convs, bad = load_convs(texts)
    assert len(convs) >= 15 and sum(bad.values()) == 0                                 # train_sft.py đọc được
    # chạy lại finetune: không tạo trùng câu hỏi đã có trên repo
    before = {c[0][0] for c in convs}
    wc.run_web_only(_args(md, "finetune", repo, out, files=1), md, tav=wc.TavilyWeb(["k"], delay=0, post=fake), say=lambda *a, **k: None)
    t3 = open(os.path.join(repo, "Fine-tune", "50M", "3.txt"), encoding="utf-8").read()
    c3, _ = load_convs({"3": t3})
    assert not ({c[0][0] for c in c3} & before)


def test_run_web_only_without_keys_does_nothing(tmp_path, monkeypatch):
    import make_data as md
    for k in list(os.environ):
        if k.endswith(("_API_KEY", "_KEYS")) or re.search(r"_API_KEY\d+$", k):
            monkeypatch.delenv(k)
    a = _args(md, "finetune", str(tmp_path / "r"), str(tmp_path / "o"))
    assert wc.run_web_only(a, md, say=lambda *a_, **k: None) == 0


def test_cli_auto_plan_web_only_needs_no_llm_key(tmp_path):
    repo = tmp_path / "repo"
    repo.mkdir()
    env = {k: v for k, v in os.environ.items() if "API_KEY" not in k}
    env.update(DATA_MODE="web_only", DATA_REPO_DIR=str(repo), MODEL_PARAMS="50000000", DATA_KINDS="finetune,pretrain", HF_TOKEN="")
    r = subprocess.run([sys.executable, os.path.join(HERE, "make_data.py"), "auto", "--plan", "--out_dir", str(tmp_path / "o")], env=env,
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-400:]
    assert "web_only finetune" in r.stdout and "credit Tavily" in r.stdout


def test_cli_web_only_missing_keys_exits_cleanly(tmp_path):
    env = {k: v for k, v in os.environ.items() if "API_KEY" not in k}
    r = subprocess.run([sys.executable, os.path.join(HERE, "make_data.py"), "finetune", "--data_mode", "web_only", "--repo_dir", str(tmp_path / "r"),
                        "--out_dir", str(tmp_path / "o"), "--tag", "50M"], env=env, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0 and "cần ít nhất một key cào web" in r.stdout


# ------------------------------------------------------------------ tool_data_os
def _convs(n=600, seed=11):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        c = tdo.gen_os(rng, 300)
        if c:
            out.append(c)
    return out


def test_every_os_conv_has_a_tool_call():
    """make_data.count_existing chỉ đếm hội thoại có <tool> là 'dùng công cụ' — hội thoại không công cụ làm mục tiêu TOOL_PAIRS không bao giờ đủ."""
    assert all(any(t.startswith("<tool>") for r, t in c) for c in _convs(1000, 21))


def test_os_convs_well_formed():
    for c in _convs():
        assert c[0][0] == "user" and c[-1][0] == "assistant" and len(c) % 2 == 0
        assert all(role == ("user" if i % 2 == 0 else "assistant") for i, (role, _) in enumerate(c))
        for role, text in c:
            if text.startswith("<tool>"):
                assert role == "assistant" and text.endswith("</tool>") and "\n" not in text
                name = text[6:-7].split(" ")[0]
                assert name in tdo.OS_TOOL_NAMES or name in ("recall", "search"), name
            elif role == "user" and text.startswith("<result>"):
                assert text.endswith("</result>")


def test_sensitive_actions_wait_for_confirmation():
    bad_claims = re.compile(r"đã (tắt|dừng|cài|mở cổng|khởi động lại|đổi)\b|xong rồi|tắt rồi|cài xong", re.I)
    n_conf = 0
    for c in _convs(800):
        for i, (role, t) in enumerate(c):
            if role == "user" and "chờ xác nhận vật lý" in t:
                n_conf += 1
                final = c[-1][1]
                assert "xác nhận" in final and "chưa" in final.lower() or "chờ" in final or "mới" in final, final
                assert not bad_claims.search(final), final
    assert n_conf > 100


def test_os_data_has_accent_variants_and_canonical_args():
    cs = _convs(800)
    users = [t for c in cs for r, t in c if r == "user" and not t.startswith("<result>")]
    assert any(re.search(r"[ạảãàáâăêôơư]", u) for u in users)                                   # có dấu
    assert any(u == wc.strip_accents(u) and re.search(r"\b(mik|giup|may)\b", u) for u in users)  # không dấu
    assert any(re.search(r"\b(ko|dc|bn|vs|bt)\b", u) for u in users)                            # teencode
    for c in cs:
        for r, t in c:
            if t.startswith(("<tool>sys_", "<tool>droid_", "<tool>store_install", "<tool>server_", "<tool>sh_goi_y")):
                assert t == wc.strip_accents(t) or t.startswith("<tool>store_search"), t         # đối số chuẩn hoá không dấu


def test_os_data_never_teaches_destructive_commands():
    blob = "\n".join(t for c in _convs(1500, 4) for _, t in c)
    for bad in ("rm -rf", "mkfs", "dd if=", ":(){", "chmod -R 777 /", "> /dev/sd"):
        assert bad not in blob
    assert "mik không đưa lệnh đó" in blob and "mik không tắt" in blob                         # có dạy từ chối
    assert "từ chối:" in blob                                                                    # lệnh shell chỉ từ danh sách đã duyệt (sh_goi_y)


def test_os_deterministic_and_in_tool_data_make():
    a = [tdo.gen_os(random.Random(3), 300) for _ in range(5)]
    assert a == [tdo.gen_os(random.Random(3), 300) for _ in range(5)]
    import tool_data
    import tool_data_x
    assert "os" in tool_data_x.EXTRA_KINDS and "os=3" in tool_data_x.DEFAULT_MIX
    mix = {k: float(v) for k, v in (kv.split("=") for kv in tool_data_x.DEFAULT_MIX.split(","))}
    convs = tool_data.make(300, 5, tool_data.load_facts(None), [], 300, mix)
    assert len(convs) == 300
    assert any("<tool>sys_" in t or "<tool>store_" in t for c in convs for _, t in c)


def test_new_tool_regex_counts_os_tools():
    import make_data
    assert make_data._NEW_TOOL_RE.search('{"messages":[{"role":"assistant","content":"<tool>sys_power tat_may</tool>"}]}')
    assert make_data._NEW_TOOL_RE.search("<tool>store_install firefox</tool>")
    assert not make_data._NEW_TOOL_RE.search("<tool>calc 1+1</tool>")
