import os
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "tools"))
import scan_secrets as ss  # noqa: E402

FAKES = {
    "khoá riêng": "-----BEGIN OPENSSH PRIVATE KEY-----",  # scan-secrets: allow
    "Hugging Face": "hf_" + "a" * 34,
    "Google API": "AIza" + "B" * 35,
    "sk-": "sk-" + "c" * 30,
    "GitHub": "ghp_" + "d" * 30,
    "Tavily": "tvly-" + "e" * 20,
    "AWS": "AKIA" + "F" * 16,
    "Slack": "xoxb-" + "1234567890",
    "gán bí mật": 'api_key = "' + "g" * 30 + '"',
}


def test_catches_every_kind():
    for kind, fake in FAKES.items():
        assert any(h[2] == kind for h in ss.scan_text("x = " + fake, "t")), kind


def test_allow_comment_and_clean_text():
    assert ss.scan_text("x = " + FAKES["Hugging Face"] + "  # scan-secrets: allow", "t") == []
    assert ss.scan_text("def f():\n    return 'hello world'\n", "t") == []


def test_tree_and_names(tmp_path):
    (tmp_path / "a.txt").write_text("k=" + FAKES["GitHub"])
    (tmp_path / "release.jks").write_bytes(b"\x00")
    (tmp_path / ".env.example").write_text("API_KEY=")
    kinds = {h[2] for h in ss.scan_tree(str(tmp_path))}
    assert "GitHub" in kinds and "tên tệp nhạy cảm" in kinds


def test_history_catches_removed_secret(tmp_path):
    def git(*a):
        subprocess.run(["git", "-C", str(tmp_path), "-c", "user.email=a@b", "-c", "user.name=t", *a], check=True, capture_output=True)
    git("init", "-q")
    f = tmp_path / "x.py"
    f.write_text("k = '" + FAKES["Hugging Face"] + "'\n")
    git("add", "."); git("commit", "-qm", "a")
    f.write_text("k = None\n")
    git("commit", "-qam", "b")
    assert ss.scan_tree(str(tmp_path)) == [] and any(h[2] == "Hugging Face" for h in ss.scan_history(str(tmp_path)))


def test_ignore_file(tmp_path):
    (tmp_path / ".scan-secrets-ignore").write_text("# giả\nsuper-fake-" + "x" * 5 + "\n")
    (tmp_path / "t.cpp").write_text('std::string pass = "super-fake-xxxxx' + "y" * 25 + '";')
    ss.load_ignore(str(tmp_path))
    try:
        assert ss.scan_tree(str(tmp_path)) == []
    finally:
        ss.IGNORE[:] = []
