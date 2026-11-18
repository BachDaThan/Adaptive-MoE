"""Chính sách tầng cào web (P9-23): mặc định TẮT, chỉ chạy khi YUI_ALLOW_SCRAPE=1."""
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import yui_web as yw  # noqa: E402
from test_yui_extra import BING, DDG_HTML, web  # noqa: E402


@pytest.fixture(autouse=True)
def clean(monkeypatch):
    monkeypatch.delenv("YUI_ALLOW_SCRAPE", raising=False)
    yw.reset_state()
    yw._WARNED.clear()
    yield


TABLE = {"https://html.duckduckgo.com": (200, "text/html", DDG_HTML), "https://www.bing.com": (200, "text/html", BING)}


def test_default_makes_zero_requests_to_scrape_tiers():
    w, net = web(TABLE, cache_s=0, min_interval_s=0, order=["ddg", "bing", "brave", "google", "kb"], google=True)
    w.search("Hà Nội")
    hosts = [u.split("/")[2] for _, u, _ in net.calls]
    assert not any(h in ("html.duckduckgo.com", "www.bing.com", "search.brave.com", "www.google.com") for h in hosts)


def test_env_flag_enables_scraping_and_warns_once(monkeypatch, capsys):
    monkeypatch.setenv("YUI_ALLOW_SCRAPE", "1")
    w, net = web(TABLE, cache_s=0, min_interval_s=0, order=["ddg"])
    assert "Hà Nội" in w.search("Hà Nội")
    assert [u.split("/")[2] for _, u, _ in net.calls] == ["html.duckduckgo.com"]
    w.search("Hà Nội khác")
    assert capsys.readouterr().err.count("CẢNH BÁO") == 1


def test_cfg_flag_also_enables():
    w, net = web(TABLE, cache_s=0, min_interval_s=0, order=["ddg"], allow_scrape=True)
    w.search("Hà Nội")
    assert net.calls and net.calls[0][1].startswith("https://html.duckduckgo.com")
