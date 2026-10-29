"""Thử đặt tên model / thư mục models/ / tìm repo data, bằng HfApi giả (không cần mạng, không cần torch)."""
import argparse, json, os, sys
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import model_registry as mr  # noqa: E402
import data_layout as dl  # noqa: E402


class Item:
    def __init__(self, i): self.id = i


class FakeApi:
    def __init__(self, repos, user_models=(), user_datasets=()):
        self.repos, self.um, self.ud = repos, user_models, user_datasets

    def list_repo_files(self, repo, repo_type="dataset"):
        if repo not in self.repos:
            raise FileNotFoundError(repo)
        return self.repos[repo]

    def list_models(self, author=None): return [Item(f"{author}/{n}") for n in self.um]
    def list_datasets(self, author=None): return [Item(f"{author}/{n}") for n in self.ud]
    def whoami(self): return {"name": "BachDaThan"}


def A(**kw):
    d = dict(hf_repo="BachDaThan/TrainAI-1B", hf_type="dataset", name=None, new_model=False, local=None, src=None)
    d.update(kw)
    return argparse.Namespace(**d)


CKPT = ["config.json", "tokenizer.json", "last.pt", "best.pt"]


def test_new_repo_gets_v1_and_folder():
    a = A(); mr.setup_identity(a, "1B", FakeApi({"BachDaThan/TrainAI-1B": []}), log=lambda *x: None)
    assert (a.name, a.hf_dir) == ("yui-1b-v1", "models/yui-1b-v1/")


def test_explicit_name_is_used_and_sanitized():
    a = A(name="Yui 1B Đặc Biệt!"); 
    with pytest.raises(ValueError):
        mr.sanitize_name("!!!")
    a = A(name="my model v2"); mr.setup_identity(a, "1B", FakeApi({"BachDaThan/TrainAI-1B": []}), log=lambda *x: None)
    assert a.name == "my-model-v2" and a.hf_dir == "models/my-model-v2/"


def test_resume_latest_then_new_model_and_never_collides():
    files = [f"models/yui-1b-v{n}/{f}" for n in (1, 2) for f in CKPT]
    api = FakeApi({"BachDaThan/TrainAI-1B": files}, user_datasets=["yui-1b-v3"])      # v3 đã bị repo khác của tài khoản chiếm
    a = A(); mr.setup_identity(a, "1B", api, log=lambda *x: None)
    assert a.name == "yui-1b-v2"                                                      # tiếp tục bản mới nhất
    a = A(new_model=True); mr.setup_identity(a, "1B", api, log=lambda *x: None)
    assert a.name == "yui-1b-v4"                                                      # bản mới, tránh cả v3 của repo khác
    a = A(); mr.setup_identity(a, "7B", api, log=lambda *x: None)
    assert a.name == "yui-7b-v1"                                                      # cỡ khác -> dãy tên riêng


def test_legacy_root_repo_is_kept():
    a = A(); mr.setup_identity(a, "50M", FakeApi({"BachDaThan/TrainAI-1B": CKPT}), log=lambda *x: None)
    assert a.name is None and a.hf_dir == ""
    a = A(name="yui-50m-v1"); mr.setup_identity(a, "50M", FakeApi({"BachDaThan/TrainAI-1B": CKPT}), log=lambda *x: None)
    assert a.hf_dir == "models/yui-50m-v1/"


def test_sft_picks_newest_pretrain_and_names_output():
    files = [f"models/yui-1b-v{n}/{f}" for n in (1, 2) for f in CKPT] + ["models/yui-1b-v1-sft/config.json", "models/yui-1b-v1-sft/sft_best.pt"]
    a = A(); mr.setup_sft(a, FakeApi({"BachDaThan/TrainAI-1B": files}), log=lambda *x: None)
    assert (a.src, a.hf_dir, a.out_name, a.sft_dir) == ("yui-1b-v2", "models/yui-1b-v2/", "yui-1b-v2-sft", "models/yui-1b-v2-sft/")
    a = A(src="yui-1b-v1"); mr.setup_sft(a, FakeApi({"BachDaThan/TrainAI-1B": files}), log=lambda *x: None)
    assert a.out_name == "yui-1b-v1-sft"                                              # chạy lại -> cập nhật tiến độ cũ
    a = A(src="yui-1b-v1", new_model=True); mr.setup_sft(a, FakeApi({"BachDaThan/TrainAI-1B": files}), log=lambda *x: None)
    assert a.out_name == "yui-1b-v1-sft2"
    a = A(src="yui-1b-v9")
    with pytest.raises(SystemExit):
        mr.setup_sft(a, FakeApi({"BachDaThan/TrainAI-1B": files}), log=lambda *x: None)
    a = A(); mr.setup_sft(a, FakeApi({"BachDaThan/TrainAI-1B": CKPT}), log=lambda *x: None)   # repo kiểu cũ
    assert a.hf_dir == "" and a.sft_dir == ""


def test_generate_prefers_finetuned_newest():
    files = [f"models/yui-1b-v1/{f}" for f in CKPT] + ["models/yui-1b-v1-sft/config.json"]
    a = A(); mr.setup_generate(a, FakeApi({"BachDaThan/TrainAI-1B": files}), log=lambda *x: None)
    assert a.hf_dir == "models/yui-1b-v1-sft/"


def test_find_data_repo_across_repos():
    api = FakeApi({"BachDaThan/TrainAI-1B": ["models/x/config.json"], "BachDaThan/TrainAI-web": ["Fine-tune/1B/1.txt", "Fine-tune/1B/2.txt", "data/1B/1.txt"]})
    repo, _ = dl.find_data_repo(["BachDaThan/TrainAI-1B", None, "BachDaThan/TrainAI-web"], dl.finetune_prefix("1B"), api=api)
    assert repo == "BachDaThan/TrainAI-web"                                           # data ở repo khác repo model vẫn tìm ra
    assert dl.find_data_repo(["BachDaThan/TrainAI-1B"], dl.data_prefix("1B"), api=api) == (None, [])


def test_identity_data_mentions_name_and_loads():
    from sft_format import load_convs
    convs, stats = load_convs({"identity.jsonl": mr.identity_jsonl("yui-1b-v2-sft").decode("utf-8")})
    assert len(convs) >= 20 and stats["hỏng"] == 0
    assert any("yui-1b-v2-sft" in a for c in convs for _, a in c)


def test_heads_and_arch_up_to_10b():
    for d in (256, 384, 512):
        assert dl.heads_for(d) == 4                                                   # preset cũ giữ nguyên
    for t in ("1B", "3B", "7B", "10B"):
        n = dl.parse_params(t); d, l, e = dl.arch_for_params(n); h = dl.heads_for(d)
        assert abs(dl.params_from_arch(16000, d, l, e) / n - 1) < 0.05 and d % (2 * h) == 0 and d // h <= 128 and e <= 256
