"""Thử graphify (yui_graph.py): sửa khoảng lệch Recall <-> độ chính xác trả lời. Gồm thử đơn vị + thử hồi quy trên bộ đo graphify_bench."""
import json
import os
import sys

import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import graphify_bench as gb  # noqa: E402
import yui_agent as ya  # noqa: E402
from yui_graph import Graphify, entities, fact_key, has_self, intent_of  # noqa: E402
from yui_rag import RAG, tokens  # noqa: E402
from yui_vault import Workspace  # noqa: E402


def mk(tmp_path, rows):
    """rows: [(ngày, người dùng, yui)] -> kho + RAG + Graphify."""
    ws = Workspace(tmp_path / "k")
    with open(f"{ws.root}/chat/a.jsonl", "w", encoding="utf-8") as f:
        for day, u, b in rows:
            f.write(json.dumps({"ts": 1_760_000_000 + day * 86400, "messages": [{"role": "user", "content": u}, {"role": "assistant", "content": b}]}, ensure_ascii=False) + "\n")
    rag = RAG(ws); rag.sync()
    return ws, rag, Graphify(rag)


def test_vietnamese_homographs_do_not_break_matching():
    assert "chó" in tokens("mik nuôi chó") and "cho" in tokens("mik nuôi chó")            # "chó" KHÔNG bị coi là từ dừng "cho"
    assert "cho" not in tokens("cho mik hỏi") and "cho" not in tokens("Cho mik hỏi")      # nhưng "cho" (đưa/để) vẫn là từ dừng
    assert "minh" in tokens("bạn Minh ở Huế") and "minh" not in tokens("mình ở Huế")       # "Minh" là tên, "mình" là đại từ
    assert not has_self("bạn Minh ở Huế") and has_self("mình ở Huế") and has_self("mik ở Huế")
    assert "p:lan" in entities("Lan nuôi mèo tên là Xíu") and entities("mik đi Đà Lạt chơi") == ["p:da lat"]          # "đi" viết thường KHÔNG phải tên riêng
    assert entities("hôm nay mik ở nhà được nghỉ") == []
    assert "n:5000000" in entities("tiết kiệm 5 triệu") and "d:14/3" in entities("sinh nhật 14/3")


def test_durable_keys_do_not_merge_different_subjects():
    assert fact_key("mèo nhà mik tên là A") != fact_key("mèo của hàng xóm tên là B") and fact_key("mèo nhà mik tên là A") == fact_key("mèo nhà mik đổi tên thành B")
    assert fact_key("mik hay quên tên người ta") is None and fact_key("mik thích ăn phở") is None
    assert fact_key("bạn Lan đang ở Huế") == fact_key("dạo này bạn Lan chuyển về Đà Lạt rồi") != fact_key("bạn Minh đang ở Huế")
    assert intent_of("lúc đầu mik ở đâu") == "history" and intent_of("sinh nhật bạn Lan là ngày nào") == "when" and intent_of("giá bao nhiêu") == "howmany"


def test_latest_value_wins_history_gets_oldest_and_echo_is_dropped(tmp_path):
    ws, rag, g = mk(tmp_path, [(1, "bạn Lan đang ở Đà Lạt", "mik nhớ rồi nha: bạn Lan đang ở Đà Lạt"), (30, "bạn Lan chuyển sang Huế sống", "ok nè"),
                               (60, "dạo này bạn Lan chuyển về Vũng Tàu rồi", "mik nhớ rồi nha: dạo này bạn Lan chuyển về Vũng Tàu rồi")] + [(i, "hôm nay trời đẹp nhỉ", "ừ đẹp thật") for i in range(2, 20)])
    now = g.recall("bây giờ bạn Lan đang ở đâu", 300)
    assert "Vũng Tàu" in now and "Đà Lạt" not in now and "Huế" not in now and "Yui" not in now        # chỉ giá trị MỚI NHẤT, không có lời Yui nhại lại
    old = g.recall("lúc đầu bạn Lan ở đâu", 300)
    assert "Đà Lạt" in old
    assert g.stats()["superseded"] == 2


def test_compact_format_budget_and_abstain(tmp_path):
    ws, rag, g = mk(tmp_path, [(1, "mèo nhà mik tên là Mướp, nó 3 tuổi và rất thích ngủ trên ghế sofa mỗi buổi chiều khi trời nắng", "dễ thương quá"), (2, "hôm nay trời đẹp nhỉ", "ừ")])
    out = g.recall("mèo nhà mik tên gì", 90)
    assert out.startswith("[") and "] bạn: " in out and "Mướp" in out and len(out) <= 90 and "p:" not in out and "c:" not in out        # câu thường, KHÔNG có mã nút/tam thể
    assert g.recall("mik có con rùa tên gì", 300) == "không tìm thấy trong lịch sử"                                                     # chưa từng kể -> từ chối, không đưa nhiễu
    assert g.recall("xe máy hiệu gì", 300) == "không tìm thấy trong lịch sử"


def test_multi_hop_bridge(tmp_path):
    rows = [(1, "bạn thân của mik là Lan", "ok"), (2, "Lan nuôi một con mèo tên là Xíu", "ok"), (3, "Thảo nuôi một con mèo tên là Bơ", "ok"), (4, "mèo nhà mik tên là Tèo", "ok")] \
        + [(5 + i, f"Hùng nuôi một con mèo tên là M{i}", "ok") for i in range(6)]
    ws, rag, g = mk(tmp_path, rows)
    out = g.recall("mèo của bạn thân mik tên gì", 300)
    assert "Xíu" in out                                                                                                              # đi qua "bạn thân" -> Lan -> mèo của Lan


def test_persistence_reload_and_survives_rag_rebuild(tmp_path):
    ws, rag, g = mk(tmp_path, [(1, "mèo nhà mik tên là Mướp", "ok"), (2, "bạn Lan đang ở Huế", "ok")])
    g.recall("mèo tên gì"); g.flush()
    assert os.path.exists(f"{ws.root}/index/graphify.bin")
    rag2 = RAG(ws); rag2.sync(); g2 = Graphify(rag2)
    assert g2.stats()["facts"] == g.stats()["facts"] and "Mướp" in g2.recall("mèo tên gì")
    os.unlink(f"{ws.root}/index/index.bin"); open(f"{ws.root}/chat/0b.jsonl", "w", encoding="utf-8").write(json.dumps({"ts": 1, "messages": [{"role": "user", "content": "chó của mik tên là Gạo"}, {"role": "assistant", "content": "ok"}]}))
    rag3 = RAG(ws); rag3.sync(); g3 = Graphify(rag3)                                    # RAG dựng lại với mã đoạn khác: đồ thị cũ phải bị bỏ chứ không trỏ nhầm
    assert "Gạo" in g3.recall("chó tên gì") and "Mướp" in g3.recall("mèo tên gì")
    assert any(e["where"] == "graph.load" for e in ws.recent_errors())


def test_regression_graphify_closes_recall_vs_answer_gap():
    runs = [gb.evaluate(seed=s, noise=400) for s in (1, 2, 3)]
    avg = lambda sysn, k: sum(r[sysn][k] for r in runs) / len(runs)  # noqa: E731
    g_rec, g_acc, g_ctx = avg("graphify", "recall"), avg("graphify", "acc"), avg("graphify", "ctx")
    assert g_acc >= 0.85 and g_rec - g_acc <= 0.08                                                  # tìm đúng thì trả lời đúng: khoảng lệch nhỏ
    assert g_acc > avg("bm25", "acc") + 0.15 and g_acc > avg("dense", "acc") + 0.3                  # hơn BM25 thường và hơn "đổ cả subgraph"
    assert avg("dense", "recall") - avg("dense", "acc") > 0.25                                      # tái hiện đúng hiện tượng: subgraph đặc -> recall cao nhưng trả lời kém
    assert g_ctx < avg("dense", "ctx") / 5 and avg("graphify", "stale") == 0 and avg("graphify", "noans") >= 0.75


def test_toolbox_uses_graphify_by_default_and_can_switch_off(tmp_path):
    root = tmp_path / "kho"; ws = Workspace(root)
    with open(f"{root}/chat/a.jsonl", "w", encoding="utf-8") as f:
        f.write(json.dumps({"ts": 1_760_000_000, "messages": [{"role": "user", "content": "mèo nhà mik tên là Mướp"}, {"role": "assistant", "content": "mik nhớ rồi nha: mèo nhà mik tên là Mướp"}]}, ensure_ascii=False))
    tb = ya.ToolBox(ya.Profile({"memory": {"dir": str(root)}}))
    out = tb.call("recall", "mèo tên gì")
    assert "Mướp" in out and "Yui" not in out and "mik nhớ rồi" not in out                         # bản graphify: không có lời nhại lại
    tb2 = ya.ToolBox(ya.Profile({"memory": {"dir": str(root), "graph": False}}))
    assert "Mướp" in tb2.call("recall", "mèo tên gì")
    tb.close()
