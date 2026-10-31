#!/usr/bin/env python3
"""
yui_orch.py - BỘ ĐIỀU PHỐI nhiều bước cho model Yui: cho model gọi liên tiếp recall / py / calc / skill / note / search... (lần 2, 3, ... nhiều lần) cho tới khi đủ
để trả lời tốt nhất, nhưng LUÔN có trần để không treo.

TRẦN AN TOÀN (đều chỉnh được):  max_steps (mặc định 5 lần gọi công cụ) | max_seconds (90s cả lượt) | chống lặp (cùng một lệnh quá max_repeat lần = dừng)
  | mỗi công cụ vẫn có giới hạn riêng của yui_agent (CPU/RAM/thời gian/mạng).
KHÔNG QUÊN TIẾN ĐỘ (model nhỏ, cửa sổ ngữ cảnh ít):
  * Sổ tiến độ ghi ra runs/<mã>/progress.json sau MỖI bước (mã hoá nếu kho có khoá): câu hỏi, các bước đã làm + kết quả, file đã tạo, trạng thái.
  * Mỗi lần gọi model, hệ thống tự xếp lại prompt cho VỪA ngân sách ctx_chars: giữ lịch sử gần nhất nếu còn chỗ; hết chỗ thì chỉ giữ câu hỏi + bước gần nhất
    và kèm "[tiến độ: 1.recall ...→...; 2.py ...→...]" (bản rút gọn của sổ) trong kết quả công cụ.
  * Khi xong (hoặc chạm trần), hệ thống TỔNG HỢP ngắn gọn các bước thành "[tóm tắt: ...]" rồi gửi cho model viết câu trả lời cuối (từ summarize_from bước trở lên,
    hoặc khi bị buộc dừng), nên model nhỏ không phải nuốt cả chuỗi dài.
DỌN DẸP: thư mục runs/<mã>/ là TẠM. Chạy xong (done) thì xoá (keep_runs="failed", mặc định), lỗi/hết giờ/lặp thì GIỮ để xem + tiếp tục (resume=mã) và ghi vào
  logs/errors.jsonl (luôn giữ). keep_runs="all" giữ hết, "none" xoá cả khi lỗi (vẫn ghi nhật ký). Run cũ hơn keep_days ngày tự dọn.
Câu hỏi + câu trả lời cuối được rag.remember() lưu vào lịch sử chat (nếu auto_remember), nên không cần giữ chi tiết từng bước vĩnh viễn.
"""
import json
import os
import shutil
import time

from yui_agent import FALLBACK, parse_call, strip_tool_tags
from yui_vault import new_id

DEFAULTS = dict(max_steps=5, max_seconds=90, max_repeat=2, ctx_chars=700, digest_chars=220, summary_chars=300, summarize_from=3,
                keep_runs="failed", keep_days=14, history_turns=3, auto_remember=True)
BAD = ("lỗi", "bị chặn", "không có công cụ")


def _short(s, n):
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[:max(0, n - 1)] + "…"


def fmt_progress(steps, limit=220):
    """'1.recall tên mèo→Mướp; 2.py x*2→10' — bản rút gọn của sổ tiến độ, ưu tiên các bước gần nhất khi quá dài."""
    parts = [f"{i}.{s['tool']} {_short(s['arg'], 22)}→{_short(s['res'], 34)}" for i, s in enumerate(steps, 1)]
    out = "; ".join(parts)
    if len(out) <= limit:
        return out
    while len(parts) > 1 and len("…; " + "; ".join(parts)) > limit:
        parts.pop(0)
    return _short("…; " + "; ".join(parts), limit)


def fmt_summary(steps, limit=300):
    """Tóm tắt cuối: các bước đã làm + kết quả cuối cùng (đầy đủ hơn), để model chỉ cần viết câu trả lời."""
    if not steps:
        return ""
    last = steps[-1]
    head = fmt_progress(steps[:-1] + [dict(last, res=_short(last["res"], 12))], max(60, limit // 2))
    return _short(f"[tóm tắt: {head}] kết quả cuối: {last['res']}", limit)


class Run:
    def __init__(self, ws, question, run_id=None):
        self.ws, self.id = ws, run_id or new_id("run-")
        self.rel = f"{self.id}/progress.json"
        self.data = {"id": self.id, "question": _short(question, 400), "status": "running", "started": int(time.time()), "steps": [], "files": []}
        if ws is not None:
            try:
                p = ws.find(self.rel, "runs")
                if run_id and p:
                    self.data = json.loads(ws.read_text(p))
            except Exception as e:  # noqa: BLE001
                ws.log_error("orch.run.load", e, run=self.id)

    def save(self, status=None):
        if status:
            self.data["status"] = status
        self.data["updated"] = int(time.time())
        if self.ws is None:
            return
        try:
            self.ws.write_text(self.rel, json.dumps(self.data, ensure_ascii=False), sub="runs")
        except Exception as e:  # noqa: BLE001 - sổ tiến độ hỏng không được làm hỏng câu trả lời
            self.ws.log_error("orch.run.save", e, run=self.id)

    def delete(self):
        if self.ws is None:
            return
        try:
            shutil.rmtree(self.ws.safe(self.id, "runs"), ignore_errors=True)
        except Exception as e:  # noqa: BLE001
            self.ws.log_error("orch.run.delete", e, run=self.id)


def prune_runs(ws, days):
    """Xoá run cũ hơn `days` ngày (kể cả run bị bỏ dở)."""
    if ws is None:
        return 0
    n, base, now = 0, os.path.join(ws.root, "runs"), time.time()
    try:
        for d in os.listdir(base):
            p = os.path.join(base, d)
            if os.path.isdir(p) and now - os.path.getmtime(p) > days * 86400:
                shutil.rmtree(p, ignore_errors=True)
                n += 1
    except OSError:
        pass
    return n


def pending_runs(ws):
    """Các run chưa xong (status != done) còn lưu: [(mã, trạng thái, câu hỏi)] để người dùng có thể tiếp tục."""
    out = []
    if ws is None:
        return out
    for p, _, _ in ws.list_files("runs", exts=(".json",)):
        try:
            d = json.loads(ws.read_text(p))
            if d.get("status") != "done":
                out.append((d["id"], d["status"], d.get("question", "")))
        except Exception:  # noqa: BLE001
            continue
    return out


class Orchestrator:
    def __init__(self, llm, toolbox, ws=None, rag=None, **cfg):
        self.llm, self.tb, self.ws, self.rag = llm, toolbox, ws, rag
        self.cfg = dict(DEFAULTS)
        self.cfg.update({k: v for k, v in cfg.items() if k in DEFAULTS})
        if ws is not None:
            prune_runs(ws, self.cfg["keep_days"])

    # ---------------- xếp prompt vừa ngân sách ----------------
    def _pairs(self, steps, prefix=""):
        out = []
        for i, s in enumerate(steps):
            res = s["res"]
            if prefix and i == len(steps) - 1:
                res = f"{prefix} {res}"
            out += [{"role": "assistant", "content": f"<tool>{s['tool']} {s['arg']}</tool>".strip()}, {"role": "user", "content": f"<result>{res}</result>"}]
        return out

    @staticmethod
    def _size(msgs):
        return sum(len(m["content"]) + 8 for m in msgs)

    def build(self, history, question, steps, final=False, forced=False):
        c = self.cfg
        hist = [dict(m, content=_short(m["content"], 160)) for m in list(history)[-2 * c["history_turns"]:]]
        q = {"role": "user", "content": question}
        if final and steps:
            note = fmt_summary(steps, c["summary_chars"]) + (" (hết lượt công cụ, hãy trả lời ngay)" if forced else "")
            last = steps[-1]
            return [q, {"role": "assistant", "content": f"<tool>{last['tool']} {last['arg']}</tool>".strip()}, {"role": "user", "content": f"<result>{note}</result>"}]
        while True:
            msgs = hist + [q] + self._pairs(steps)
            if self._size(msgs) <= c["ctx_chars"] or not hist:
                break
            hist = hist[2:]
        if self._size(msgs) > c["ctx_chars"] and len(steps) > 1:   # hết chỗ: chỉ giữ câu hỏi + bước gần nhất, kèm tiến độ rút gọn
            prog = f"[tiến độ: {fmt_progress(steps[:-1], c['digest_chars'])}]"
            msgs = [dict(q, content=_short(question, 300))] + self._pairs(steps[-1:], prefix=prog)
        return msgs

    # ---------------- chạy ----------------
    def ask(self, question, history=None, resume=None, cancel=None):
        """-> (câu trả lời, info). info: status (done | cancelled | timeout | max_steps | loop | error), steps, run_id, seconds.
        cancel: threading.Event; set() là DỪNG NGAY (nút Dừng): không gọi model/công cụ nữa, py đang chạy bị giết, run được giữ lại để tiếp tục."""
        c, history, t0 = self.cfg, list(history or []), time.monotonic()
        run = Run(self.ws, question, resume)
        steps, seen, loops = run.data["steps"], {}, 0
        run.save()
        names = set(self.tb.names())
        answer, status, forced = None, "done", False
        self.tb.cancel = cancel
        stopped = lambda: cancel is not None and cancel.is_set()   # noqa: E731
        try:
            while True:
                if stopped():
                    status = "cancelled"
                    break
                if time.monotonic() - t0 > c["max_seconds"]:
                    status, forced = "timeout", True
                    break
                out = self.llm(self.build(history, question, steps))
                if stopped():
                    status = "cancelled"
                    break
                call = parse_call(out, names)
                if call is None:
                    answer = strip_tool_tags(out) or None
                    break
                if len(steps) >= c["max_steps"]:
                    status, forced = "max_steps", True
                    break
                name, arg = call
                key = (name, " ".join(arg.lower().split()))
                seen[key] = seen.get(key, 0) + 1
                if seen[key] > c["max_repeat"]:
                    res, loops = "(đã làm lệnh này rồi, hãy dùng kết quả trước đó)", loops + 1
                    if loops >= 2:
                        status, forced = "loop", True
                        break
                else:
                    res = self.tb.call(name, arg, who=run.id)
                ok = not res.startswith(BAD)
                steps.append({"tool": name, "arg": _short(arg, 200), "res": _short(res, 400), "ok": ok, "t": round(time.monotonic() - t0, 1)})
                if not ok and self.ws is not None:
                    self.ws.log_error(f"tool:{name}", res, run=run.id)
                if ok and name == "write" and res.startswith("đã ghi"):
                    run.data["files"].append(arg.split("|", 1)[0].strip()[:120])
                run.save()
            if status == "cancelled":
                answer = "Mik đã dừng lại theo ý bạn nha."
            elif forced or (len(steps) >= c["summarize_from"] and steps):
                final_out = self.llm(self.build(history, question, steps, final=True, forced=forced))
                answer = strip_tool_tags(final_out) or answer
            if not answer:
                answer = (f"Mik tìm được: {steps[-1]['res']}" if steps and steps[-1]["ok"] else FALLBACK)
        except Exception as e:  # noqa: BLE001 - mất mạng/model lỗi: không treo, không văng lỗi thô cho người dùng
            status, answer = "error", FALLBACK
            if self.ws is not None:
                self.ws.log_error("orch.ask", f"{type(e).__name__}: {e}", run=run.id)
        run.data["answer"] = _short(answer, 300)
        run.save(status)
        if status not in ("done", "cancelled") and self.ws is not None:
            self.ws.log_error("orch.status", f"lượt kết thúc: {status} sau {len(steps)} bước", run=run.id)
        if status == "done" and c["keep_runs"] != "all" or c["keep_runs"] == "none":
            run.delete()
        self.tb.cancel = None
        if self.rag is not None and c["auto_remember"] and answer != FALLBACK and status != "cancelled":
            try:
                self.rag.remember(question, answer)
            except Exception as e:  # noqa: BLE001
                self.ws.log_error("orch.remember", e, run=run.id)
        return answer, {"status": status, "steps": steps, "run_id": run.id, "seconds": round(time.monotonic() - t0, 1)}
