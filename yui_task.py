#!/usr/bin/env python3
"""
yui_task.py - TÁC VỤ NỀN có TIẾN ĐỘ và NÚT DỪNG (kill) cho Yui. Chỉ thư viện chuẩn; dùng cho server/desktop/Termux, và là bản tham chiếu cho app Flutter (Y-App).

  tm = TaskManager(notify=hàm_thông_báo)      # notify(task) được gọi khi tác vụ XONG/LỖI/BỊ DỪNG (app đẩy thành thông báo hệ thống)
  t  = tm.start("hỏi", lambda task: orch.ask(q, hist, cancel=task.cancel))   # chạy ngầm trong luồng riêng, trả về ngay
  t.progress, t.message  -> thanh tiến độ;  tm.stop(t.id) -> NÚT DỪNG: set cờ cancel, giết sandbox py đang chạy, không gọi thêm model/công cụ
  tm.stop_all()                                # dừng khẩn cấp mọi thứ (giải phóng CPU/RAM/pin)
Mặc định mỗi lúc chạy 1 tác vụ nặng (max_parallel=1), tác vụ sau xếp hàng; tác vụ treo quá `timeout_s` bị tự dừng.
Luồng Python không thể bị giết cưỡng bức, nên "dừng" = (1) cờ cancel được kiểm tra giữa mỗi bước/vòng lặp, (2) tiến trình con py bị kill ngay, (3) mọi công cụ khác đều có hạn thời gian riêng
(yui_agent.limits). Trên app Flutter, nút Dừng còn huỷ Isolate/engine C++ (xem Y-App-patch/).
"""
import threading
import time
import uuid

QUEUED, RUNNING, DONE, ERROR, CANCELLED = "queued", "running", "done", "error", "cancelled"


class Task:
    def __init__(self, kind, fn, timeout_s):
        self.id, self.kind, self.fn, self.timeout_s = uuid.uuid4().hex[:8], kind, fn, timeout_s
        self.status, self.progress, self.message, self.result, self.error = QUEUED, 0, "", None, ""
        self.cancel = threading.Event()
        self.created, self.started, self.finished = time.time(), None, None
        self.done_evt = threading.Event()

    def report(self, pct=None, message=None):
        if pct is not None:
            self.progress = max(0, min(100, int(pct)))
        if message is not None:
            self.message = str(message)[:120]

    def info(self):
        return {"id": self.id, "kind": self.kind, "status": self.status, "progress": self.progress, "message": self.message, "error": self.error,
                "seconds": round((self.finished or time.time()) - (self.started or self.created), 1)}


class TaskManager:
    def __init__(self, notify=None, max_parallel=1, timeout_s=600, keep=20):
        self.notify, self.max_parallel, self.timeout_s, self.keep = notify, max_parallel, timeout_s, keep
        self.tasks, self._lock, self._queue = {}, threading.Lock(), []
        self._watch = threading.Thread(target=self._watchdog, daemon=True)
        self._watch.start()

    def start(self, kind, fn, timeout_s=None):
        t = Task(kind, fn, timeout_s or self.timeout_s)
        with self._lock:
            self.tasks[t.id] = t
            self._queue.append(t)
            for old in [k for k, v in self.tasks.items() if v.finished and len(self.tasks) > self.keep]:
                self.tasks.pop(old, None)
        self._pump()
        return t

    def _pump(self):
        with self._lock:
            running = sum(1 for t in self.tasks.values() if t.status == RUNNING)
            while self._queue and running < self.max_parallel:
                t = self._queue.pop(0)
                if t.cancel.is_set():
                    self._finish(t, CANCELLED)
                    continue
                t.status, t.started = RUNNING, time.time()
                threading.Thread(target=self._run, args=(t,), daemon=True).start()
                running += 1

    def _run(self, t):
        try:
            t.result = t.fn(t)
            status = CANCELLED if t.cancel.is_set() else DONE
        except Exception as e:  # noqa: BLE001 - tác vụ lỗi không được làm sập app
            t.error, status = f"{type(e).__name__}: {e}"[:200], (CANCELLED if t.cancel.is_set() else ERROR)
        if status == DONE:
            t.progress = 100
        self._finish(t, status)
        self._pump()

    def _finish(self, t, status):
        if t.finished:
            return
        t.status, t.finished = status, time.time()
        t.done_evt.set()
        if self.notify:
            try:
                self.notify(t)
            except Exception:  # noqa: BLE001
                pass

    def _watchdog(self):
        while True:
            time.sleep(0.2)
            for t in list(self.tasks.values()):
                if t.status == RUNNING and t.started and time.time() - t.started > t.timeout_s and not t.cancel.is_set():
                    t.error = f"quá {int(t.timeout_s)}s nên bị dừng"
                    t.cancel.set()

    def stop(self, task_id):
        t = self.tasks.get(task_id)
        if not t or t.finished:
            return False
        t.cancel.set()
        if t.status == QUEUED:
            with self._lock:
                if t in self._queue:
                    self._queue.remove(t)
            self._finish(t, CANCELLED)
        return True

    def stop_all(self):
        return sum(1 for tid in list(self.tasks) if self.stop(tid))

    def wait(self, task_id, timeout=None):
        t = self.tasks[task_id]
        t.done_evt.wait(timeout)
        return t

    def list(self):
        return [t.info() for t in self.tasks.values()]


def ask_task(tm, orch, question, history=None, resume=None):
    """Hỏi model chạy nền qua bộ điều phối (nhiều bước); task.result = (câu trả lời, info). Tiến độ = số bước đã làm."""
    def fn(task):
        task.report(5, "đang suy nghĩ")
        ans, info = orch.ask(question, history, resume, cancel=task.cancel)
        task.report(100, info["status"])
        return ans, info
    return tm.start("hỏi", fn)


def kb_task(tm, kb, force=False):
    """Nạp/cô đọng tài liệu nền, có thanh tiến độ + dừng được; task.result = báo cáo."""
    return tm.start("nạp tài liệu", lambda task: kb.ingest(on_progress=lambda p, s, n: task.report(p, f"{s} {n}".strip()), stop=task.cancel, force=force), timeout_s=3600)
