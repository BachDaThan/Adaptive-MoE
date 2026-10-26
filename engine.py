"""
engine.py - bộ chạy (runtime) riêng cho Adaptive-MoE, kiểu Ollama: KV cache + sinh từng token + server HTTP.

  Chat ngay trong terminal:   python engine.py --out thu_muc_model
  Mở server (PC / VPS):       python engine.py --out thu_muc_model --serve --host 0.0.0.0 --port 11434 --api_key MAT_KHAU
  Tự tải model từ HF:         thêm --hf_repo BachDaThan/TrainAI-web   (repo private cần HF_TOKEN)

API (gọi giống Ollama / OpenAI nên hệ thống Yui dễ nối vào):
  POST /api/chat              {"messages":[{"role":"user","content":"..."}], "stream":true, "options":{"temperature":0.8}}
  POST /api/generate          {"prompt":"...", "stream":true}                  (viết tiếp văn bản thuần)
  POST /v1/chat/completions   chuẩn OpenAI, hỗ trợ "stream":true (SSE)
  GET  /api/tags, /v1/models, /health, và trang chat nhỏ ở "/" (mở được bằng trình duyệt điện thoại)
Nếu đặt --api_key thì các API cần header  Authorization: Bearer MAT_KHAU.

Về bộ nhớ ngữ cảnh: model được train với ngữ cảnh L token (config.json, mặc định 128) và KHÔNG được train với cửa sổ trượt,
nên engine giữ KV cache đúng trong [0, L). Khi cache đầy nó "dịch ngữ cảnh": dựng lại cache từ nửa cuối lịch sử (vị trí về lại 0..),
giống llama.cpp; nhờ đó chat dài bao lâu cũng được mà không ra ngoài vùng model đã học. Muốn nhớ lâu hơn thì phải train với --L lớn hơn.
"""
import argparse, json, os, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from sft_format import USER_TAG, clean, prompt_ids, to_pairs

MODEL_NAME = "yui-adaptive-moe"


class Engine:
    def __init__(self, out, ckpt="auto", device=None, k=2, hf_repo=None, hf_type="dataset"):
        import torch
        from tokenizers import Tokenizer
        from elastic_ctx import CtxLM
        os.makedirs(out, exist_ok=True)
        if hf_repo:
            from huggingface_hub import hf_hub_download
            for fn in ("config.json", "tokenizer.json", "sft_best.pt", "best.pt"):
                if not os.path.exists(os.path.join(out, fn)):
                    try:
                        hf_hub_download(repo_id=hf_repo, filename=fn, repo_type=hf_type, local_dir=out,
                                        token=os.environ.get("HF_TOKEN"))
                    except Exception as e:  # noqa: BLE001
                        if fn in ("config.json", "tokenizer.json"):
                            raise
                        print(f"[engine] không tải được {fn} ({str(e)[:60]})", flush=True)
        self.cfg = json.load(open(os.path.join(out, "config.json"), encoding="utf-8"))
        self.tok = Tokenizer.from_file(os.path.join(out, "tokenizer.json"))
        self.dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
        c = self.cfg
        self.model = CtxLM(vocab=c["vocab"], d=c["d"], n_layers=c["layers"], n_experts=c["experts"], d_ff=2 * c["d"],
                           min_k=1, max_k=8, top_p=c["top_p"], n_shared=1, compute_coef=0.0).to(self.dev)
        if ckpt == "auto":
            ckpt = "sft_best.pt" if os.path.exists(os.path.join(out, "sft_best.pt")) else "best.pt"
        ck = torch.load(os.path.join(out, ckpt), map_location=self.dev)
        self.model.load_state_dict(ck["model"])
        self.model.eval()
        self.ckpt, self.is_sft = ckpt, ckpt.startswith("sft")
        self.L = c["L"]
        self.eos = self.tok.token_to_id("<eos>")
        self.eos = -1 if self.eos is None else self.eos
        self.k = k
        self.name = MODEL_NAME
        self.lock = threading.Lock()   # model dùng chung, mỗi lúc chỉ phục vụ một yêu cầu (các yêu cầu khác xếp hàng)
        print(f"[engine] {ckpt} | thiết bị {self.dev} | ngữ cảnh {self.L} token | k={k} | chế độ chat={self.is_sft}", flush=True)

    # ---------- lõi: sinh token có KV cache ----------
    def _sample(self, logits, gen, temp, top_k, top_p, rep):
        import torch
        lg = logits.float().clone()
        if rep != 1.0 and gen:
            idx = torch.tensor(sorted(set(gen[-48:])), device=lg.device)
            v = lg[idx]
            lg[idx] = torch.where(v > 0, v / rep, v * rep)
        if temp <= 0:
            return int(lg.argmax())
        lg = lg / temp
        if 0 < top_k < lg.numel():
            lg[lg < lg.topk(int(top_k)).values[-1]] = -float("inf")
        p = lg.softmax(-1)
        if top_p < 1.0:
            sp, si = p.sort(descending=True)
            sp[(sp.cumsum(0) - sp) > top_p] = 0
            p = torch.zeros_like(p).scatter(0, si, sp)
            p = p / p.sum()
        return int(torch.multinomial(p, 1))

    def stream_ids(self, ids, max_new=120, temp=0.8, top_k=40, top_p=0.95, rep=1.1, k=None):
        """Sinh từng id token. Prefill 1 lần, sau đó mỗi bước chỉ nạp 1 token nhờ KV cache."""
        import torch
        L, dev, model = self.L, self.dev, self.model
        for b in model.blocks:
            b.moe.k_cap = int(k or self.k)
        hist = list(ids) or [self.eos if self.eos >= 0 else 0]
        keep = max(16, L // 2)

        def prefill(seq):
            lg, pr = model(torch.tensor([seq], dtype=torch.long, device=dev), None)
            return lg[0, -1], pr

        with torch.no_grad():
            logits, past = prefill(hist[-(L - 8):])
            gen = []
            for _ in range(int(max_new)):
                nxt = self._sample(logits, gen, temp, top_k, top_p, rep)
                if nxt == self.eos:
                    return
                gen.append(nxt)
                hist.append(nxt)
                yield nxt
                if past["n"] >= L:   # cache đầy: dịch ngữ cảnh (dựng lại từ nửa cuối)
                    logits, past = prefill(hist[-keep:])
                else:
                    lg, past = model(torch.tensor([[nxt]], dtype=torch.long, device=dev), past)
                    logits = lg[0, -1]

    # ---------- đổi id -> văn bản, phát từng đoạn mới ----------
    def _view(self, gen, chat):
        text = self.tok.decode(gen)
        if chat:
            text = text.lstrip()
            cut = text.find(USER_TAG)
            if cut >= 0:
                return text[:cut].rstrip(), True
        return text, False

    def _stream_text(self, ids, chat, **kw):
        gen, shown = [], ""
        for t in self.stream_ids(ids, **kw):
            gen.append(t)
            text, stop = self._view(gen, chat)
            if not stop:
                if text.endswith("\ufffd"):      # ký tự UTF-8 chưa đủ byte
                    continue
                if chat:                           # chưa phát phần có thể là đầu của "User:"
                    for n in range(len(USER_TAG) - 1, 0, -1):
                        if text.endswith(USER_TAG[:n]):
                            text = text[:-n]
                            break
            if text.startswith(shown) and len(text) > len(shown):
                yield text[len(shown):]
                shown = text
            if stop:
                return
        text, _ = self._view(gen, chat)
        if text.startswith(shown) and len(text) > len(shown):
            yield text[len(shown):]

    # ---------- giao diện cấp cao ----------
    def check_messages(self, messages):
        """messages kiểu OpenAI/Ollama -> (lịch sử [(user, yui)], câu user cuối). Bỏ qua role 'system'."""
        msgs = [m for m in (messages or []) if isinstance(m, dict) and str(m.get("role", "")).lower() != "system"]
        last = max((i for i, m in enumerate(msgs) if str(m.get("role", "")).lower() in ("user", "human")), default=None)
        if last is None or not clean(msgs[last].get("content", "")):
            raise ValueError("cần ít nhất một tin nhắn role=user có nội dung")
        return to_pairs(msgs[:last]), clean(msgs[last]["content"])

    def chat_stream(self, messages, **kw):
        hist, user = self.check_messages(messages)
        ids = prompt_ids(self.tok, hist, user, self.eos, self.L, reserve=32)
        with self.lock:
            yield from self._stream_text(ids, True, **kw)

    def complete_stream(self, prompt, **kw):
        ids = self.tok.encode(str(prompt)).ids
        with self.lock:
            yield from self._stream_text(ids, False, **kw)


# ======================= server HTTP =======================
PAGE = """<!doctype html><meta charset=utf-8><meta name=viewport content="width=device-width,initial-scale=1">
<title>Yui</title><style>body{font:16px system-ui;margin:0;display:flex;flex-direction:column;height:100dvh}
#log{flex:1;overflow:auto;padding:12px}.u,.a{margin:6px 0;padding:8px 12px;border-radius:12px;max-width:85%;white-space:pre-wrap}
.u{background:#dbeafe;margin-left:auto}.a{background:#f3f4f6}form{display:flex;gap:6px;padding:8px;border-top:1px solid #ddd}
input{flex:1;font:inherit;padding:8px}button{font:inherit;padding:8px 14px}</style>
<div id=log></div><form id=f><input id=m placeholder="Nhắn cho Yui..." autocomplete=off><button>Gửi</button></form>
<script>
const log=document.getElementById('log'),msgs=[];
function add(c,t){const d=document.createElement('div');d.className=c;d.textContent=t;log.appendChild(d);log.scrollTop=1e9;return d}
document.getElementById('f').onsubmit=async e=>{e.preventDefault();const i=document.getElementById('m'),t=i.value.trim();if(!t)return;i.value='';
msgs.push({role:'user',content:t});add('u',t);const el=add('a','…');let out='';
const key=localStorage.getItem('key')||'';const h={'Content-Type':'application/json'};if(key)h.Authorization='Bearer '+key;
const r=await fetch('/api/chat',{method:'POST',headers:h,body:JSON.stringify({messages:msgs.slice(-8),stream:true})});
if(r.status==401){const k=prompt('API key:');if(k)localStorage.setItem('key',k);el.textContent='(nhập key rồi gửi lại)';msgs.pop();return}
if(!r.ok){el.textContent='Lỗi: '+(await r.text());msgs.pop();return}
const rd=r.body.getReader(),dec=new TextDecoder();let buf='';
for(;;){const{done,value}=await rd.read();if(done)break;buf+=dec.decode(value,{stream:true});const ls=buf.split('\\n');buf=ls.pop();
for(const l of ls){if(!l.trim())continue;const j=JSON.parse(l);if(j.message){out+=j.message.content;el.textContent=out||'…'}}}
msgs.push({role:'assistant',content:out})};
</script>"""


def parse_opts(body):
    o = body.get("options") or {}

    def g(names, default):
        for n in names:
            if n in o and o[n] is not None:
                return o[n]
            if n in body and body[n] is not None:
                return body[n]
        return default

    mx = int(g(("num_predict", "max_tokens"), 120))
    return {"max_new": 256 if mx <= 0 else min(mx, 4096), "temp": float(g(("temperature",), 0.8)),
            "top_k": int(g(("top_k",), 40)), "top_p": float(g(("top_p",), 0.95)),
            "rep": float(g(("repeat_penalty",), 1.1)), "k": g(("k",), None)}


def make_handler(engine, api_key=None):
    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.0"      # đóng kết nối khi xong -> stream không cần chunked

        def log_message(self, fmt, *a):
            print(f"[server] {self.address_string()} {fmt % a}", flush=True)

        def _cors(self):
            self.send_header("Access-Control-Allow-Origin", "*")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")
            self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")

        def _json(self, code, obj):
            b = json.dumps(obj, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(b)))
            self._cors()
            self.end_headers()
            self.wfile.write(b)

        def _start(self, ctype):
            self.send_response(200)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-cache")
            self._cors()
            self.end_headers()

        def _write(self, s):
            self.wfile.write(s.encode("utf-8"))
            self.wfile.flush()

        def _auth(self):
            if api_key and self.headers.get("Authorization", "") != f"Bearer {api_key}":
                self._json(401, {"error": "unauthorized"})
                return False
            return True

        def do_OPTIONS(self):
            self.send_response(204)
            self._cors()
            self.end_headers()

        def do_GET(self):
            path = self.path.split("?")[0]
            if path == "/":
                b = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(b)))
                self.end_headers()
                self.wfile.write(b)
            elif path == "/health":
                self._json(200, {"status": "ok", "model": engine.name})
            elif path in ("/api/tags", "/v1/models"):
                if self._auth():
                    self._json(200, {"models": [{"name": engine.name, "model": engine.name}],
                                     "object": "list", "data": [{"id": engine.name, "object": "model"}]})
            else:
                self._json(404, {"error": "not found"})

        def do_POST(self):
            if not self._auth():
                return
            try:
                n = int(self.headers.get("Content-Length") or 0)
                body = json.loads(self.rfile.read(n).decode("utf-8") or "{}")
                if not isinstance(body, dict):
                    raise ValueError("body phải là JSON object")
                o = parse_opts(body)
            except (ValueError, TypeError) as e:
                return self._json(400, {"error": f"yêu cầu không hợp lệ: {e}"})
            path = self.path.split("?")[0]
            now = lambda: time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())  # noqa: E731
            try:
                if path == "/api/chat":
                    engine.check_messages(body.get("messages"))
                    self._ollama(engine.chat_stream(body.get("messages"), **o), body.get("stream", True), now, chat=True)
                elif path == "/api/generate":
                    self._ollama(engine.complete_stream(body.get("prompt", ""), **o), body.get("stream", True), now, chat=False)
                elif path == "/v1/chat/completions":
                    engine.check_messages(body.get("messages"))
                    self._openai(engine.chat_stream(body.get("messages"), **o), body.get("stream", False))
                else:
                    self._json(404, {"error": "not found"})
            except ValueError as e:
                self._json(400, {"error": str(e)})
            except (BrokenPipeError, ConnectionResetError):
                pass

        def _ollama(self, gen, stream, now, chat):
            t0, n = time.time(), 0

            def piece(d, done):
                p = {"model": engine.name, "created_at": now(), "done": done}
                if chat:
                    p["message"] = {"role": "assistant", "content": d}
                else:
                    p["response"] = d
                return p

            try:
                if stream:
                    self._start("application/x-ndjson")
                    for d in gen:
                        n += 1
                        self._write(json.dumps(piece(d, False), ensure_ascii=False) + "\n")
                    last = piece("", True)
                    last.update({"done_reason": "stop", "eval_count": n, "total_duration": int((time.time() - t0) * 1e9)})
                    self._write(json.dumps(last, ensure_ascii=False) + "\n")
                else:
                    text = "".join(gen)
                    last = piece(text, True)
                    last.update({"done_reason": "stop", "total_duration": int((time.time() - t0) * 1e9)})
                    self._json(200, last)
            finally:
                gen.close()

        def _openai(self, gen, stream):
            cid, created = f"chatcmpl-{int(time.time() * 1000)}", int(time.time())

            def chunk(delta, fin):
                return {"id": cid, "object": "chat.completion.chunk", "created": created, "model": engine.name,
                        "choices": [{"index": 0, "delta": delta, "finish_reason": fin}]}

            try:
                if stream:
                    self._start("text/event-stream")
                    first = True
                    for d in gen:
                        delta = {"content": d}
                        if first:
                            delta["role"], first = "assistant", False
                        self._write("data: " + json.dumps(chunk(delta, None), ensure_ascii=False) + "\n\n")
                    self._write("data: " + json.dumps(chunk({}, "stop"), ensure_ascii=False) + "\n\ndata: [DONE]\n\n")
                else:
                    text = "".join(gen)
                    self._json(200, {"id": cid, "object": "chat.completion", "created": created, "model": engine.name,
                                     "choices": [{"index": 0, "message": {"role": "assistant", "content": text},
                                                  "finish_reason": "stop"}]})
            finally:
                gen.close()

    return H


def serve(engine, host, port, api_key):
    srv = ThreadingHTTPServer((host, port), make_handler(engine, api_key))
    srv.daemon_threads = True
    print(f"[server] http://{host}:{port}  (mở trang chat ở '/'; API /api/chat, /api/generate, /v1/chat/completions)"
          f"{'  [có API key]' if api_key else '  [KHÔNG có API key - đừng mở ra Internet]'}", flush=True)
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass


def terminal_chat(engine, args):
    if not engine.is_sft:
        print("[engine] CHƯA có sft_best.pt: đang ở chế độ viết tiếp văn bản (không phải chat). Gõ chữ mở đầu.")
    print("Dòng trống hoặc Ctrl+C để thoát.")
    msgs = []
    opts = {"max_new": args.max_new, "temp": args.temp, "top_k": args.top_k, "top_p": args.top_p, "rep": args.rep}
    while True:
        try:
            text = input("\nBạn: ")
        except (EOFError, KeyboardInterrupt):
            break
        if not clean(text):
            break
        t0, n, out = time.time(), 0, ""
        print("Yui:", end=" ", flush=True)
        if engine.is_sft:
            msgs.append({"role": "user", "content": text})
            gen = engine.chat_stream(msgs[-2 * args.max_turns - 1:], **opts)
        else:
            gen = engine.complete_stream(text, **opts)
        for d in gen:
            n += 1
            out += d
            print(d, end="", flush=True)
        print(f"\n  [{n} đoạn, {n / max(time.time() - t0, 1e-6):.1f}/s]")
        if engine.is_sft:
            msgs.append({"role": "assistant", "content": out})


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="thư mục có config.json, tokenizer.json, sft_best.pt/best.pt")
    ap.add_argument("--hf_repo", default=None)
    ap.add_argument("--hf_type", default="dataset")
    ap.add_argument("--ckpt", default="auto")
    ap.add_argument("--device", default=None)
    ap.add_argument("--k", type=int, default=2, choices=[1, 2], help="số expert dùng: 1 nhanh hơn, 2 chất lượng hơn")
    ap.add_argument("--serve", action="store_true")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=11434)
    ap.add_argument("--api_key", default=os.environ.get("ENGINE_API_KEY"))
    ap.add_argument("--max_new", type=int, default=120)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--rep", type=float, default=1.1)
    ap.add_argument("--max_turns", type=int, default=4)
    args = ap.parse_args(argv)
    eng = Engine(args.out, args.ckpt, args.device, args.k, args.hf_repo, args.hf_type)
    if args.serve:
        serve(eng, args.host, args.port, args.api_key)
    else:
        terminal_chat(eng, args)


if __name__ == "__main__":
    main()
