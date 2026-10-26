"""
chat.py - dùng model đã train. Chạy được trên PC (CPU hoặc GPU) và, qua web, trên điện thoại.
Hai chế độ (tự chọn theo checkpoint):
  - có sft_best.pt (do train_sft.py tạo): CHAT hội thoại "User: ...\nYui: ...", nhớ vài lượt gần nhất;
  - chỉ có best.pt (pretrain): gõ chữ mở đầu, model viết tiếp (thêm --text để ép chế độ này).

Cần: pip install torch tokenizers  (và gradio nếu dùng --web), cùng thư mục có adaptive_moe.py, elastic_ctx.py.
Thư mục --out cần config.json, tokenizer.json, best.pt và/hoặc sft_best.pt (hoặc dùng --hf_repo để tự tải; repo private cần HF_TOKEN).

    python chat.py --out duong_dan/adaptive_moe                 # gõ trong terminal, k=2
    python chat.py --out duong_dan/adaptive_moe --k 1           # nhanh hơn
    python chat.py --out duong_dan/adaptive_moe --web           # mở web ở http://127.0.0.1:7860
    python chat.py --out duong_dan/adaptive_moe --web --share   # có link công khai, mở được từ điện thoại (Colab cũng dùng được)
"""
import argparse, json, os
import torch
from sft_format import USER_TAG, clean, prompt_ids


def load(args):
    from tokenizers import Tokenizer
    from elastic_ctx import CtxLM
    out = args.out
    if args.hf_repo:
        from huggingface_hub import hf_hub_download
        for fn in ("config.json", "tokenizer.json", "sft_best.pt", "best.pt"):
            if not os.path.exists(os.path.join(out, fn)):
                try:
                    hf_hub_download(repo_id=args.hf_repo, filename=fn, repo_type=args.hf_type, local_dir=out,
                                    token=os.environ.get("HF_TOKEN"))
                except Exception as e:  # noqa: BLE001
                    if fn in ("config.json", "tokenizer.json"):
                        raise
                    print(f"[chat] không tải được {fn} ({str(e)[:60]})", flush=True)
    cfg = json.load(open(os.path.join(out, "config.json"), encoding="utf-8"))
    tok = Tokenizer.from_file(os.path.join(out, "tokenizer.json"))
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CtxLM(vocab=cfg["vocab"], d=cfg["d"], n_layers=cfg["layers"], n_experts=cfg["experts"], d_ff=2 * cfg["d"],
                  min_k=1, max_k=8, top_p=cfg["top_p"], n_shared=1, compute_coef=0.0).to(dev)
    ckpt = args.ckpt
    if ckpt == "auto":
        ckpt = "sft_best.pt" if os.path.exists(os.path.join(out, "sft_best.pt")) else "best.pt"
    ck = torch.load(os.path.join(out, ckpt), map_location=dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"[chat] dùng {ckpt}", flush=True)
    return model, tok, cfg, dev, ckpt.startswith("sft")


@torch.no_grad()
def stream(model, tok, cfg, dev, prompt, k=2, max_new=120, temp=0.8, top_k=40, rep=1.15):
    """Sinh từng token; mỗi lần trả về toàn bộ văn bản tính đến hiện tại."""
    for b in model.blocks:
        b.moe.k_cap = int(k)
    ids = tok.encode(prompt).ids
    eos = tok.token_to_id("<eos>")
    L = cfg["L"]
    for _ in range(int(max_new)):
        x = torch.tensor([ids[-L:]], dtype=torch.long, device=dev)
        lg, _ = model(x, None)
        logits = lg[0, -1].float()
        if rep != 1.0:
            for t in set(ids[-64:]):
                logits[t] = logits[t] / rep if logits[t] > 0 else logits[t] * rep
        logits = logits / max(temp, 1e-5)
        if top_k > 0:
            v, _ = logits.topk(int(top_k))
            logits[logits < v[-1]] = -float("inf")
        nxt = int(torch.multinomial(logits.softmax(-1), 1))
        if nxt == eos:
            break
        ids.append(nxt)
        yield tok.decode(ids)


@torch.no_grad()
def stream_chat(model, tok, cfg, dev, history, user_msg, k=2, max_new=80, temp=0.8, top_k=40, rep=1.1):
    """Chat theo định dạng SFT. history = [(user, yui), ...]. Mỗi lần trả về câu trả lời của Yui tính đến hiện tại."""
    for b in model.blocks:
        b.moe.k_cap = int(k)
    eos = tok.token_to_id("<eos>")
    L = cfg["L"]
    ids = prompt_ids(tok, history, user_msg, eos, L)
    gen = []
    for _ in range(int(max_new)):
        x = torch.tensor([ids[-L:]], dtype=torch.long, device=dev)
        lg, _ = model(x, None)
        logits = lg[0, -1].float()
        if rep != 1.0:
            for t in set(gen[-48:]):
                logits[t] = logits[t] / rep if logits[t] > 0 else logits[t] * rep
        logits = logits / max(temp, 1e-5)
        if top_k > 0:
            v, _ = logits.topk(int(top_k))
            logits[logits < v[-1]] = -float("inf")
        nxt = int(torch.multinomial(logits.softmax(-1), 1))
        if nxt == eos:
            break
        ids.append(nxt); gen.append(nxt)
        text = tok.decode(gen).strip()
        cut = text.find(USER_TAG)
        if cut >= 0:   # model quên <eos> và tự viết tiếp lượt của User
            yield text[:cut].strip()
            return
        yield text


def _norm_history(h):
    """Gradio có thể đưa lịch sử dạng [(user, bot)] hoặc [{'role','content'}]; đổi về [(user, bot)]."""
    pairs, cur = [], None
    for it in h or []:
        if isinstance(it, dict):
            c = it.get("content")
            if isinstance(c, list):
                c = " ".join(str(x.get("text", "")) if isinstance(x, dict) else str(x) for x in c)
            if it.get("role") == "user":
                cur = c
            elif it.get("role") == "assistant" and cur is not None:
                pairs.append((cur, c)); cur = None
        elif isinstance(it, (list, tuple)) and len(it) == 2 and it[0] is not None and it[1] is not None:
            pairs.append((str(it[0]), str(it[1])))
    return pairs


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="model")
    ap.add_argument("--hf_repo", default=None)
    ap.add_argument("--hf_type", default="dataset")
    ap.add_argument("--device", default=None)
    ap.add_argument("--threads", type=int, default=0, help="số luồng CPU (0 = mặc định)")
    ap.add_argument("--k", type=int, default=2)
    ap.add_argument("--max_new", type=int, default=120)
    ap.add_argument("--temp", type=float, default=0.8)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--rep", type=float, default=1.15)
    ap.add_argument("--ckpt", default="auto", help="auto = sft_best.pt nếu có, không thì best.pt; hoặc tên file cụ thể")
    ap.add_argument("--text", action="store_true", help="ép chế độ viết tiếp văn bản (bỏ chế độ chat)")
    ap.add_argument("--max_turns", type=int, default=4, help="số lượt gần nhất nhớ trong chế độ chat")
    ap.add_argument("--web", action="store_true")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    model, tok, cfg, dev, is_sft = load(args)
    chat_mode = is_sft and not args.text
    print(f"[chat] đã nạp model ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M tham số) trên {dev}", flush=True)
    if chat_mode and args.web:
        import gradio as gr

        def cfn(message, history, k, max_new, temp, top_k, rep):
            hist = _norm_history(history)[-args.max_turns:]
            last = ""
            for last in stream_chat(model, tok, cfg, dev, hist, message, k, max_new, temp, top_k, rep):
                yield last
            yield last

        gr.ChatInterface(
            cfn,
            additional_inputs=[gr.Radio([1, 2], value=args.k, label="k (1 nhanh hơn)"),
                               gr.Slider(10, 150, value=min(args.max_new, 80), step=10, label="Số token tối đa"),
                               gr.Slider(0.1, 1.5, value=args.temp, step=0.05, label="Nhiệt độ"),
                               gr.Slider(0, 100, value=args.top_k, step=1, label="top-k"),
                               gr.Slider(1.0, 1.5, value=1.1, step=0.01, label="Phạt lặp")],
            title="Yui (Adaptive-MoE)",
        ).queue().launch(share=args.share, server_name="0.0.0.0")
        return
    if chat_mode:
        print("Chat với Yui. Dòng trống hoặc Ctrl+C để thoát.")
        hist = []
        while True:
            try:
                msg = input("\nBạn: ")
            except (EOFError, KeyboardInterrupt):
                break
            if not clean(msg):
                break
            print("Yui:", end=" ", flush=True)
            shown, reply = "", ""
            for reply in stream_chat(model, tok, cfg, dev, hist, msg, args.k, min(args.max_new, 80), args.temp, args.top_k, 1.1):
                if reply.startswith(shown):
                    print(reply[len(shown):], end="", flush=True)
                    shown = reply
            if reply.startswith(shown):
                print(reply[len(shown):], end="")
            print()
            hist = (hist + [(clean(msg), reply)])[-args.max_turns:]
        return
    if args.web:
        import gradio as gr

        def fn(prompt, k, max_new, temp, top_k, rep):
            last = prompt
            for last in stream(model, tok, cfg, dev, prompt, k, max_new, temp, top_k, rep):
                yield last
            yield last

        gr.Interface(
            fn,
            [gr.Textbox(lines=4, label="Văn bản mở đầu", value="Việt Nam là"),
             gr.Radio([1, 2], value=args.k, label="k (số expert mỗi token; 1 nhanh hơn)"),
             gr.Slider(10, 300, value=args.max_new, step=10, label="Số token viết thêm"),
             gr.Slider(0.1, 1.5, value=args.temp, step=0.05, label="Nhiệt độ"),
             gr.Slider(0, 100, value=args.top_k, step=1, label="top-k"),
             gr.Slider(1.0, 1.5, value=args.rep, step=0.01, label="Phạt lặp")],
            gr.Textbox(lines=12, label="Kết quả"),
            title="Adaptive-MoE (model tự train)", flagging_mode="never",
        ).queue().launch(share=args.share, server_name="0.0.0.0")
        return
    print("Gõ chữ mở đầu rồi Enter để model viết tiếp. Dòng trống hoặc Ctrl+C để thoát.")
    while True:
        try:
            prompt = input("\n>>> ")
        except (EOFError, KeyboardInterrupt):
            break
        if not prompt.strip():
            break
        shown = prompt
        print(prompt, end="", flush=True)
        for text in stream(model, tok, cfg, dev, prompt, args.k, args.max_new, args.temp, args.top_k, args.rep):
            print(text[len(shown):], end="", flush=True)
            shown = text
        print()


if __name__ == "__main__":
    main()
