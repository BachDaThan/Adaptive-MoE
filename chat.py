"""
chat.py - dùng model đã train: gõ chữ mở đầu, model viết tiếp. Chạy được trên PC (CPU hoặc GPU) và, qua web, trên điện thoại.
Lưu ý: model hiện chỉ học Wikipedia nên nó VIẾT TIẾP văn bản chứ chưa biết trả lời câu hỏi; muốn nó trò chuyện thật cần
fine-tune trên dữ liệu hội thoại.

Cần: pip install torch tokenizers  (và gradio nếu dùng --web), cùng thư mục có adaptive_moe.py, elastic_ctx.py.
Thư mục --out cần config.json, tokenizer.json, best.pt (train_lm.py tạo ra; hoặc dùng --hf_repo để tự tải).

    python chat.py --out duong_dan/adaptive_moe                 # gõ trong terminal, k=2
    python chat.py --out duong_dan/adaptive_moe --k 1           # nhanh hơn
    python chat.py --out duong_dan/adaptive_moe --web           # mở web ở http://127.0.0.1:7860
    python chat.py --out duong_dan/adaptive_moe --web --share   # có link công khai, mở được từ điện thoại (Colab cũng dùng được)
"""
import argparse, json, os
import torch


def load(args):
    from tokenizers import Tokenizer
    from elastic_ctx import CtxLM
    out = args.out
    if args.hf_repo:
        from huggingface_hub import hf_hub_download
        for fn in ("config.json", "tokenizer.json", "best.pt"):
            if not os.path.exists(os.path.join(out, fn)):
                hf_hub_download(repo_id=args.hf_repo, filename=fn, repo_type=args.hf_type, local_dir=out,
                                token=os.environ.get("HF_TOKEN"))
    cfg = json.load(open(os.path.join(out, "config.json"), encoding="utf-8"))
    tok = Tokenizer.from_file(os.path.join(out, "tokenizer.json"))
    dev = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CtxLM(vocab=cfg["vocab"], d=cfg["d"], n_layers=cfg["layers"], n_experts=cfg["experts"], d_ff=2 * cfg["d"],
                  min_k=1, max_k=8, top_p=cfg["top_p"], n_shared=1, compute_coef=0.0).to(dev)
    ck = torch.load(os.path.join(out, "best.pt"), map_location=dev)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, tok, cfg, dev


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
    ap.add_argument("--web", action="store_true")
    ap.add_argument("--share", action="store_true")
    args = ap.parse_args(argv)
    if args.threads:
        torch.set_num_threads(args.threads)
    model, tok, cfg, dev = load(args)
    print(f"[chat] đã nạp model ({sum(p.numel() for p in model.parameters()) / 1e6:.0f}M tham số) trên {dev}", flush=True)
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
