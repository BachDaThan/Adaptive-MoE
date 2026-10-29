"""model_card.py - sinh README.md RIÊNG cho từng model (chỉ ghi những gì có thật trong config/trạng thái train, không bịa số liệu).

Dùng ở: train_lm.py (sau pretrain / khi tạm dừng), train_sft.py (sau fine-tune), export_yui.py (cạnh file .yui).
README nằm cùng thư mục với model: models/<tên>/README.md trên HF.
"""
import datetime
import os
import re

HEAD = "---\nlanguage:\n- vi\ntags:\n- yui\n- adaptive-moe\n---\n\n"


def _now():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _n(x):
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    for div, suf in ((1e9, "B"), (1e6, "M"), (1e3, "K")):
        if x >= div:
            return f"{x / div:.2f}{suf}".replace(".00", "")
    return str(int(x))


def _arch(cfg):
    rows = [("Số tham số", _n(cfg["params"]) if cfg.get("params") else None), ("Chiều ẩn (d)", cfg.get("d")), ("Số layer", cfg.get("layers")),
            ("Số expert", cfg.get("experts")), ("Số head", cfg.get("heads")), ("Vocab", cfg.get("vocab")), ("Độ dài ngữ cảnh (L)", cfg.get("L")),
            ("top_p (router)", cfg.get("top_p")), ("k_max", cfg.get("k_max")), ("Ngôn ngữ/nguồn train", ", ".join(cfg["langs"]) if cfg.get("langs") else None)]
    return "\n".join(f"| {k} | {v} |" for k, v in rows if v not in (None, ""))


def _table(cfg):
    return "| Thông số | Giá trị |\n|---|---|\n" + _arch(cfg) + "\n"


def write(folder, text, name="README.md"):
    os.makedirs(folder, exist_ok=True)
    with open(os.path.join(folder, name), "w", encoding="utf-8") as f:
        f.write(text)


def pretrain_card(cfg, step, steps, best, data_repo=None, tag=None, name=None):
    name = name or cfg.get("name") or "model"
    done = step >= steps
    val = f"{best[0]:.4f} (bước {best[1]})" if best and best[0] < 1e8 else "chưa có"
    return (HEAD + f"# {name}\n\nModel ngôn ngữ tiếng Việt (Adaptive-MoE), bản **pretrain**.\n\n"
            f"- Trạng thái: {'đã train xong' if done else 'đang train dở (chạy lại cùng lệnh để tiếp tục)'} — bước {step}/{steps}\n"
            f"- Val loss tốt nhất: {val}\n- Cập nhật: {_now()}\n"
            + (f"- Data: repo `{data_repo}`" + (f", thư mục `data/{tag}/`" if tag else "") + "\n" if data_repo else "")
            + "\n## Kiến trúc\n\n" + _table(cfg)
            + "\n## Tệp trong thư mục này\n\n- `best.pt`: checkpoint val loss tốt nhất (dùng để suy luận)\n- `last.pt`: checkpoint mới nhất (kèm trạng thái optimizer, để train tiếp)\n"
            + "- `config.json`, `tokenizer.json`: cấu hình và tokenizer\n\n"
            "Lưu ý: đây là model pretrain, chưa fine-tune hội thoại. Bản fine-tune (nếu có) nằm ở thư mục `<tên>-sft` cạnh thư mục này.\n")


def sft_card(name, base, cfg, info):
    files = ", ".join(info.get("files") or []) or "không rõ"
    b = info.get("before_val")
    return (HEAD + f"# {name}\n\nModel hội thoại tiếng Việt (Adaptive-MoE), bản **fine-tune (SFT)**.\n\n"
            f"- Model gốc: `{base}`" + (f" (pretrain ở bước {info['base_step']})" if info.get("base_step") is not None else "") + "\n"
            f"- Số mẫu train/val: {info.get('n_train')}/{info.get('n_val')}\n"
            f"- Val loss tốt nhất: {info.get('best_val'):.4f} ở epoch {info.get('best_epoch')}"
            + (f" (trước fine-tune: {b:.4f})" if isinstance(b, float) else "") + "\n"
            f"- File data: {files}\n- Cập nhật: {_now()}\n\n## Kiến trúc\n\n" + _table(cfg)
            + "\n## Tệp trong thư mục này\n\n- `sft_best.pt`: trọng số fine-tune tốt nhất (dùng để suy luận / export `.yui`)\n"
            "- `config.json`, `tokenizer.json`: cấu hình và tokenizer\n- `sft_state.json`: tiến độ fine-tune\n\n"
            "Định dạng hội thoại: `User: ...` / `Yui: ...` kết thúc bằng `<eos>`.\n")


def yui_card(folder, base, cfg, src_dir, ckpt):
    rows = []
    for fn in sorted(os.listdir(folder)):
        m = re.fullmatch(re.escape(base) + r"_(f32|int8|q8|q4all|q4)\.yui", fn)
        if m:
            rows.append(f"| `{fn}` | {m.group(1)} | {os.path.getsize(os.path.join(folder, fn)) / 1e6:.1f} MB |")
    return (HEAD + f"# {cfg.get('name') or base} (.yui)\n\nBản xuất cho engine C++ (`yui`) từ `{src_dir}` (checkpoint `{ckpt}`).\n\n"
            "| File | Lượng tử hoá | Dung lượng |\n|---|---|---|\n" + "\n".join(rows) + f"\n\nCập nhật: {_now()}\n\n## Kiến trúc\n\n" + _table(cfg)
            + "\nKiểm chất lượng bản q4 trước khi dùng: `python verify_cpp.py --out ... --model <file q4>` rồi thử chat vài câu.\n")
