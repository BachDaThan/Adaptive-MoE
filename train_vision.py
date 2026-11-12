"""
train_vision.py - dạy model Yui XEM ẢNH (không pretrain lại phần chữ).

Hai giai đoạn (chạy lần lượt, đều trên Kaggle/Colab GPU):
  Giai đoạn 1  (--stage 1)  Đóng băng cả SigLIP lẫn LM, CHỈ train Projector. Dữ liệu: mô tả ảnh tiếng Việt (q có thể để trống). Nhanh, rẻ.
  Giai đoạn 2  (--stage 2)  Nạp projector của giai đoạn 1, mở khoá LM (lr nhỏ) + projector, train trên hỏi-đáp ảnh. Lưu vision_best.pt
                            cùng khuôn với sft_best.pt nên export_yui.py dùng được: python export_yui.py --out <dir> --ckpt vision_best.pt --quant t2 ...
Đầu ra trong --out:  vision_proj.pt (projector + cấu hình)  |  vision_best.pt (chỉ giai đoạn 2: trọng số LM đã học ảnh)

Dữ liệu (chọn một):
  --data file.jsonl [--images_dir thư_mục]   mỗi dòng {"image": "đường/dẫn.jpg", "q": "câu hỏi (có thể rỗng)", "a": "câu trả lời"}
  --hf_dataset tên_dataset [--hf_split train] [--image_col image --q_col question --a_col answer]
        dataset kiểu LLaVA (cột "conversations": [{"from":"human","value":"<image>\\n..."},{"from":"gpt","value":"..."}]) cũng đọc được.
        Tự kiểm tra giấy phép dữ liệu trước khi dùng.
Chẩn đoán quan trọng: in "val(ảnh đúng)" và "val(ảnh xáo)". Nếu hai số gần bằng nhau thì model đang BỎ QUA ảnh (chỉ đoán theo chữ).

Cần: pip install torch transformers pillow datasets tokenizers; cạnh file: adaptive_moe.py elastic_ctx.py langcheck.py train_lm.py vision_yui.py
Ngữ cảnh: n_img = (14/pool)^2 token ảnh (pool=2 -> 49). Cần --L của model >= n_img + ~100 (khuyên 512).
"""
import argparse, json, math, os, random, time
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from train_lm import build_model, set_cap, atomic_save
import vision_yui as V


def extract_qa(row, a):
    """-> (question, answer) hoặc None."""
    if a.q_col in row or a.a_col in row:
        q, ans = row.get(a.q_col) or "", row.get(a.a_col)
        if isinstance(ans, list): ans = ans[0] if ans else None
        return (str(q), str(ans)) if ans else None
    conv = row.get("conversations") or row.get("messages")
    if conv:
        human = next((c for c in conv if c.get("from", c.get("role")) in ("human", "user")), None)
        bot = next((c for c in conv if c.get("from", c.get("role")) in ("gpt", "assistant")), None)
        if human and bot:
            q = (human.get("value", human.get("content")) or "").replace("<image>", "").strip()
            return q, (bot.get("value", bot.get("content")) or "").strip()
    return None


class Pairs(Dataset):
    def __init__(self, a, split_ids):
        self.a, self.idx = a, split_ids
        self.rows = None
        if a.hf_dataset:
            from datasets import load_dataset
            self.rows = load_dataset(a.hf_dataset, split=a.hf_split)
        else:
            self.rows = [json.loads(l) for l in open(a.data, encoding="utf-8") if l.strip()]

    def __len__(self): return len(self.idx)

    def __getitem__(self, i):
        from PIL import Image
        row = self.rows[self.idx[i]]
        qa = extract_qa(row, self.a)
        try:
            if self.a.hf_dataset:
                img = row[self.a.image_col]
            else:
                img = Image.open(os.path.join(self.a.images_dir or "", row["image"]))
            return V.preprocess(img), qa
        except Exception:  # noqa: BLE001 - ảnh hỏng thì bỏ qua mẫu này
            return None


def make_collate(tok, n_img, eos, L, max_a_chars):
    def collate(batch):
        batch = [b for b in batch if b is not None and b[1] is not None]
        if not batch: return None
        pix, items = [], []
        for p, (q, ans) in batch:
            a, b, an = V.text_ids(tok, q[:300], eos, ans[:max_a_chars])
            if len(a) + n_img + len(b) + 8 > L: continue          # ảnh + câu hỏi không vừa ngữ cảnh -> bỏ
            pix.append(p); items.append((a, b, an))
        if not items: return None
        return torch.stack(pix), items
    return collate


def lr_at(step, total, lr, warm):
    if step < warm: return lr * (step + 1) / warm
    return lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))


def step_loss(model, tower, proj, pixels, items, n_img, eos, L, device, amp, shuffle_img=False):
    ids, is_img, msk = V.build_batch(items, n_img, eos, device, L)
    with torch.no_grad():
        feats = tower(pixels.to(device).to(next(tower.parameters()).dtype)).float()
    if shuffle_img and feats.shape[0] > 1: feats = feats.roll(1, 0)
    with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
        emb = proj(feats)
        lg = V.lm_with_images(model, ids[:, :-1], is_img[:, :-1], emb)
    y, m = ids[:, 1:], msk[:, 1:]
    ce = F.cross_entropy(lg.float().reshape(-1, lg.shape[-1]), y.reshape(-1), reduction="none").view_as(y)
    return (ce * m).sum(), m.sum()


@torch.no_grad()
def evaluate(model, tower, proj, loader, n_img, eos, L, device, amp, shuffle):
    model.eval(); proj.eval(); tot = n = 0.0
    for b in loader:
        if b is None: continue
        a, c = step_loss(model, tower, proj, b[0], b[1], n_img, eos, L, device, amp, shuffle)
        tot += a.item(); n += c.item()
    return tot / max(n, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="thư mục có config.json, tokenizer.json, checkpoint của model chữ")
    ap.add_argument("--ckpt", default="auto", help="auto = vision_best.pt (nếu stage 2 làm tiếp) > sft_best.pt > best.pt")
    ap.add_argument("--stage", type=int, choices=[1, 2], default=1)
    ap.add_argument("--pool", type=int, default=2, help="gộp mảnh pool x pool: 2 -> 49 token ảnh, 1 -> 196, 7 -> 4")
    ap.add_argument("--data", default=None); ap.add_argument("--images_dir", default="")
    ap.add_argument("--hf_dataset", default=None); ap.add_argument("--hf_split", default="train")
    ap.add_argument("--image_col", default="image"); ap.add_argument("--q_col", default="question"); ap.add_argument("--a_col", default="answer")
    ap.add_argument("--max_samples", type=int, default=0); ap.add_argument("--max_a_chars", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=2); ap.add_argument("--B", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-3, help="projector"); ap.add_argument("--lr_lm", type=float, default=2e-5, help="LM (chỉ stage 2)")
    ap.add_argument("--warmup", type=int, default=100); ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--val_frac", type=float, default=0.02); ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no_amp", action="store_true")
    ap.add_argument("--push_repo", default=None, help="đẩy vision_proj.pt/vision_best.pt lên repo HF này (cần HF_TOKEN)")
    a = ap.parse_args()
    if not a.data and not a.hf_dataset: raise SystemExit("cần --data file.jsonl hoặc --hf_dataset")

    random.seed(a.seed); torch.manual_seed(a.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = (not a.no_amp) and device == "cuda"
    cfg = json.load(open(os.path.join(a.out, "config.json"), encoding="utf-8"))
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(os.path.join(a.out, "tokenizer.json")); eos = tok.token_to_id("<eos>")
    L, n_img = cfg["L"], V.n_tokens(a.pool)
    if L < n_img + 100: raise SystemExit(f"model có L={L} quá nhỏ cho {n_img} token ảnh (cần >= {n_img + 100}); đặt --pool lớn hơn hoặc pretrain với --L 512")

    ck_name = a.ckpt
    if ck_name == "auto":
        ck_name = next((n for n in ("vision_best.pt", "sft_best.pt", "best.pt") if os.path.exists(os.path.join(a.out, n)) and (n != "vision_best.pt" or a.stage == 2)), None)
        if ck_name is None: raise SystemExit("không thấy checkpoint trong --out")
    ck = torch.load(os.path.join(a.out, ck_name), map_location=device)
    model = build_model(cfg, device); model.load_state_dict(ck["model"]); del ck
    set_cap(model, 2)
    print(f"[vision] LM từ {ck_name}; L={L}; {n_img} token ảnh; stage {a.stage}", flush=True)

    tower = V.VisionTower(a.pool).to(device)
    if device == "cuda": tower.half()
    tower.eval()
    proj = V.Projector(tower.out_dim, cfg["d"]).to(device)
    pp = os.path.join(a.out, "vision_proj.pt")
    if a.stage == 2:
        if not os.path.exists(pp): raise SystemExit("stage 2 cần vision_proj.pt của stage 1 trong --out")
        proj.load_state_dict(torch.load(pp, map_location=device)["proj"])
    for p in model.parameters(): p.requires_grad_(a.stage == 2)

    total_rows = (len(__import__("datasets").load_dataset(a.hf_dataset, split=a.hf_split)) if a.hf_dataset
                  else sum(1 for l in open(a.data, encoding="utf-8") if l.strip()))
    idx = list(range(total_rows)); random.shuffle(idx)
    if a.max_samples: idx = idx[:a.max_samples]
    nv = max(8, int(len(idx) * a.val_frac)); val_i, tr_i = idx[:nv], idx[nv:]
    col = make_collate(tok, n_img, eos, L, a.max_a_chars)
    tr = DataLoader(Pairs(a, tr_i), batch_size=a.B, shuffle=True, num_workers=a.workers, collate_fn=col, drop_last=True)
    va = DataLoader(Pairs(a, val_i), batch_size=a.B, num_workers=a.workers, collate_fn=col)
    print(f"[vision] train {len(tr_i)} / val {len(val_i)} mẫu", flush=True)

    groups = [{"params": proj.parameters(), "lr": a.lr}]
    if a.stage == 2: groups.append({"params": model.parameters(), "lr": a.lr_lm})
    opt = torch.optim.AdamW(groups, weight_decay=0.01)
    base = [g["lr"] for g in opt.param_groups]
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    total = max(1, len(tr) * a.epochs); step = 0; best = float("inf"); t0 = time.time()

    def save(tag):
        meta = {"proj": proj.state_dict(), "pool": a.pool, "d_model": cfg["d"], "n_img": n_img, "tower": V.SIGLIP, "stage": a.stage}
        atomic_save(meta, pp)
        if a.stage == 2: atomic_save({"model": model.state_dict(), "step": step, "vision": True}, os.path.join(a.out, "vision_best.pt"))
        print(f"[vision] lưu ({tag})", flush=True)
        if a.push_repo:
            try:
                from huggingface_hub import HfApi
                api = HfApi()
                for fn in ["vision_proj.pt"] + (["vision_best.pt"] if a.stage == 2 else []):
                    api.upload_file(path_or_fileobj=os.path.join(a.out, fn), path_in_repo=f"vision/{fn}", repo_id=a.push_repo, repo_type="model")
            except Exception as e:  # noqa: BLE001
                print("[vision] đẩy HF lỗi:", e, flush=True)

    for ep in range(a.epochs):
        for b in tr:
            if b is None: continue
            model.train(a.stage == 2); proj.train()
            for g, bl in zip(opt.param_groups, base): g["lr"] = lr_at(step, total, bl, a.warmup)
            s, n = step_loss(model, tower, proj, b[0], b[1], n_img, eos, L, device, amp)
            loss = s / n.clamp(min=1)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward(); scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
            scaler.step(opt); scaler.update(); step += 1
            if step % 50 == 0:
                print(f"[vision] ep {ep} bước {step}/{total} loss {loss.item():.4f}  {time.time() - t0:.0f}s", flush=True)
        v_ok = evaluate(model, tower, proj, va, n_img, eos, L, device, amp, False)
        v_sh = evaluate(model, tower, proj, va, n_img, eos, L, device, amp, True)
        print(f"[vision] ep {ep}: val(ảnh đúng) {v_ok:.4f}   val(ảnh xáo) {v_sh:.4f}   chênh {v_sh - v_ok:+.4f}"
              f"{'   ⚠ chênh ~0: model đang bỏ qua ảnh' if v_sh - v_ok < 0.02 else ''}", flush=True)
        if v_ok < best: best = v_ok; save(f"ep {ep}, val {v_ok:.4f}")
    print(f"[vision] xong. val tốt nhất {best:.4f}. Tiếp: stage 2 (nếu vừa chạy stage 1), rồi export_vision.py + export_yui.py --ckpt vision_best.pt", flush=True)


if __name__ == "__main__":
    main()
