"""
vision_yui.py - xem ảnh cho Yui: bộ mã hoá ảnh (đóng băng) -> gộp token -> projector -> chèn vào chuỗi embedding của LM.

  ảnh --SigLIP-base (224x224, 14x14 = 196 mảnh)--> gộp avg_pool(pool x pool) --> n_img = (14/pool)^2 token ảnh
      --Projector (LayerNorm -> Linear -> GELU -> Linear)--> [n_img, D]  (D = chiều embedding của LM)

Khuôn prompt (PHẢI trùng yui.cpp::build_prompt khi có ảnh):
      "User: " + [ảnh x n_img] + (" " + câu hỏi nếu có) + "\\nYui:"   rồi câu trả lời " ..." + <eos>
Đoạn chữ được mã hoá TỪNG KHÚC (không gộp) đúng như engine, nên id token khớp tuyệt đối.
Cần: pip install torch transformers pillow
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

SIGLIP = "google/siglip-base-patch16-224"
USER_TAG, BOT_TAG = "User:", "Yui:"
GRID = 14                       # 224 / 16
MEAN = STD = 0.5                # SigLIP chuẩn hoá về [-1, 1]


def n_tokens(pool):
    if GRID % pool:
        raise ValueError(f"pool phải chia hết {GRID} (1, 2, 7, 14)")
    return (GRID // pool) ** 2


class VisionTower(nn.Module):
    """SigLIP vision tower + gộp mảnh. Đóng băng. forward(pixels [B,3,224,224]) -> [B, n_img, 768]."""

    def __init__(self, pool=2, name=SIGLIP):
        super().__init__()
        from transformers import SiglipVisionModel
        self.vit = SiglipVisionModel.from_pretrained(name)
        self.pool = pool
        self.out_dim = self.vit.config.hidden_size
        for p in self.vit.parameters():
            p.requires_grad_(False)

    def forward(self, pixels):
        h = self.vit(pixel_values=pixels).last_hidden_state          # [B, 196, C]
        B, T, C = h.shape
        g = int(T ** 0.5)
        h = h.transpose(1, 2).reshape(B, C, g, g)
        if self.pool > 1:
            h = F.avg_pool2d(h, self.pool)
        return h.flatten(2).transpose(1, 2)                          # [B, n_img, C]


class Projector(nn.Module):
    def __init__(self, in_dim, d_model, hidden=None):
        super().__init__()
        hidden = hidden or 2 * d_model
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, hidden), nn.GELU(), nn.Linear(hidden, d_model))

    def forward(self, x):
        return self.net(x)


def preprocess(img, size=224):
    """PIL.Image -> tensor [3,size,size] trong [-1,1] (resize thẳng, giống SigLIP)."""
    import numpy as np
    img = img.convert("RGB").resize((size, size))
    a = torch.from_numpy(np.asarray(img, dtype="float32") / 255.0).permute(2, 0, 1)
    return (a - MEAN) / STD


def text_ids(tok, question, eos, answer=None):
    """-> (prefix_a, prefix_b, answer_ids). prefix_a trước ảnh, prefix_b sau ảnh (đúng khuôn engine)."""
    a = tok.encode(f"{USER_TAG} ").ids
    q = (question or "").strip()
    b = tok.encode((" " + q if q else "") + f"\n{BOT_TAG}").ids
    ans = (tok.encode(" " + answer.strip()).ids + [eos]) if answer is not None else []
    return a, b, ans


def build_batch(items, n_img, eos, device, max_len):
    """items: list[(a, b, ans)] -> ids [B,T], is_img [B,T] (bool), loss_mask [B,T]. Vị trí ảnh dùng id 0 + cờ is_img."""
    seqs = []
    for a, b, ans in items:
        ids = a + [0] * n_img + b + ans
        img = [False] * len(a) + [True] * n_img + [False] * (len(b) + len(ans))
        msk = [0] * (len(a) + n_img + len(b)) + [1] * len(ans)
        seqs.append((ids[:max_len], img[:max_len], msk[:max_len]))
    T = max(len(s[0]) for s in seqs)
    ids = torch.full((len(seqs), T), eos, dtype=torch.long)
    img = torch.zeros((len(seqs), T), dtype=torch.bool)
    msk = torch.zeros((len(seqs), T), dtype=torch.float)
    for r, (i, g, m) in enumerate(seqs):
        ids[r, :len(i)] = torch.tensor(i)
        img[r, :len(g)] = torch.tensor(g)
        msk[r, :len(m)] = torch.tensor(m, dtype=torch.float)
    return ids.to(device), img.to(device), msk.to(device)


def lm_with_images(model, ids, is_img, img_emb):
    """Chạy LM với embedding ảnh chèn vào các vị trí is_img. img_emb [B, n_img, D]. Trả logits [B,T,V]."""
    x = model.emb(ids)
    x = x.clone()
    x[is_img] = img_emb.reshape(-1, img_emb.shape[-1]).to(x.dtype)
    logits, _ = model(ids, None, embeds=x)
    return logits
