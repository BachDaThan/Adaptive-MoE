"""
adaptive_moe.py (v0.2) - MoE với số chuyên gia và "sức mạnh" thay đổi theo từng token.

Thành phần:
  AdaptiveRouter   : top-p routing + strength head + loss phụ (cân bằng tải, compute, z-loss).
  AdaptiveMoE      : layer MoE thích ứng (có shared expert tùy chọn, k_cap, LRU offload).
  set_k_cap        : đặt trần số expert cho cả model.
  gather_aux_loss  : cộng loss phụ của mọi layer (không cần tự gom từ forward).
  moe_stats        : thống kê avg_k / avg_strength theo từng layer.
  cascade_forward  : chạy tầng nhỏ trước, kém tự tin thì nâng tầng.
  adaptive_generate: sinh text, mỗi bước tự nâng tầng khi cần.

Chỉ phụ thuộc PyTorch.
"""
import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

__version__ = "0.3.0"


class Expert(nn.Module):
    """FFN kiểu SwiGLU."""

    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w_gate = nn.Linear(d_model, d_ff, bias=False)
        self.w_up = nn.Linear(d_model, d_ff, bias=False)
        self.w_down = nn.Linear(d_ff, d_model, bias=False)

    def forward(self, x):
        return self.w_down(F.silu(self.w_gate(x)) * self.w_up(x))


class AdaptiveRouter(nn.Module):
    """
    Top-p routing: lấy các expert có xác suất cao nhất cho tới khi tổng xác suất
    đạt `top_p`. Gate "nhọn" (token dễ) -> ít expert, gate phẳng (token khó) -> nhiều.

    target_k: nếu đặt, compute_loss kéo số expert trung bình về gần giá trị này
              (dễ kiểm soát hơn phạt thô). None = phạt số expert kỳ vọng.
    noise_std: nhiễu Gaussian lên logits khi train (giúp router khám phá).
    """

    def __init__(self, d_model, n_experts, min_k=1, max_k=None, top_p=0.7,
                 min_strength=0.3, tau=0.05, target_k=None, noise_std=0.0, ste=False,
                 difficulty=False, p_min=0.4, p_max=0.95, rand_k=None, renorm=True):
        super().__init__()
        self.renorm = renorm   # False: giữ nguyên xác suất gate (kiểu Switch), top-1 vẫn có gradient cho router
        self.ste = ste
        self.rand_k = rand_k   # (lo, hi, frac_hi): đối chứng, k ngẫu nhiên theo token
        self.difficulty = difficulty
        self.p_min = p_min
        self.p_max = p_max
        self.n_experts = n_experts
        self.min_k = min_k
        self.max_k = max_k or n_experts
        self.top_p = top_p
        self.min_strength = min_strength
        self.tau = tau
        self.target_k = target_k
        self.noise_std = noise_std
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.strength_head = nn.Linear(d_model, 1)
        if difficulty:
            self.diff_head = nn.Linear(d_model, 1)   # dự đoán độ khó của token (0..1)

    def forward(self, x, k_cap=None):
        E = self.n_experts
        logits = self.gate(x).float()
        if self.training and self.noise_std > 0:
            logits = logits + torch.randn_like(logits) * self.noise_std
        probs = F.softmax(logits, dim=-1)                            # [T, E]
        sorted_p, sorted_idx = probs.sort(dim=-1, descending=True)
        cum_before = sorted_p.cumsum(-1) - sorted_p                  # xác suất tích lũy trước expert j

        cap = min(self.max_k, k_cap) if k_cap else self.max_k
        cap = max(cap, self.min_k)
        if self.rand_k is not None:
            # đối chứng: mỗi token nhận k=hi với xác suất frac_hi, ngược lại k=lo (không dựa vào độ khó)
            lo, hi, frac = self.rand_k
            if k_cap:   # đánh giá: ép mọi token dùng đúng k_cap expert
                cap = torch.full((x.shape[0], 1), int(k_cap), device=x.device, dtype=torch.long)
            else:
                cap = lo + (torch.rand(x.shape[0], 1, device=x.device) < frac).long() * (hi - lo)

        d_logit = None
        if self.difficulty:
            # top_p theo từng token: token được dự đoán khó -> top_p cao -> nhiều expert hơn
            d_logit = self.diff_head(x).float().squeeze(-1)
            top_p = (self.p_min + (self.p_max - self.p_min) * torch.sigmoid(d_logit.detach())).unsqueeze(-1)
        else:
            top_p = self.top_p
        pos = torch.arange(E, device=x.device).unsqueeze(0)          # [1, E]
        keep = ((cum_before < top_p) | (pos < self.min_k)) & (pos < cap)

        soft_keep = torch.sigmoid((top_p - cum_before) / self.tau)
        keep_f = keep.float()
        if self.ste and self.training:
            # straight-through: forward dùng quyết định cứng, backward cho gradient của
            # task loss chảy vào ranh giới "có/không thêm expert" (chỉ ở vị trí không bị ép).
            free = ((pos >= self.min_k) & (pos < cap)).float()
            keep_f = keep_f + (soft_keep - soft_keep.detach()) * free
        w_sorted = sorted_p * keep_f
        if self.renorm:
            w_sorted = w_sorted / w_sorted.sum(-1, keepdim=True).clamp_min(1e-9)

        weights = torch.zeros_like(probs).scatter(1, sorted_idx, w_sorted)
        hard_mask = torch.zeros_like(probs).scatter(1, sorted_idx, keep.float())

        # "% sức mạnh": hệ số học được trong [min_strength, 1]
        strength = self.min_strength + (1 - self.min_strength) * torch.sigmoid(
            self.strength_head(x).float())
        weights = weights * strength

        # --- loss phụ ---
        # 1) cân bằng tải (kiểu Switch): E * sum(f_i * P_i)
        f = hard_mask.mean(0) / hard_mask.sum(-1).mean().clamp_min(1.0)
        balance = E * (f * probs.mean(0)).sum()
        # 2) compute: số expert "mềm" kỳ vọng (có gradient)
        soft_k = soft_keep.sum(-1).mean()
        if self.target_k is None:
            compute = soft_k / E
        else:
            compute = ((soft_k - self.target_k) / self.target_k) ** 2
        # 3) z-loss: giữ logits không phình to
        z_loss = torch.logsumexp(logits, dim=-1).pow(2).mean()

        info = {
            "avg_k": hard_mask.sum(-1).mean().detach(),
            "k_per_token": hard_mask.sum(-1).detach(),
            "avg_strength": strength.mean().detach(),
            "balance_loss": balance,
            "compute_loss": compute,
            "z_loss": z_loss,
        }
        if d_logit is not None:
            info["diff_logit"] = d_logit
        return weights.to(x.dtype), info


class AdaptiveMoE(nn.Module):
    """
    Layer MoE thích ứng. forward(x[B,S,D]) -> (out[B,S,D], aux_loss, info).
    Cũng lưu `last_aux` và `last_info` để dùng với gather_aux_loss / moe_stats.

    n_shared: số expert luôn bật (kiểu DeepSeek). Giúp chất lượng ổn khi router chọn k=1.
    d_ff_shared: độ rộng FFN của expert dùng chung (mặc định = d_ff). Nên đặt riêng khi d_ff của
    expert định tuyến rất nhỏ (nhiều expert mịn), để expert dùng chung không bị teo theo.
    max_resident: (chỉ khi eval) số expert tối đa giữ trên device, còn lại nằm CPU, nạp theo LRU.
    """

    def __init__(self, d_model, d_ff, n_experts, min_k=1, max_k=None, top_p=0.7,
                 min_strength=0.3, target_k=None, noise_std=0.0, ste=False,
                 difficulty=False, p_min=0.4, p_max=0.95, n_shared=0,
                 balance_coef=0.01, compute_coef=0.01, z_coef=1e-3,
                 max_resident=None, d_ff_shared=None, rand_k=None, renorm=True):
        super().__init__()
        self.experts = nn.ModuleList(Expert(d_model, d_ff) for _ in range(n_experts))
        self.shared = nn.ModuleList(Expert(d_model, d_ff_shared or d_ff) for _ in range(n_shared))
        self.router = AdaptiveRouter(d_model, n_experts, min_k, max_k, top_p,
                                     min_strength, target_k=target_k, noise_std=noise_std, ste=ste,
                                     difficulty=difficulty, p_min=p_min, p_max=p_max,
                                     rand_k=rand_k, renorm=renorm)
        self.balance_coef = balance_coef
        self.compute_coef = compute_coef
        self.z_coef = z_coef
        self.k_cap = None
        self.max_resident = max_resident
        self.last_aux = None
        self.last_info = None
        self.last_diff_logit = None
        self._lru = OrderedDict()   # expert_id -> None, thứ tự dùng gần nhất

    def _ensure_resident(self, e, device):
        if self.max_resident is None or self.training:
            return
        self._lru.pop(e, None)
        self._lru[e] = None
        self.experts[e].to(device)
        while len(self._lru) > self.max_resident:
            old, _ = self._lru.popitem(last=False)
            self.experts[old].to("cpu")

    @torch.no_grad()
    def release(self, keep=0):
        """Dọn dẹp sau khi trả lời: đẩy expert ra CPU (giữ lại `keep` expert dùng gần nhất)."""
        if self.max_resident is None:
            return
        while len(self._lru) > keep:
            old, _ = self._lru.popitem(last=False)
            self.experts[old].to("cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def forward(self, x):
        B, S, D = x.shape
        flat = x.reshape(-1, D)
        weights, info = self.router(flat, self.k_cap)
        out = torch.zeros_like(flat)
        for sh in self.shared:
            out = out + sh(flat)
        # Dispatch gộp: sắp xếp (token, expert) theo expert, chạy mỗi expert một lần trên
        # lát liên tiếp. Chỉ 2 lần đồng bộ host-device thay vì một lần cho mỗi expert.
        tok, exp = (weights > 0).nonzero(as_tuple=True)
        if tok.numel() > 0:
            order = exp.argsort(stable=True)
            tok, exp = tok[order], exp[order]
            counts = torch.bincount(exp, minlength=len(self.experts)).tolist()
            xs = flat[tok]
            ws = weights[tok, exp].unsqueeze(-1)
            ys, start = [], 0
            for e, n in enumerate(counts):
                if n == 0:
                    continue
                self._ensure_resident(e, flat.device)
                ys.append(self.experts[e](xs[start:start + n]))
                start += n
            out.index_add_(0, tok, (torch.cat(ys) * ws).to(out.dtype))
        aux = (self.balance_coef * info["balance_loss"]
               + self.compute_coef * info["compute_loss"]
               + self.z_coef * info["z_loss"])
        self.last_aux = aux
        self.last_diff_logit = info.get("diff_logit")
        self.last_info = {k: v.detach() for k, v in info.items() if k != "diff_logit"}
        return out.reshape(B, S, D), aux, info


# ----------------------------------------------------------------------------
# Tiện ích cho cả model
# ----------------------------------------------------------------------------
def set_k_cap(model: nn.Module, k):
    """Đặt trần số expert cho mọi AdaptiveMoE trong model (None = bỏ trần)."""
    for m in model.modules():
        if isinstance(m, AdaptiveMoE):
            m.k_cap = k


def gather_aux_loss(model: nn.Module):
    """Tổng loss phụ của mọi AdaptiveMoE sau một lần forward. loss = task_loss + gather_aux_loss(model)."""
    total = None
    for m in model.modules():
        if isinstance(m, AdaptiveMoE) and m.last_aux is not None:
            total = m.last_aux if total is None else total + m.last_aux
    if total is None:
        p = next(model.parameters(), None)
        return torch.zeros((), device=p.device if p is not None else "cpu")
    return total


def difficulty_loss(model: nn.Module, token_loss):
    """
    Loss huấn luyện "đầu dự đoán độ khó" của router (chỉ dùng khi difficulty=True).
    token_loss: loss CE từng token, cùng thứ tự flatten với đầu vào (B*S).
    Mục tiêu = phân vị của loss trong batch (0 = dễ nhất, 1 = khó nhất), đã detach.
    """
    t = token_loss.detach().reshape(-1).float()
    target = t.argsort().argsort().float() / max(1, t.numel() - 1)
    total = None
    for m in model.modules():
        if isinstance(m, AdaptiveMoE) and m.last_diff_logit is not None:
            l = F.binary_cross_entropy_with_logits(m.last_diff_logit, target)
            total = l if total is None else total + l
    return total if total is not None else torch.zeros((), device=t.device)


def moe_stats(model: nn.Module):
    """Thống kê của lần forward gần nhất: {tên layer: {avg_k, avg_strength}}."""
    stats = {}
    for name, m in model.named_modules():
        if isinstance(m, AdaptiveMoE) and m.last_info is not None:
            stats[name] = {
                "avg_k": float(m.last_info["avg_k"]),
                "avg_strength": float(m.last_info["avg_strength"]),
            }
    return stats


# ----------------------------------------------------------------------------
# Inference: nâng tầng theo độ tự tin
# ----------------------------------------------------------------------------
def _confidence(last_logits, mode="maxprob"):
    """Độ tự tin mỗi mẫu trong batch, thang 0..1."""
    p = F.softmax(last_logits.float(), dim=-1)
    if mode == "maxprob":
        return p.max(-1).values
    if mode == "entropy":
        ent = -(p * torch.log(p.clamp_min(1e-12))).sum(-1)
        return 1.0 - ent / math.log(p.shape[-1])
    raise ValueError("mode phải là 'maxprob' hoặc 'entropy'")


@torch.no_grad()
def cascade_forward(model, inputs, tiers=(2, 4, 8), min_confidence=0.6, confidence="maxprob"):
    """
    Chạy tầng nhỏ trước; nếu độ tự tin (token cuối) thấp hơn `min_confidence`
    thì nâng lên tầng kế tiếp. `model(inputs)` phải trả về logits [B, S, V].
    Trả về (logits, k_đã_dùng).
    """
    logits, k = None, tiers[-1]
    try:
        for k in tiers:
            set_k_cap(model, k)
            logits = model(inputs)
            if _confidence(logits[:, -1], confidence).min().item() >= min_confidence:
                break
    finally:
        set_k_cap(model, None)      # luôn bỏ trần k, kể cả khi forward lỗi
    return logits, k


@torch.no_grad()
def adaptive_generate(model, input_ids, max_new_tokens=32, tiers=(2, 4, 8),
                      min_confidence=0.6, confidence="maxprob", temperature=0.0,
                      eos_id=None):
    """
    Sinh text, mỗi bước tự chọn tầng (token dễ dùng ít expert, token khó nâng tầng).
    Chưa có KV cache: mỗi bước chạy lại toàn bộ chuỗi (đủ cho demo/nghiên cứu;
    model thật nên tự thêm KV cache).
    Trả về (ids, ks): ids [B, S+new], ks là danh sách tầng đã dùng cho từng bước.
    """
    ids = input_ids
    ks = []
    for _ in range(max_new_tokens):
        logits, k = cascade_forward(model, ids, tiers, min_confidence, confidence)
        last = logits[:, -1].float()
        if temperature and temperature > 0:
            nxt = torch.multinomial(F.softmax(last / temperature, dim=-1), 1)
        else:
            nxt = last.argmax(-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        ks.append(k)
        if eos_id is not None and (nxt == eos_id).all():
            break
    return ids, ks
