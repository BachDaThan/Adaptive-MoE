"""
adaptive_moe.py - MoE với số chuyên gia và "sức mạnh" thay đổi theo từng token.

Thành phần:
  AdaptiveRouter : chọn số expert theo top-p (token khó -> nhiều expert hơn),
                   kèm "strength" (0..1) điều chỉnh mức đóng góp của expert.
  AdaptiveMoE    : layer MoE dùng router trên, có k_cap (trần số expert) và
                   cơ chế giữ expert trên GPU theo LRU + release() để dọn dẹp.
  set_k_cap      : đặt trần số expert cho cả model.
  cascade_forward: chạy tầng nhỏ trước, kém tự tin thì nâng tầng (2 -> 4 -> 8...).

Chỉ phụ thuộc PyTorch.
"""
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F


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

    Trả về weights [T, E] (0 ở expert không được chọn) và dict thông tin/loss phụ.
    """

    def __init__(self, d_model, n_experts, min_k=1, max_k=None, top_p=0.7,
                 min_strength=0.3, tau=0.05):
        super().__init__()
        self.n_experts = n_experts
        self.min_k = min_k
        self.max_k = max_k or n_experts
        self.top_p = top_p
        self.min_strength = min_strength
        self.tau = tau
        self.gate = nn.Linear(d_model, n_experts, bias=False)
        self.strength_head = nn.Linear(d_model, 1)

    def forward(self, x, k_cap=None):
        T, E = x.shape[0], self.n_experts
        probs = F.softmax(self.gate(x).float(), dim=-1)            # [T, E]
        sorted_p, sorted_idx = probs.sort(dim=-1, descending=True)
        cum_before = sorted_p.cumsum(-1) - sorted_p                 # xác suất tích lũy trước expert j

        cap = min(self.max_k, k_cap) if k_cap else self.max_k
        cap = max(cap, self.min_k)

        pos = torch.arange(E, device=x.device).unsqueeze(0)         # [1, E]
        keep = (cum_before < self.top_p) | (pos < self.min_k)
        keep = keep & (pos < cap)

        w_sorted = sorted_p * keep
        w_sorted = w_sorted / w_sorted.sum(-1, keepdim=True).clamp_min(1e-9)

        weights = torch.zeros_like(probs).scatter(1, sorted_idx, w_sorted)
        hard_mask = torch.zeros_like(probs).scatter(1, sorted_idx, keep.float())

        # "% sức mạnh": hệ số học được trong [min_strength, 1]
        strength = self.min_strength + (1 - self.min_strength) * torch.sigmoid(
            self.strength_head(x).float())
        weights = weights * strength

        # loss phụ
        # 1) cân bằng tải (kiểu Switch): f_i * P_i * E
        f = hard_mask.mean(0) / hard_mask.sum(-1).mean().clamp_min(1.0)
        balance = E * (f * probs.mean(0)).sum()
        # 2) phạt compute: số expert "mềm" kỳ vọng, có gradient
        soft_keep = torch.sigmoid((self.top_p - cum_before) / self.tau)
        compute = soft_keep.sum(-1).mean() / E

        info = {
            "avg_k": hard_mask.sum(-1).mean().detach(),
            "avg_strength": strength.mean().detach(),
            "balance_loss": balance,
            "compute_loss": compute,
        }
        return weights.to(x.dtype), info


class AdaptiveMoE(nn.Module):
    """
    Layer MoE thích ứng. forward(x[B,S,D]) -> (out[B,S,D], aux_loss, info).
    Tổng loss: task_loss + aux_loss.

    max_resident: (chỉ dùng khi eval/inference) số expert tối đa giữ trên device,
    các expert còn lại nằm trên CPU và được nạp theo LRU. None = tắt.
    """

    def __init__(self, d_model, d_ff, n_experts, min_k=1, max_k=None, top_p=0.7,
                 min_strength=0.3, balance_coef=0.01, compute_coef=0.01,
                 max_resident=None):
        super().__init__()
        self.experts = nn.ModuleList(Expert(d_model, d_ff) for _ in range(n_experts))
        self.router = AdaptiveRouter(d_model, n_experts, min_k, max_k, top_p, min_strength)
        self.balance_coef = balance_coef
        self.compute_coef = compute_coef
        self.k_cap = None
        self.max_resident = max_resident
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
        for e, expert in enumerate(self.experts):
            idx = (weights[:, e] > 0).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            self._ensure_resident(e, flat.device)
            y = expert(flat[idx]) * weights[idx, e].unsqueeze(-1)
            out.index_add_(0, idx, y.to(out.dtype))
        aux = self.balance_coef * info["balance_loss"] + self.compute_coef * info["compute_loss"]
        return out.reshape(B, S, D), aux, info


def set_k_cap(model: nn.Module, k):
    """Đặt trần số expert cho mọi AdaptiveMoE trong model (None = bỏ trần)."""
    for m in model.modules():
        if isinstance(m, AdaptiveMoE):
            m.k_cap = k


@torch.no_grad()
def cascade_forward(model, inputs, tiers=(2, 4, 8), min_confidence=0.6):
    """
    Chạy tầng nhỏ trước; nếu độ tự tin (max prob của token cuối) thấp hơn
    `min_confidence` thì nâng lên tầng kế tiếp.

    `model(inputs)` phải trả về logits [B, S, V].
    Trả về (logits, k_đã_dùng).
    """
    logits = None
    for k in tiers:
        set_k_cap(model, k)
        logits = model(inputs)
        conf = F.softmax(logits[:, -1].float(), dim=-1).max(-1).values
        if conf.min().item() >= min_confidence:
            break
    set_k_cap(model, None)
    return logits, k
