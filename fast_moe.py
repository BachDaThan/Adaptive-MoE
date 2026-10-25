"""
fast_moe.py - chạy AdaptiveMoE nhanh hơn trên GPU bằng cách thay vòng lặp Python qua từng expert bằng 3 phép bmm.

adaptive_moe.py đã gom token theo expert (sắp xếp, mỗi expert chạy một lần), nên cái còn lại là vòng lặp tối đa E lần,
mỗi lần ~5 kernel nhỏ. Ở đây: xếp token của mọi expert vào một tensor [E, C, D] (C = số token của expert đông nhất,
phần thừa là 0), rồi chạy SwiGLU của cả E expert bằng 3 phép bmm. Kết quả y hệt vòng lặp (không bỏ token nào).

Chỉ nhanh khi tải giữa các expert đủ cân bằng: nếu E*C > fast_waste * (số cặp token-expert) thì tự quay về vòng lặp cũ,
vì padding sẽ tốn hơn lợi. Router không có loss cân bằng tải (train_lm.py --aux 0, mặc định) thường lệch nhiều,
khi đó thử --aux 1 để tải đều hơn.

Không đổi tham số hay state_dict nên dùng được với checkpoint đã train:
    from fast_moe import enable_fast_moe
    enable_fast_moe(model)            # model chứa các AdaptiveMoE
Không dùng với max_resident (offload expert), và công cụ đếm expert bằng hook trên expert (langcheck.py) sẽ không thấy
expert nào khi đang dùng đường bmm.
"""
import torch
import torch.nn.functional as F
from adaptive_moe import AdaptiveMoE


class FastAdaptiveMoE(AdaptiveMoE):
    fast_waste = 3.0

    def forward(self, x):
        B, S, D = x.shape
        flat = x.reshape(-1, D)
        weights, info = self.router(flat, self.k_cap)
        out = torch.zeros_like(flat)
        for sh in self.shared:
            out = out + sh(flat)
        E = len(self.experts)
        tok, exp = (weights > 0).nonzero(as_tuple=True)
        self.last_path = "none"
        if tok.numel() > 0:
            order = exp.argsort(stable=True)
            tok, exp = tok[order], exp[order]
            counts = torch.bincount(exp, minlength=E)
            cl = counts.tolist()
            total, C = tok.numel(), max(cl)
            self.last_counts = counts.detach()
            xs = flat[tok]
            ws = weights[tok, exp].unsqueeze(-1)
            if self.max_resident is None and E * C <= self.fast_waste * total:
                self.last_path = "bmm"
                starts = counts.cumsum(0) - counts
                slot = exp * C + (torch.arange(total, device=flat.device) - starts[exp])
                xp = flat.new_zeros(E * C, D).index_copy(0, slot, xs.to(flat.dtype)).view(E, C, D)
                wg = torch.stack([e.w_gate.weight for e in self.experts])   # [E, F, D]
                wu = torch.stack([e.w_up.weight for e in self.experts])     # [E, F, D]
                wd = torch.stack([e.w_down.weight for e in self.experts])   # [E, D, F]
                h = F.silu(torch.bmm(xp, wg.transpose(1, 2).to(xp.dtype))) * torch.bmm(xp, wu.transpose(1, 2).to(xp.dtype))
                yp = torch.bmm(h, wd.transpose(1, 2).to(h.dtype)).reshape(E * C, D)
                ys = yp[slot]
            else:
                self.last_path = "loop"
                ys_list, start = [], 0
                for e, n in enumerate(cl):
                    if n == 0:
                        continue
                    self._ensure_resident(e, flat.device)
                    ys_list.append(self.experts[e](xs[start:start + n]))
                    start += n
                ys = torch.cat(ys_list)
            out.index_add_(0, tok, (ys * ws).to(out.dtype))
        aux = (self.balance_coef * info["balance_loss"]
               + self.compute_coef * info["compute_loss"]
               + self.z_coef * info["z_loss"])
        self.last_aux = aux
        self.last_diff_logit = info.get("diff_logit")
        self.last_info = {k: v.detach() for k, v in info.items() if k != "diff_logit"}
        return out.reshape(B, S, D), aux, info


def enable_fast_moe(model, waste=3.0):
    """Đổi mọi AdaptiveMoE trong model sang FastAdaptiveMoE (cùng tham số). Trả về số layer đã đổi."""
    n = 0
    for m in model.modules():
        if isinstance(m, AdaptiveMoE) and not isinstance(m, FastAdaptiveMoE):
            m.__class__ = FastAdaptiveMoE
            m.fast_waste = waste
            n += 1
    return n


def load_report(model):
    """Sau một forward: với mỗi layer MoE, (tải expert đông nhất / tải trung bình, đường đã chạy 'bmm' hoặc 'loop')."""
    rep = []
    for m in model.modules():
        if isinstance(m, FastAdaptiveMoE) and getattr(m, "last_counts", None) is not None:
            c = m.last_counts.float()
            rep.append((float(c.max() / c.mean().clamp_min(1e-9)), m.last_path))
    return rep
