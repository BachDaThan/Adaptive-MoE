"""
bitnet_yui.py — BitNet b1.58 tự chế cho Yui (Adaptive-MoE + yui.cpp + Y-App). 1 file, MIT.

Khớp đúng kiến trúc của bạn:
  * Chỉ chuyển nn.Linear của attention (qkv, o) và expert/shared expert (w_gate, w_up, w_down).
    GIỮ float: router (gate, strength_head, diff_head), embedding, head, LayerNorm.
  * KHÔNG thêm tham số mới, KHÔNG đổi tên khoá state_dict (lam là số Python, không vào checkpoint)
    -> checkpoint cũ nạp thẳng vào model ternary, export_yui.py đọc như cũ.
  * Kích hoạt lượng tử int8 THEO KHỐI 32 phần tử (y như quant_act trong yui.cpp) nên train == chạy thật.
  * Trọng số: ternary {-1,0,+1} với scale absmean THEO NHÓM (mặc định 64 cột, bội số của 32).
So với BitNet gốc: scale theo nhóm (ít lỗi hơn), fade-in từ fp (tiếp tục được checkpoint cũ),
giữ router fp để routing không vỡ, và có kernel riêng trong yui.cpp (kiểu .yui "t2").
Thật thà: kernel không nhanh hơn bitnet.cpp (chỉ AVX2/NEON, không có LUT); lợi thế là chất lượng + một engine
dùng chung cho PC, Android (Y-App) và server.
"""
import re
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F

ACT_BLOCK = 32                                   # = khối lượng tử kích hoạt trong yui.cpp
SKIP_DEFAULT = (r"router", r"(^|\.)head$", r"(^|\.)emb", r"lm_head")   # regex trên tên module


def _group(in_f, g):
    """Nhóm scale: bội số của 32 và chia hết in_f; không được thì 0 (lớp đó giữ nguyên fp/int8)."""
    if in_f % ACT_BLOCK:
        return 0
    g = max(ACT_BLOCK, (g // ACT_BLOCK) * ACT_BLOCK)
    while g > ACT_BLOCK and in_f % g:
        g -= ACT_BLOCK
    return g


@torch.no_grad()
def ternarize(w, group):
    """w [O,I] -> (q int8 [O,I] trong {-1,0,1}, scale float32 [O,I/group]); absmean từng nhóm."""
    O, I = w.shape
    wg = w.float().reshape(O, I // group, group)
    s = wg.abs().mean(-1, keepdim=True).clamp(min=1e-5)
    q = (wg / s).round().clamp(-1, 1)
    return q.reshape(O, I).to(torch.int8), s.squeeze(-1)


def quant_act(x, block=ACT_BLOCK):
    """int8 đối xứng theo khối 32 (giống quant_act C++: d = amax/127, q = round(x/d)); STE."""
    dt, n = x.dtype, x.shape[-1]
    xf = x.float()
    pad = (-n) % block
    if pad:
        xf = F.pad(xf, (0, pad))
    xb = xf.reshape(*xf.shape[:-1], -1, block)
    d = (xb.detach().abs().amax(-1, keepdim=True) / 127.0).clamp(min=1e-8)
    xq = ((xb / d).round().clamp(-127, 127) * d).reshape(*xf.shape)[..., :n]
    return (x.float() + (xq - x.float()).detach()).to(dt)


class BitLinear(nn.Linear):
    def __init__(self, in_features, out_features, bias=False, group=64, act_quant=True):
        super().__init__(in_features, out_features, bias=bias)
        self.group, self.act_quant = _group(in_features, group), act_quant
        self.lam = 1.0   # số Python thường (không phải buffer): 0 = fp, 1 = ternary đầy đủ; không vào state_dict, không ép đồng bộ GPU

    @classmethod
    def from_linear(cls, lin, **kw):
        m = cls(lin.in_features, lin.out_features, bias=lin.bias is not None, **kw)
        m.weight.data.copy_(lin.weight.data)
        if lin.bias is not None:
            m.bias.data.copy_(lin.bias.data)
        return m.to(lin.weight.device, lin.weight.dtype)

    def qweight(self):
        w = self.weight
        with torch.no_grad():
            q, s = ternarize(w, self.group)
            wq = (q.float().reshape(w.shape[0], -1, self.group) * s.unsqueeze(-1)).reshape(w.shape).to(w.dtype)
        return w + self.lam * (wq - w.detach())                          # STE: gradient chảy thẳng vào w

    def forward(self, x):
        if self.lam > 0 and self.act_quant:
            x = x + self.lam * (quant_act(x) - x)
        return F.linear(x, self.qweight(), self.bias)


def convert_to_bitnet(model, group=64, skip=SKIP_DEFAULT, act_quant=True, verbose=True):
    """Thay nn.Linear -> BitLinear tại chỗ (giữ trọng số). Trả về số lớp đã chuyển."""
    pats = [re.compile(p) for p in skip]
    n, kept = 0, []
    for name, mod in list(model.named_modules()):
        for cname, child in list(mod.named_children()):
            full = f"{name}.{cname}" if name else cname
            if type(child) is not nn.Linear:
                continue
            if any(p.search(full) for p in pats) or _group(child.in_features, group) == 0:
                kept.append(full)
                continue
            setattr(mod, cname, BitLinear.from_linear(child, group=group, act_quant=act_quant))
            n += 1
    if verbose:
        print(f"[bitnet] {n} lớp -> BitLinear (nhóm {group}, kích hoạt int8 khối {ACT_BLOCK}); giữ fp: {len(kept)} lớp (router/head/...)", flush=True)
        if n == 0:
            print("[bitnet] ⚠️ 0 lớp được chuyển: kiểm tra tên module / kiểu lớp.", flush=True)
    return n


def set_bit_lambda(model, v):
    v = float(max(0.0, min(1.0, v)))
    for m in model.modules():
        if isinstance(m, BitLinear):
            m.lam = v
    return v


# ---------------- định dạng ternary của yui.cpp (kiểu ma trận 3 = "t2") ----------------
from bitnet_pack import pack_t2, unpack_t2, ternary_quantize_np   # định dạng t2 (numpy thuần), dùng chung với export_yui.py


def selftest():
    torch.manual_seed(0)
    q = np.random.randint(-1, 2, (7, 96)).astype(np.int8)
    assert (unpack_t2(pack_t2(q)) == q).all()
    print("✅ pack/unpack t2 khớp 100%")
    w = torch.randn(48, 128)
    q1, s1 = ternarize(w, 64)
    s2, _, q2 = ternary_quantize_np(w.numpy(), 64)
    assert (q1.numpy() == q2).all() and np.allclose(s1.numpy(), s2.astype(np.float32), rtol=2e-3)
    print("✅ ternarize (torch) == ternary_quantize_np (export)")
    net = nn.Sequential(nn.Linear(64, 256), nn.GELU(), nn.Linear(256, 64), nn.GELU(), nn.Linear(64, 1))
    X = torch.randn(512, 64); Y = X[:, :8].sum(1, keepdim=True).tanh()
    assert convert_to_bitnet(net, group=32, skip=(r"^4$",), verbose=False) == 2
    opt = torch.optim.AdamW(net.parameters(), 3e-3)
    for st in range(500):
        set_bit_lambda(net, st / 250)
        loss = F.mse_loss(net(X), Y); opt.zero_grad(); loss.backward(); opt.step()
    print(f"✅ QAT ternary: loss {loss.item():.4f} (baseline {Y.var().item():.4f})")
    assert loss.item() < Y.var().item() * 0.6
    sd = net.state_dict(); assert not any(k.endswith("lam") for k in sd), "lam không được vào checkpoint"
    print("✅ state_dict không có khoá mới -> checkpoint cũ/mới dùng chung")
    print("🎉 selftest OK")


if __name__ == "__main__":
    import sys
    selftest() if len(sys.argv) > 1 and sys.argv[1] == "selftest" else print(__doc__)
