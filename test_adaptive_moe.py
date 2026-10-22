import torch
import torch.nn as nn

from adaptive_moe import (AdaptiveMoE, AdaptiveRouter, adaptive_generate,
                          cascade_forward, gather_aux_loss, moe_stats, set_k_cap)


class TinyLM(nn.Module):
    def __init__(self, vocab=100, d=32, **moe_kw):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.moe = AdaptiveMoE(d, 64, n_experts=8, top_p=0.7, **moe_kw)
        self.head = nn.Linear(d, vocab)

    def forward(self, ids):
        h = self.emb(ids)
        y, _, _ = self.moe(h)
        return self.head(h + y)


def test_router_shapes_and_bounds():
    r = AdaptiveRouter(16, 8, min_k=1, max_k=6, top_p=0.7)
    w, info = r(torch.randn(50, 16))
    nz = (w > 0).sum(-1)
    assert w.shape == (50, 8)
    assert nz.min() >= 1 and nz.max() <= 6


def test_k_cap_limits_experts():
    r = AdaptiveRouter(16, 8, top_p=0.99)
    w, _ = r(torch.randn(50, 16), k_cap=2)
    assert (w > 0).sum(-1).max() <= 2


def test_training_step_has_gradients():
    m = TinyLM(n_shared=1, noise_std=0.1)
    ids = torch.randint(0, 100, (4, 10))
    logits = m(ids)
    loss = nn.functional.cross_entropy(logits.view(-1, 100), ids.view(-1)) + gather_aux_loss(m)
    loss.backward()
    assert m.moe.router.gate.weight.grad is not None
    assert m.moe.router.strength_head.weight.grad is not None
    assert m.moe.shared[0].w_up.weight.grad is not None


def test_target_k_loss():
    r = AdaptiveRouter(16, 8, top_p=0.7, target_k=2.0)
    _, info = r(torch.randn(50, 16))
    assert info["compute_loss"].item() >= 0


def test_ste_gives_boundary_gradient():
    r = AdaptiveRouter(16, 8, top_p=0.7, ste=True).train()
    x = torch.randn(30, 16, requires_grad=True)
    w, _ = r(x)
    (w * torch.arange(8.0)).sum().backward()
    assert r.gate.weight.grad is not None and x.grad is not None


def test_stats():
    m = TinyLM()
    m(torch.randint(0, 100, (2, 5)))
    s = moe_stats(m)
    assert "moe" in s and 1 <= s["moe"]["avg_k"] <= 8


def test_cascade_runs():
    m = TinyLM().eval()
    logits, k = cascade_forward(m, torch.randint(0, 100, (2, 6)), tiers=(2, 4, 8), min_confidence=0.99)
    assert logits.shape == (2, 6, 100)
    assert k in (2, 4, 8)
    assert m.moe.k_cap is None


def test_entropy_confidence_and_generate():
    m = TinyLM().eval()
    ids = torch.randint(0, 100, (2, 4))
    out, ks = adaptive_generate(m, ids, max_new_tokens=5, confidence="entropy", min_confidence=0.5)
    assert out.shape == (2, 9)
    assert len(ks) == 5


def test_lru_release():
    m = TinyLM(max_resident=2).eval()
    m(torch.randint(0, 100, (2, 6)))
    assert len(m.moe._lru) <= 2
    m.moe.release()
    assert len(m.moe._lru) == 0


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
    print("OK")
