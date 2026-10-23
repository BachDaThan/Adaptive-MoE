import torch
import torch.nn as nn

from adaptive_moe import (AdaptiveMoE, AdaptiveRouter, adaptive_generate,
                          cascade_forward, difficulty_loss, gather_aux_loss, moe_stats, set_k_cap)


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


def test_difficulty_head():
    m = TinyLM(difficulty=True)
    ids = torch.randint(0, 100, (4, 10))
    logits = m(ids)
    tok = nn.functional.cross_entropy(logits.view(-1, 100), ids.view(-1), reduction="none")
    loss = tok.mean() + gather_aux_loss(m) + 0.1 * difficulty_loss(m, tok)
    loss.backward()
    assert m.moe.router.diff_head.weight.grad is not None
    assert 1 <= m.moe.last_info["avg_k"].item() <= 8


def test_matches_dense_reference():
    # dispatch gộp phải cho kết quả giống cách tính dày: sum_e w_e * expert_e(x) (+ shared)
    m = AdaptiveMoE(16, 32, n_experts=6, n_shared=1).eval()
    x = torch.randn(2, 7, 16)
    out, _, _ = m(x)
    flat = x.reshape(-1, 16)
    w, _ = m.router(flat, None)
    ref = sum(sh(flat) for sh in m.shared) + sum(
        w[:, e:e + 1] * ex(flat) for e, ex in enumerate(m.experts))
    assert torch.allclose(out.reshape(-1, 16), ref, atol=1e-5)


def test_state_dict_roundtrip():
    a = TinyLM(difficulty=True, n_shared=1).eval()
    b = TinyLM(difficulty=True, n_shared=1).eval()
    b.load_state_dict(a.state_dict())
    ids = torch.randint(0, 100, (2, 9))
    assert torch.allclose(a(ids), b(ids), atol=1e-6)


def test_small_shapes_are_finite():
    m = TinyLM().eval()
    for shape in [(1, 1), (1, 5), (3, 1)]:
        out = m(torch.randint(0, 100, shape))
        assert out.shape == (*shape, 100) and torch.isfinite(out).all()


def test_extreme_inputs_stay_finite():
    r = AdaptiveRouter(16, 8, top_p=0.7)
    w, info = r(torch.randn(40, 16) * 1e3)
    assert torch.isfinite(w).all()
    assert all(torch.isfinite(v).all() for v in info.values())


def test_bf16_autocast_cpu():
    m = TinyLM(difficulty=True)
    ids = torch.randint(0, 100, (2, 8))
    with torch.autocast("cpu", dtype=torch.bfloat16):
        logits = m(ids)
        loss = nn.functional.cross_entropy(logits.float().view(-1, 100), ids.view(-1)) + gather_aux_loss(m)
    loss.backward()
    assert torch.isfinite(loss)


def test_random_k_control():
    r = AdaptiveRouter(16, 8, min_k=1, max_k=2, top_p=1.0, rand_k=(1, 2, 0.5))
    w, _ = r(torch.randn(2000, 16))
    nz = (w > 0).sum(-1)
    assert nz.min() >= 1 and nz.max() <= 2
    assert 1.4 < nz.float().mean().item() < 1.6


def test_no_renorm_top1_gives_router_gradient():
    r = AdaptiveRouter(16, 8, min_k=1, max_k=1, top_p=1.0, renorm=False)
    w, _ = r(torch.randn(20, 16))
    (w * torch.arange(8.0)).sum().backward()
    assert r.gate.weight.grad.abs().sum() > 0


def test_random_k_forced_by_cap():
    r = AdaptiveRouter(16, 8, min_k=1, max_k=2, top_p=1.0, rand_k=(1, 2, 0.5))
    w2, _ = r(torch.randn(100, 16), k_cap=2)
    w1, _ = r(torch.randn(100, 16), k_cap=1)
    assert ((w2 > 0).sum(-1) == 2).all() and ((w1 > 0).sum(-1) == 1).all()


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
