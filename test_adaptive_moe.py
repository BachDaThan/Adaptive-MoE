import torch
import torch.nn as nn

from adaptive_moe import AdaptiveMoE, AdaptiveRouter, cascade_forward, set_k_cap


class TinyLM(nn.Module):
    def __init__(self, vocab=100, d=32):
        super().__init__()
        self.emb = nn.Embedding(vocab, d)
        self.moe = AdaptiveMoE(d, 64, n_experts=8, top_p=0.7)
        self.head = nn.Linear(d, vocab)
        self.aux = 0.0

    def forward(self, ids):
        h = self.emb(ids)
        y, aux, _ = self.moe(h)
        self.aux = aux
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
    m = TinyLM()
    ids = torch.randint(0, 100, (4, 10))
    logits = m(ids)
    loss = nn.functional.cross_entropy(logits.view(-1, 100), ids.view(-1)) + m.aux
    loss.backward()
    assert m.moe.router.gate.weight.grad is not None
    assert m.moe.router.strength_head.weight.grad is not None


def test_cascade_runs():
    m = TinyLM().eval()
    logits, k = cascade_forward(m, torch.randint(0, 100, (2, 6)), tiers=(2, 4, 8), min_confidence=0.99)
    assert logits.shape == (2, 6, 100)
    assert k in (2, 4, 8)
    assert m.moe.k_cap is None


if __name__ == "__main__":
    test_router_shapes_and_bounds()
    test_k_cap_limits_experts()
    test_training_step_has_gradients()
    test_cascade_runs()
    print("OK")
