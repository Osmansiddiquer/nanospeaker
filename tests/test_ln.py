"""Tests for layer normalization, including the fused and hand-rolled paths agreeing."""

import pytest
import torch

from src.model.ln import LayerNormalization

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])


@pytest.fixture(autouse=True)
def _seed():
    torch.manual_seed(0)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("shape", [(32,), (5, 32), (2, 7, 32)])
def test_fused_matches_the_written_out_form(device, shape):
    """F.layer_norm has to be the same arithmetic as the expanded version it replaces."""
    ln = LayerNormalization(32).to(device).double()
    with torch.no_grad():                       # non-trivial affine, not just identity
        ln.gamma.normal_(1.0, 0.2)
        ln.beta.normal_(0.0, 0.2)
    x = torch.randn(*shape, device=device, dtype=torch.double)

    ln.fused = True
    fused = ln(x)
    ln.fused = False
    assert torch.allclose(fused, ln(x), atol=1e-12)


@pytest.mark.parametrize("fused", [True, False])
def test_matches_torch_layernorm(fused):
    ln = LayerNormalization(32, fused=fused).double()
    ref = torch.nn.LayerNorm(32).double()
    with torch.no_grad():
        ln.gamma.normal_(1.0, 0.2), ln.beta.normal_(0.0, 0.2)
        ref.weight.copy_(ln.gamma), ref.bias.copy_(ln.beta)

    x = torch.randn(4, 6, 32, dtype=torch.double)
    assert torch.allclose(ln(x), ref(x), atol=1e-12)


@pytest.mark.parametrize("fused", [True, False])
def test_output_is_normalized(fused):
    ln = LayerNormalization(32, fused=fused).double()
    y = ln(torch.randn(8, 32, dtype=torch.double) * 5 + 3)
    assert torch.allclose(y.mean(-1), torch.zeros(8, dtype=torch.double), atol=1e-12)
    assert torch.allclose(y.std(-1, unbiased=False), torch.ones(8, dtype=torch.double), atol=1e-6)


@pytest.mark.parametrize("fused", [True, False])
def test_gradients_reach_both_parameters(fused):
    ln = LayerNormalization(32, fused=fused)
    ln(torch.randn(4, 32)).square().mean().backward()
    assert ln.gamma.grad.abs().sum() > 0 and ln.beta.grad.abs().sum() > 0


def test_constant_input_does_not_divide_by_zero():
    """Zero variance is what epsilon exists for; both paths must survive it."""
    x = torch.full((3, 32), 2.0)
    for fused in (True, False):
        assert torch.isfinite(LayerNormalization(32, fused=fused)(x)).all()
