import types
import torch
import helion
import helion.language as hl

def random_reduce(err):
    weights = torch.rand_like(err)
    loss = weights * (err) ** 2
    return loss.mean()

_POINTWISE_FWD = None
_POINTWISE_BWD = None

@helion.kernel(autotune_effort="quick")
def _pointwise_fwd_kernel(x: torch.Tensor) -> torch.Tensor:
    _b, _w = x.size()
    out = torch.empty_like(x)
    for tile_b in hl.tile(_b):
        out[tile_b, :] = _POINTWISE_FWD(x[tile_b, :])
    return out

@helion.kernel(autotune_effort="quick")
def _pointwise_bwd_kernel(g: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    _b, _w = g.size()
    out = torch.empty_like(g)
    for tile_b in hl.tile(_b):
        out[tile_b, :] = _POINTWISE_BWD(g[tile_b, :], y[tile_b, :])
    return out

def _pointwise_layer():
    # test data
    torch.manual_seed(42)
    x = torch.randn(1024, 512, device="cuda")
    x.requires_grad_()

    y = true_fwd(x) # compute true forward
    y_hat = _pointwise_fwd_kernel(x)              # compute helion forward
    assert torch.allclose(y_hat, y) # verify forward

    # compute true backward
    y.retain_grad() # make sure we can access y.grad
    random_reduce(y).backward()
    x_g = x.grad

    x_g_hat = _pointwise_bwd_kernel(y.grad, y)   # compute helion backward
    assert torch.allclose(x_g, x_g_hat) # verify backward