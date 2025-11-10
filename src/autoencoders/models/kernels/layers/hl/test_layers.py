import types
import torch
import helion
import helion.language as hl

def random_reduce(err):
    weights = torch.rand_like(err)
    loss = weights * (err) ** 2
    return loss.mean()

_TRUE_FWD = None
_POINTWISE_FWD = None
_POINTWISE_BWD = None

@helion.kernel(autotune_effort="none")
def _pointwise_fwd_kernel(x: torch.Tensor) -> torch.Tensor:
    _b, _w = x.size()
    out = torch.empty_like(x)
    for tile_b in hl.tile(_b):
        out[tile_b, :] = _POINTWISE_FWD(x[tile_b, :])
    return out

@helion.kernel(autotune_effort="none")
def _pointwise_bwd_kernel_unsaved(g: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    _b, _w = g.size()
    out = torch.empty_like(g)
    for tile_b in hl.tile(_b):
        out[tile_b, :] = _POINTWISE_BWD(g[tile_b, :], y[tile_b, :])
    return out

@helion.kernel(autotune_effort="none")
def _pointwise_bwd_kernel(g: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    _b, _w = g.size()
    out = torch.empty_like(g)
    for tile_b in hl.tile(_b):
        out[tile_b, :] = _POINTWISE_BWD(g[tile_b, :], y[tile_b, :], x[tile_b, :])
    return out

def _pointwise_layer(state_saved: bool = False):
    # test data
    torch.manual_seed(42)
    x = torch.randn(1024, 512, device="cuda")
    x.requires_grad_()

    y = _TRUE_FWD(x) # compute true forward
    y_hat = _pointwise_fwd_kernel(x)              # compute helion forward
    assert torch.allclose(y_hat, y) # verify forward

    # compute true backward
    y.retain_grad() # make sure we can access y.grad
    random_reduce(y).backward()
    x_g = x.grad

    if state_saved:
        x_g_hat = _pointwise_bwd_kernel(y.grad, y, x)   # compute helion backward
    else:
        x_g_hat = _pointwise_bwd_kernel_unsaved(y.grad, y)   # compute helion backward
    assert torch.allclose(x_g, x_g_hat) # verify backward