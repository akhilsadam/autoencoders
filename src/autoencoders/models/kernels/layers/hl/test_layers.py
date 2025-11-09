import torch
import helion
import helion.language as hl
from functools import partial

def random_reduce(err):
    weights = torch.rand_like(err)
    loss = weights * (err) ** 2
    return loss.mean()


# tester kernels
def make_pointwise_kernels(_p_fwd,
                           _p_bwd):
    def _pointwise_fwd_kernel(x: torch.Tensor, _pointwise_fwd) -> torch.Tensor:
        _b, _w = x.size()
        out = torch.empty_like(x)
        for tile_b in hl.tile(_b):
            out[tile_b, :] = _pointwise_fwd(x[tile_b, :])
        return out

    def _pointwise_bwd_kernel(g: torch.Tensor, y: torch.Tensor, _pointwise_bwd) -> torch.Tensor:
        _b, _w = g.size()
        out = torch.empty_like(g)
        for tile_b in hl.tile(_b):
            out[tile_b, :] = _pointwise_bwd(g[tile_b, :], y[tile_b, :])
        return out

    # rename for uniqueness
    _pointwise_fwd_kernel.__name__ = "_pointwise_fwd_" + str(id(_pointwise_fwd))
    _pointwise_bwd_kernel.__name__ = "_pointwise_bwd_" + str(id(_pointwise_bwd))

    return partial(_pointwise_fwd_kernel, _pointwise_fwd=_p_fwd), \
           partial(_pointwise_bwd_kernel, _pointwise_bwd=_p_bwd)


def _pointwise(_pointwise_fwd,
               _pointwise_bwd,
               true_fwd):
    
    _pointwise_fwd_kernel, _pointwise_bwd_kernel = make_pointwise_kernels(_pointwise_fwd, _pointwise_bwd)
        
    hl_pointwise_fwd = helion.kernel(autotune_effort="quick")(_pointwise_fwd_kernel)
    hl_pointwise_bwd = helion.kernel(autotune_effort="quick")(_pointwise_bwd_kernel)
    
    # test data
    torch.manual_seed(42)
    x = torch.randn(1024, 512, device="cuda")
    x.requires_grad_()
    
    y = true_fwd(x) # compute true forward
    y_hat = hl_pointwise_fwd(x)              # compute helion forward
    assert torch.allclose(y_hat, y) # verify forward
    
    # compute true backward
    y.requires_grad_()
    random_reduce(y).backward()
    x_g = x.grad
    
    x_g_hat = hl_pointwise_bwd(y.grad, y)   # compute helion backward
    assert torch.allclose(x_g, x_g_hat) # verify backward