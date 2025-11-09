import torch
import helion
import helion.language as hl

def random_reduce(err):
    weights = torch.rand_like(err)
    loss = weights * (err) ** 2
    return loss.mean()

def _pointwise(_pointwise_fwd,
               _pointwise_bwd):
    
    # tester kernels
    @helion.kernel(autotune_effort="quick")
    def _pointwise_k(x: torch.Tensor) -> torch.Tensor:
        _b, _w = x.size()
        out = torch.empty_like(x)
        for tile_b in hl.tile(_b):
            out[tile_b, :] = _pointwise_fwd(x[tile_b, :])
        return out

    @helion.kernel(autotune_effort="quick")
    def _pointwise_grad_k(g: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _b, _w = g.size()
        out = torch.empty_like(g)
        for tile_b in hl.tile(_b):
            out[tile_b, :] = _pointwise_bwd(g[tile_b, :], y[tile_b, :])
        return out
    
    # test data
    torch.manual_seed(42)
    x = torch.randn(1024, 512, device="cuda")
    x.requires_grad_()
    
    y = torch.nn.functional.pointwise(x) # compute true forward
    y_hat = _pointwise_k(x)              # compute helion forward
    assert torch.allclose(y_hat, y) # verify forward
    
    # compute true backward
    y.requires_grad_()
    random_reduce(y).backward()
    x_g = x.grad
    
    x_g_hat = _pointwise_grad_k(y.grad, y)   # compute helion backward
    assert torch.allclose(x_g, x_g_hat) # verify backward