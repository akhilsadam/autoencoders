import os
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'
os.environ['TRITON_BACKEND'] = 'cuda'
# important for helion (else finds nvc, not nvcc)
# need it here for pytest

import helion
import helion.language as hl

from .util import random_reduce

# device functions
def _relu_fwd(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=0)

def _relu_bwd(g: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return g * (y > 0).to(g.dtype)



def test_relu_layer():
    
    # tester kernels
    @helion.kernel(autotune_effort="quick")
    def _relu_k(x: torch.Tensor) -> torch.Tensor:
        _b, _w = x.size()
        out = torch.empty_like(x)
        for tile_b in hl.tile(_b):
            out[tile_b, :] = _relu_fwd(x[tile_b, :])
        return out

    @helion.kernel(autotune_effort="quick")
    def _relu_grad_k(g: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        _b, _w = g.size()
        out = torch.empty_like(g)
        for tile_b in hl.tile(_b):
            out[tile_b, :] = _relu_bwd(g[tile_b, :], y[tile_b, :])
        return out
    
    # test data
    torch.manual_seed(42)
    x = torch.randn(1024, 512, device="cuda")
    x.requires_grad_()
    
    y = torch.nn.functional.relu(x) # compute true forward
    y_hat = _relu_k(x)              # compute helion forward
    assert torch.allclose(y_hat, y) # verify forward
    
    # compute true backward
    y.requires_grad_()
    random_reduce(y).backward()
    x_g = x.grad
    
    x_g_hat = _relu_grad_k(y.grad, y)   # compute helion backward
    assert torch.allclose(x_g, x_g_hat) # verify backward


if __name__ == "__main__":
    test_relu_layer()