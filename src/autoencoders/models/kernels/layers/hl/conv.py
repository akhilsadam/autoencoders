import os
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'
os.environ['TRITON_BACKEND'] = 'cuda'
# important for helion (else finds nvc, not nvcc)
# need it here for pytest

import torch
import helion
from helion._testing import run_example
import helion.language as hl

# # investigate "device functions"
# def add_5(x: torch.Tensor) -> torch.Tensor:
#     return x + 5

# def add_5_grad(x: torch.Tensor) -> torch.Tensor:
#     return x

# @helion.kernel(autotune_effort="quick")
# def h_plus(x: torch.Tensor) -> torch.Tensor:
#     _b, _w = x.size()
#     out = torch.empty_like(x)
#     for tile_b in hl.tile(_b):
#         out[tile_b, :] = add_5(x[tile_b, :])
#     return out

# @helion.kernel(autotune_effort="quick")
# def h_plus_grad(x: torch.Tensor) -> torch.Tensor:
#     _b, _w = x.size()
#     out = torch.empty_like(x)
#     for tile_b in hl.tile(_b):
#         out[tile_b, :] = add_5_grad(x[tile_b, :])
#     return out
