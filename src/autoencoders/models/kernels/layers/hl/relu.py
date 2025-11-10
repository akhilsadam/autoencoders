import os
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'
os.environ['TRITON_BACKEND'] = 'cuda'
# important for helion (else finds nvc, not nvcc)
# need it here for pytest

import torch
import torch.nn as nn
import helion
import helion.language as hl


# device functions
def _relu_fwd(x: torch.Tensor) -> torch.Tensor:
    return torch.clamp(x, min=0)

def _relu_bwd(g: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    return g * (y > 0).to(g.dtype)

def _test_relu():
    from . import test_layers as tl
    tl._POINTWISE_FWD = _relu_fwd
    tl._POINTWISE_BWD = _relu_bwd
    tl._pointwise_layer()