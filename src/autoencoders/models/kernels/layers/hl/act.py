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

def _sine_fwd(x: torch.Tensor) -> torch.Tensor:
    return torch.sin(x)

def _sine_bwd(g: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return g * torch.cos(x)

def _silu_fwd(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)

def _silu_bwd(g: torch.Tensor, y: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    sig = torch.sigmoid(x)
    return g * (sig * (1 + x * (1 - sig)))

# def _test_relu():
#     from . import test_layers as tl
#     tl._TRUE_FWD = nn.functional.relu
#     tl._POINTWISE_FWD = _relu_fwd
#     tl._POINTWISE_BWD = _relu_bwd
#     tl._pointwise_layer(False)
    
# def _test_sine():
#     from . import test_layers as tl
#     tl._TRUE_FWD = torch.sin
#     tl._POINTWISE_FWD = _sine_fwd
#     tl._POINTWISE_BWD = _sine_bwd
#     tl._pointwise_layer(True)
    
def _test_silu():
    from . import test_layers as tl
    tl._TRUE_FWD = nn.functional.silu
    tl._POINTWISE_FWD = _silu_fwd
    tl._POINTWISE_BWD = _silu_bwd
    tl._pointwise_layer(True)