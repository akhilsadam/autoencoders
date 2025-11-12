"""Convolutional autoencoder Lightning module and configuration."""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Dict, Tuple

import torch
from torch import nn

# Convention: model class is named 'Autoencoder' or endswith 'Autoencoder', config is 'Config' or endswith 'Config'

from .cu.compile import compile
activations = compile(
    device_functions=[],
    kernel="src/autoencoders/models/cu/kernels/act.cu",
)
##
# compiled?
print("Activations compiled:", activations is not None)

def _test_relu():
    class _ReLU(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            y = torch.empty_like(x)
            activations.relu_fwd(x, y)
            ctx.save_for_backward(y)
            return y

        @staticmethod
        def backward(ctx, grad_y):
            y, = ctx.saved_tensors
            grad_x = torch.empty_like(grad_y)
            activations.relu_bwd(grad_y, y, grad_x)
            return grad_x
        
    class ReLU(nn.Module):
        def forward(self, x):
            return _ReLU.apply(x)

    from . import test_layers as tl
    tl._check(nn.ReLU(), ReLU())
