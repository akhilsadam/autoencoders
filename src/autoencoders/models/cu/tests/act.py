import os
import torch
from torch import nn

# Convention: model class is named 'Autoencoder' or endswith 'Autoencoder', config is 'Config' or endswith 'Config'

from cu.compile import compile
activations = compile(
    kernel=os.path.join(os.path.dirname(__file__), "act.cu"),
)
##
# compiled?
print("Activations compiled:", activations is not None)

def _test_relu():
    class _ReLU(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            # y = torch.empty_like(x)
            # activations.relu_fwd(x, y)
            y = x.clone()  # placeholder
            ctx.save_for_backward(y)
            return y

        @staticmethod
        def backward(ctx, grad_y):
            y, = ctx.saved_tensors
            grad_x = torch.empty_like(grad_y)
            # placeholder for x argument
            activations.relu_bwd(grad_y, y, grad_x, y)
            return grad_x
        
    class ReLU(nn.Module):
        def forward(self, x):
            return _ReLU.apply(x)

    from cu.tests import test_layers as tl
    tl._check(nn.ReLU(), ReLU())
