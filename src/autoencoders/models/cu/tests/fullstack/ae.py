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

def _test_conv2d():
    class _Conv2D(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x):
            y = torch.empty_like(x)
            activations.Conv2D_fwd(x, y)
            ctx.save_for_backward(y)
            return y

        @staticmethod
        def backward(ctx, grad_y):
            y, = ctx.saved_tensors
            grad_x = torch.empty_like(grad_y)
            # placeholder for x argument
            activations.Conv2D_bwd(grad_y, y, grad_x, y)
            return grad_x
        
    class Conv2D(nn.Module):
        def forward(self, x):
            return _Conv2D.apply(x)

    from cu.tests import test_layers as tl
    tl._check(nn.Conv2D(), Conv2D())
    # tl._check(nn.Identity(), Conv2D())
    
