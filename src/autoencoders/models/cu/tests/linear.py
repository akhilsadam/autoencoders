import os
import torch
from torch import nn

# Convention: model class is named 'Autoencoder' or endswith 'Autoencoder', config is 'Config' or endswith 'Config'

def _test_linear():

    from cu.compile import compile
    activations = compile(
        kernel=os.path.join(os.path.dirname(__file__), "linear.cu"),
    )
    ##
    # compiled?
    print("Activations compiled:", activations is not None)
    
    class _Linear(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, W, b):
            y = torch.empty_like(x)
            activations.linear_fwd(x, y, W, b)
            ctx.save_for_backward(y)
            return y

        @staticmethod
        def backward(ctx, grad_y, W, b):
            y, = ctx.saved_tensors
            grad_x = torch.empty_like(grad_y)
            # placeholder for x argument
            activations.linear_bwd(grad_y, y, grad_x, y, W, b)
            return grad_x
        
    class Linear(nn.Module):
        def forward(self, x, w, b):
            return _Linear.apply(x, w[None, None, :, :], b[None, None, None, :])

    class TorchLinear(nn.Module):
        def forward(self, x, w, b):
            return torch.einsum("cC, bChw -> bchw", w, x) + b.view(1, -1, 1, 1)
        
    from cu.tests import test_layers as tl
    
    
    _sizes=[
        (1, 64, 64, 64),
        (32, 64, 64, 64),
        (4, 64, 128, 128),
        (4, 64, 512, 512),
        (16, 64, 28, 28)
    ]
    
    W = torch.randn(32, 64, device="cuda")
    b = torch.randn(32, device="cuda")
    tl_f = TorchLinear()
    test_f = Linear()
    
    tl._check(
        lambda x: tl_f(x, W, b),
        lambda x: test_f(x, W, b))
    
    # tl._check(nn.Identity(), Linear())
    
