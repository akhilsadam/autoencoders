import os
import torch
from torch import nn

# Convention: model class is named 'Autoencoder' or endswith 'Autoencoder', config is 'Config' or endswith 'Config'

from cu.compile import compile
activations = compile(
    kernel=os.path.join(os.path.dirname(__file__), "conv.cu"),
)
##
# compiled?
print("Activations compiled:", activations is not None)

def _test_conv():
    class _Conv(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x, W, s, p):
            y = torch.empty_like(x)
            activations.conv_fwd(x, y, W, s, p)
            ctx.save_for_backward(y)
            return y

        @staticmethod
        def backward(ctx, grad_y, W, s, p):
            y, = ctx.saved_tensors
            grad_x = torch.empty_like(grad_y)
            # placeholder for x argument
            activations.conv_bwd(grad_y, y, grad_x, y, W, s, p)
            return grad_x
        
    class Conv(nn.Module):
        def forward(self, x, w, s, p):
            return _Conv.apply(x, w[None, :, :, :], s, p)

    class TorchConv(nn.Module):
        def forward(self, xs, w, stride, padding):
            cs = [
                torch.nn.functional.conv2d(xs[:,i:i+1], w[None, i:i+1], bias=None, stride=stride, padding=(2,2))
                for i in range(xs.shape[1])
                ]
            return torch.cat(cs, dim=1)
        
    from cu.tests import test_layers as tl
    
    
    _sizes=[
        (1, 16, 64, 64),
        (32, 16, 64, 64),
        (4, 16, 128, 128),
        (4, 16, 512, 512),
        (16, 16, 28, 28)
    ]
    
    W = torch.randn(16, 5, 5, device="cuda")
    tl_f = TorchConv()
    test_f = Conv()
    
    tl._check(
        lambda x: tl_f(x, W, 1, 0),
        lambda x: test_f(x, W, 1, 0))
    
    # tl._check(
    #     lambda x: tl_f(x, W, stride=2, padding=0),
    #     lambda x: test_f(x, W, stride=2, padding=0))
    # tl._check(nn.Identity(), conv())
    
