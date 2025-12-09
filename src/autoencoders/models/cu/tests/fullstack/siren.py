import os
import torch
from torch import nn

# Convention: model class is named 'Autoencoder' or endswith 'Autoencoder', config is 'Config' or endswith 'Config'

from cu.compile import compile
nn_siren = compile(
    kernel=os.path.join(os.path.dirname(__file__), "siren.cu"),
)
##
# compiled?
print("Network compiled:", nn_siren is not None)

def _test_nn_basic_siren():
    cx = torch.linspace(-1.0, 1.0, steps=32)[None, :].repeat(32, 1)
    cy = torch.linspace(-1.0, 1.0, steps=32)[:, None].repeat(1, 32)
    x = torch.stack([cx, cy, cx], dim=0).unsqueeze(0).cuda()  # Shape: (1, 2, 32, 32)
    
    f = lambda c: torch.sin(4*c[:,0] + 2*c[:,1]) + 0.5*torch.cos(3*c[:,0] - c[:,1])
    
    y = f(x)
    
    
    yhat = torch.zeros_like(y)
        
    mem_pointer = 0
    mem_pointer = nn_siren.train(x, y, mem_pointer, 1) # warmup quirk
    for i in range(10):
        # load new data too every iteration, technically
        mem_pointer = nn_siren.train(x, y, mem_pointer, 100)
        # print("Mem pointer after training:", mem_pointer)

    nn_siren.eval(x, yhat, mem_pointer, 0)
    # print("Output after eval:", yhat)
    error = torch.mean((y - yhat) ** 2)
    signal = torch.mean((yhat - x) ** 2)
    print("MSE:", error.item())
    print("Signal after eval:", signal)
    print(f"SNR: {10 * torch.log10(signal / error).item():.2f} dB")
    
    from cu.tests import test_layers as tl
    tl._plot_diff(x, yhat, title="NN Siren Check Difference")
    
    assert error < 1e-4, f"NN Siren Check failed with MSE {error}"
