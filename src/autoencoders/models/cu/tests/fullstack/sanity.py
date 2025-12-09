import os
import torch
from torch import nn

# Convention: model class is named 'Autoencoder' or endswith 'Autoencoder', config is 'Config' or endswith 'Config'

from cu.compile import compile
nn_sanity = compile(
    kernel=os.path.join(os.path.dirname(__file__), "sanity.cu"),
)
##
# compiled?
print("Network compiled:", nn_sanity is not None)

def _test_nn_sanity():
        # return nn.train
        
    x = torch.randn(1, 3, 32, 32).cuda()
    y = x * 2.78
    yhat = torch.zeros_like(y)
        
    mem_pointer = 0
    mem_pointer = nn_sanity.train(x, y, mem_pointer, 1) # warmup quirk
    for i in range(100):
        # load new data too every iteration, technically
        mem_pointer = nn_sanity.train(x, y, mem_pointer, 1000)
        # print("Mem pointer after training:", mem_pointer)

    nn_sanity.eval(x, yhat, mem_pointer, 0)
    # print("Output after eval:", yhat)
    error = torch.mean((y - yhat) ** 2)
    signal = torch.mean((yhat - x) ** 2)
    print("MSE:", error.item())
    print("Signal after eval:", signal)
    print(f"SNR: {10 * torch.log10(signal / error).item():.2f} dB")
    
    from cu.tests import test_layers as tl
    tl._plot_diff(x, yhat, title="NN Sanity Check Difference")
    
    assert error < 1e-4, f"NN Sanity Check failed with MSE {error}"

    # from cu.tests import test_layers as tl
    # tl._check(nn.nn_sanity(), nn_sanity())
    # # tl._check(nn.Identity(), nn_sanity())
    
