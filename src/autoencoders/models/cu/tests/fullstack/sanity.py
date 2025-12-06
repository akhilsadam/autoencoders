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
        
    mem_pointer = nn_sanity.train(x, y, 0, 1)
    print("Mem pointer after training:", mem_pointer)
    
    # mem_pointer = nn_sanity.train(x, y, mem_pointer, 100)
    # print("Mem pointer after training:", mem_pointer)
    
    print(y.shape)
    print(y[0,0, :5, :5])

    # nn_sanity.eval(x, yhat, mem_pointer, 0)
    # print("Output after eval:", yhat)
    # error = torch.mean((y - yhat) ** 2).item()
    # signal = torch.mean((yhat - x) ** 2).item()
    # print("MSE after eval:", error)
    # print("Signal after eval:", signal)

    # from cu.tests import test_layers as tl
    # tl._check(nn.nn_sanity(), nn_sanity())
    # # tl._check(nn.Identity(), nn_sanity())
    
