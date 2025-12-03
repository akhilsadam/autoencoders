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
        
    x = torch.randn(10, 3, 32, 32).cuda()
    y = x * 2.78
        
    nn_sanity.train(x, y)

    # from cu.tests import test_layers as tl
    # tl._check(nn.nn_sanity(), nn_sanity())
    # # tl._check(nn.Identity(), nn_sanity())
    
