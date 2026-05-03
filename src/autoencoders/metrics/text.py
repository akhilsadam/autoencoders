import os
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'
os.environ['TRITON_BACKEND'] = 'cuda'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json

def generation(net, loader, dirs):
    for i, batch in enumerate(loader):
        tk = net.detokenize(*net.tokenize(batch))
        rpns = net.decode(net.encode(batch))
        d = {'in':batch, 'out':rpns, 'token_check':tk}
        
        with open(os.path.join(dirs[0], f'rpn_gen_{i:04d}.json'), 'w') as f:
            json.dump(d, f, indent=4)