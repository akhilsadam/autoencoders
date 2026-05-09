import os
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'
os.environ['TRITON_BACKEND'] = 'cuda'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import math

def generation(net, loader, dirs):
    for i, batch in enumerate(loader):
        tks = net.detokenize(*net.tokenize(batch))
        rpns = net.decode(net.encode(batch))
        
        tokens = ' '.join(tks).split(' ') # Split the tokens into a list
        dec_tokens = ' '.join(rpns).split(' ') # Split the decoded tokens into a list
        # accuracy
        acc = 0.0
        num_mse = 0.0
        num_rmse = 0.0
        nums = 0.0
        for tk, dec_tk in zip(tokens, dec_tokens):
            if tk == dec_tk:
                acc += 1.0
            if '.' in tk and '.' in dec_tk:
                num_mse += (float(tk) - float(dec_tk))**2
                nums += 1.0
            
        
        if nums > 0:
            num_rmse = math.sqrt(num_mse / nums) # rmse
        acc = acc / len(tokens)

        d = {'in':batch, 'out':rpns, 'token_check':tks, 'token_accuracy':acc , 'numerical_rmse':num_rmse}
        
        with open(os.path.join(dirs[0], f'rpn_gen_{i:04d}.json'), 'w') as f:
            json.dump(d, f, indent=4)