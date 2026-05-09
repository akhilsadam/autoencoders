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
        
        acc = 0.0
        num_mse = 0.0
        num_rmse = 0.0
        nums = 0.0
        total = 0.0

        for tk_sent, dec_sent in zip(tks, rpns):
            tokens = tk_sent.split(' ')
            dec_tokens = dec_sent.split(' ')

            min_len = min(len(tokens), len(dec_tokens))

            # Compare overlapping tokens
            for tk, dec_tk in zip(tokens[:min_len], dec_tokens[:min_len]):
                total += 1.0

                if tk == dec_tk:
                    acc += 1.0

                if '.' in tk and '.' in dec_tk:
                    num_mse += (float(tk) - float(dec_tk)) ** 2
                    nums += 1.0

            # Count length mismatch only once per sentence
            if len(tokens) != len(dec_tokens):
                total += 1.0

        acc = acc / total if total > 0 else 0.0
        if nums > 0:
            num_rmse = math.sqrt(num_mse / nums) # rmse

        d = {'in':batch, 'out':rpns, 'token_check':tks, 'token_accuracy':acc, 'numerical_rmse':num_rmse}
        
        with open(os.path.join(dirs[0], f'rpn_gen_{i:04d}.json'), 'w') as f:
            json.dump(d, f, indent=4)
            
        break