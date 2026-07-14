import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SkipLayer(nn.Module):
    def __init__(self, in_dim, out_dim, scale = 1.0):
        super().__init__()
        linear = torch.zeros(out_dim, in_dim)
        # print(f"SkipLayer: {in_dim} -> {out_dim}")
        
        if in_dim == out_dim:
            linear.data[:, :] = torch.eye(in_dim) * scale
        elif in_dim > out_dim:
            group_size = int(in_dim // out_dim)
            linear.data[:, :] = \
            torch.eye(out_dim).repeat(1, group_size) * 0.8 * scale / np.sqrt(group_size) + \
            0.25 * sum([
                torch.roll(
                    torch.eye(out_dim).repeat_interleave(group_size, dim=1) * 0.2 * scale / np.sqrt(group_size),
                    out_dim * i,
                    1)
                for i in range(group_size)
            ])
            # torch.rand_like(linear.data) * (np.sqrt(1/in_dim)) # worst
            # torch.eye(out_dim).repeat(1, group_size) * scale / np.sqrt(group_size)
            # torch.eye(out_dim).repeat_interleave(group_size, dim=1) * scale / np.sqrt(group_size)
        else:
            # ignored
            group_size = int(out_dim // in_dim)
            linear.data[:, :] = torch.eye(in_dim).repeat_interleave(group_size, dim=0) * scale / np.sqrt(group_size)

        # self.register_buffer('linear',linear)
        self.linear = nn.Parameter(linear)

    def forward(self, input):
        # print(input.shape, self.linear.shape)
        return torch.einsum('bchw,dc->bdhw', input, self.linear)

    def reverse(self, input):
        return torch.einsum('bdhw,dc->bchw', input, self.linear)

class Skip(nn.Module):
    def __init__(self, net):
        super(Skip, self).__init__()
        self.net = net
            
    def forward(self, x):
        return self.net(x) + x