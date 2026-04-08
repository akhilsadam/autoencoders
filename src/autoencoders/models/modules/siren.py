import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange

def coord(x):
    xs = torch.arange(x.shape[-1], device=x.device) / x.shape[-1] * 2 - 1
    ys = torch.arange(x.shape[-2], device=x.device) / x.shape[-2] * 2 - 1
    xy = torch.stack([xs[None,:].repeat(x.shape[-2],1),
                      ys[:,None].repeat(1,x.shape[-1])
                     ], dim=0)[None,...] # BCHW
    return xy
    
def coord_1d(x):
    zs = torch.arange(x.shape[-3], device=x.device) / x.shape[-3] * 2 - 1
    return zs[None,:,None,None] # BCHW

ups = lambda x, f: torch.nn.functional.pixel_shuffle(x[:,:,None,...].repeat(1,1,f**2,1,1), upscale_factor=f)[:,:,0,...]

class Sine(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        a = torch.sin(x)
        return a# + a**3 / 3

class SineW(nn.Module):
    def __init__(self, w):
        super().__init__()
        self.w = w
        
    def forward(self, x):
        a = torch.sin(x * self.w)
        return a# + a**3 / 3

class Linear(nn.Module):
    def __init__(self, in_features, out_features, kernel_size, w=1, act=nn.Identity, _id = 1, **kwargs):
        super().__init__()
        if kernel_size > 0:
            self.linear = nn.Conv2d(in_features, out_features, kernel_size, **kwargs, padding='same', padding_mode='zeros')
        else:
            self.linear = nn.Linear(in_features, out_features)
            kernel_size = 1
            
        self.in_features = in_features
        self.out_features = out_features

        # first, hidden, last : 0, 1, 2
        self.w = w if _id != 2 else 1
        
        if _id == 0:
            self.scale = (1 / self.in_features) / (kernel_size)
        else:
            self.scale = np.sqrt(6 / self.in_features) / (w * kernel_size)

        self.init_weights()
        self.act = act()
        
    def init_weights(self):
        with torch.no_grad():
            self.linear.weight.uniform_(-self.scale, self.scale)
            # self.linear.bias.uniform_(-0,0)
        
    def forward(self, x):
        a = self.w * self.linear(x)
        # print(self.in_features, self.out_features, torch.std(a))
        y = self.act(a)
        return y
    
class Siren(nn.Module):
    def __init__(self, c_in, c_out, width=256, layers=3, w=30, act=Sine, k=1):
        super().__init__()

        _layers = [
            Linear(width, width, k, w=w, act=act) for _ in range(layers)
        ]
        self.fwd = nn.Sequential(
            Linear(c_in, width, k, w=w, act=act, _id=0),
            *_layers,
            Linear(width, c_out, k, w=w, act=nn.Identity, _id = 2)            
        )
        
    def forward(self, x):
        return self.fwd(x)
    

class FiLM(nn.Module):
    def __init__(self, c_cond, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(c_cond, 2 * c_out),
            nn.SiLU(),
            nn.Linear(2 * c_out, 2 * c_out),
        )

    def forward(self, x, cond):
        gamma, beta = self.net(cond).chunk(2, dim=1)
        return x * (1 + gamma[..., None, None]) + beta[..., None, None]
    
class FiLMLayer(nn.Module):
    def __init__(self, c_in, c_out, c_cond, w=30, act=Sine, k=1, is_last=False):
        super().__init__()
        self.linear = Linear(c_in, c_out, k, w=w, act=act if not is_last else nn.Identity)
        self.film = FiLM(c_cond, c_out)
        self.is_last = is_last

    def forward(self, x, cond):
        x = self.linear(x)
        if not self.is_last:
            x = self.film(x, cond)
        return x

class FiLMSiren(nn.Module):
    def __init__(self, c_in, c_out, c_cond, width=256, layers=3, w=30, act=Sine, k=1):
        super().__init__()

        self.first = FiLMLayer(c_in, width, c_cond, w=w, act=act, k=k)

        self.hidden = nn.ModuleList([
            FiLMLayer(width, width, c_cond, w=w, act=act, k=k)
            for _ in range(layers)
        ])

        self.final = FiLMLayer(width, c_out, c_cond, w=w, act=act, k=k, is_last=True)

    def forward(self, x, cond):
        x = self.first(x, cond)
        for layer in self.hidden:
            x = layer(x, cond)
        x = self.final(x, cond)
        return x