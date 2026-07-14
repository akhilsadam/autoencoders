import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_dct as dct

class Swish(nn.Module):
    def forward(self,x):
        return x * torch.sigmoid(x) 

# from fast_hadamard_transform import hadamard_transform as dht
# def dht2(x):
#     d = dht(x) / torch.sqrt(x.shape[-1])
#     d = d.mT
#     d = dht(d) / torch.sqrt(d.shape[-1])
#     return d

class Tri(nn.Module):
    def forward(self,z):
        return (2/torch.pi) * torch.arcsin(0.98 * torch.sin(z))
    
# class Tri2(nn.Module):
#     def forward(self,z):
#         z = z * 2 / torch.pi
#         t1 = torch.frac(z+1)
#         t2 = torch.frac(0.5*z + 2)
#         q = 1 - t1**(1.6)
#         return q * torch.sign(t2) * 0.85

def saw(x):
    return x - torch.floor(x)
    
class Tri2(nn.Module):
    def forward(self, x):
        a = saw(0.25 * x - 0.25) - 0.5  # saw in [0,1)
        b = saw(0.25 * x - 0.75) - 0.5
        t = 4 * (a + b) * (a - b) # a.pow(2) - b.pow(2)
        return t
        
class Sharp(nn.Module):
    def forward(self,z):
        return torch.sin(z) + torch.sin(z)**3/3
class Sinc(nn.Module):
    def forward(self,z):
        # return torch.sinc(z)
        return torch.sin(z) / (z + 1e-10)

class Finer(nn.Module):
    def forward(self,z):
        # mostly correct
        return torch.sin((torch.abs(z) + 1) * z)

class FFT(nn.Module):
    def forward(self,z):
        return dct.dct_2d(z)
class IFFT(nn.Module):
    def forward(self,z):
        return dct.idct_2d(z)