import torch
import torch.nn as nn
import torch.nn.functional as F

class PixelLayer(nn.Module):
    def __init__(self, factor=2, reverse=False):
        super(PixelLayer, self).__init__()
        self.factor = factor
        
        if not reverse:
            self._up = nn.PixelShuffle(factor)
            self._dn = nn.PixelUnshuffle(factor)
        else:
            self._up = nn.PixelUnshuffle(factor)
            self._dn = nn.PixelShuffle(factor)
            
    def forward(self, x):
        y = self._dn(x)
        return y
    
    def reverse(self, x):
        return self._up(x)