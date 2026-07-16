import torch
import torch.nn as nn

from .shuffle import PixelLayer
from .skip import SkipLayer, Skip
from .siren import Siren, Sine
from .act import Swish, Tri


class SpatialLayer(nn.Module):
    def __init__(self, input_dim, factor=2, scale=1, freq=5, k=3):
        super(SpatialLayer, self).__init__()
        self.factor = factor
        in_dim = input_dim * (factor**2) * (factor**2)
        out_dim = input_dim * (factor**2) * int(scale)

        self.pixel = PixelLayer(factor)
        self.squash = SkipLayer(in_dim, out_dim)

        w = in_dim

        print(in_dim)

        self.fnet = Skip(Siren(in_dim, in_dim, width=w, layers=0, w=freq, k=k, act=Sine))
        self.fnet2 = Siren(in_dim, out_dim, width=w, layers=2, w=freq, k=k, act=Swish)
        self.rnet = Skip(Siren(in_dim, in_dim, width=w, layers=0, w=freq, k=k, act=Sine))
        self.rnet2 = Siren(out_dim, in_dim, width=w, layers=2, w=freq, k=k, act=Swish)

    def init_weights(self, s, x):
        with torch.no_grad():
            if hasattr(x, 'weight'):
                x.weight.uniform_(-s, s)
                x.bias.uniform_(-0,0)

    def forward(self, z):
        z = self.pixel(self.pixel(z))
        z = self.fnet(z)
        z = self.squash(z) + self.fnet2(z)
        z = self.pixel.reverse(z)
        return z

    def reverse(self, z):
        z = self.pixel(z)
        z = self.squash.reverse(z) + self.rnet2(z)
        z = self.rnet(z)
        z = self.pixel.reverse(self.pixel.reverse(z))
        return z


class SpatialLayer2(nn.Module):
    """Spatial processing layer with local and frequency components (v2)."""

    def __init__(self, input_dim, factor=2, scale=1, freq=5, k=3):
        super().__init__()
        self.factor = factor
        in_dim = input_dim * (factor**2) * (factor**2)
        out_dim = input_dim * (factor**2) * int(scale)

        self.pixel = PixelLayer(factor)
        self.squash = SkipLayer(in_dim, out_dim)

        w = in_dim * 2

        self.fnet = Skip(Siren(in_dim, in_dim, width=w, layers=2, w=0.5, k=k, act=Sine))
        self.fnet2 = Siren(in_dim, out_dim, width=w, layers=6, w=freq, k=k, act=Swish)

        self.rnet = Skip(Siren(in_dim, in_dim, width=w, layers=2, w=0.5, k=k, act=Sine))
        self.rnet2 = Siren(out_dim, in_dim, width=w, layers=6, w=freq, k=k, act=Swish)

    def forward(self, z):
        z = self.pixel(self.pixel(z))
        z = self.fnet(z)
        z = self.squash(z) + self.fnet2(z)
        z = self.pixel.reverse(z)
        return z

    def reverse(self, z):
        z = self.pixel(z)
        z = self.squash.reverse(z) + self.rnet2(z)
        z = self.rnet(z)
        z = self.pixel.reverse(self.pixel.reverse(z))
        return z