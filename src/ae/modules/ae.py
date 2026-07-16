import torch.nn as nn

from .shuffle import PixelLayer
from .patch_att import PatchAttLayer
from .spatial import SpatialLayer, SpatialLayer2


class AE(nn.Module):
    def __init__(self):
        super(AE, self).__init__()

    def encoder(self, x):
        for layer in self.mlist:
            x = layer(x)
        return x
    
    def decoder(self, z):
        for layer in reversed(self.mlist):
            z = layer.reverse(z)
        return z

    def forward(self, x):
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return x_hat
    
class TFSpatialAutoencoder(AE):
    def __init__(self, in_dim, lift_steps=3, encode_layers=3, p=16, att_layers=3, factor=2):
        super().__init__()

        mlist = []

        if lift_steps > 0:
            # for i in range(lift_steps):
            #     mlist.append(PixelLayer(factor))
            #     in_dim = in_dim * (factor**2)
            mlist.append(PixelLayer(factor**lift_steps))
            in_dim = in_dim * ((factor**2)**lift_steps)
                
            mlist.append(PatchAttLayer(in_dim, p=p, layers=att_layers))

            # for i in range(lift_steps):
            #     mlist.append(PixelLayer(factor, reverse=True))
            #     in_dim = in_dim // (factor**2)
            
            mlist.append(PixelLayer(factor**lift_steps, reverse=True))
            in_dim = in_dim // ((factor**2)**lift_steps)
            
        for i in range(encode_layers):
            mlist.append(
                SpatialLayer(in_dim, factor=factor),
            )

        self.latent_dim = in_dim
        self.mlist = nn.ModuleList(mlist)


class BasicSpatialAutoencoder(nn.Module):
    """Spatial autoencoder with multiple encode/decode layers"""

    def __init__(self, in_dim, encode_layers=3, factor=2, spatial_layer=SpatialLayer2):
        super().__init__()

        mlist = []
        for i in range(encode_layers):
            mlist.append(spatial_layer(in_dim, factor=factor))

        self.latent_dim = in_dim
        self.mlist = nn.ModuleList(mlist)

    def encoder(self, x):
        """Encode input through all layers."""
        z = x
        for layer in self.mlist:
            z = layer(z)
        return z

    def decoder(self, z):
        """Decode latent representation through all layers in reverse."""
        x = z
        for layer in reversed(self.mlist):
            x = layer.reverse(x)
        return x

    def forward(self, x):
        return self.decoder(self.encoder(x))