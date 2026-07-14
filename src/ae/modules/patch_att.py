


import torch.nn as nn
from einops import rearrange


class PatchAtt(nn.Module):
    def __init__(self, c, px=8, num_heads=8, layers = 6, act=nn.GELU):
        super().__init__()
        self.px = px
        self.c = c
        
        # Embedding dimension is channels (c)
        # Sequence length is the grid (h * w)
        self.mha = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=c, 
                    num_heads=num_heads, 
                    batch_first=True
                ),
        ] * layers)
        
        # self.norm = nn.LayerNorm(c)
        self.act = act()
        
        # Output projection step
        self.proj = nn.ModuleList([
                nn.Linear(c, c),
        ] * layers)

    # @torch.compile
    def forward(self, x):
        b, c, H, W = x.shape
        p = self.px
        w_grid = W // p
        h_grid = H // p
        
        # 1. inside the patch is sequence
        out = rearrange(x, 'b c (h p1) (w p2) -> (b h w) (p1 p2) c', p1=p, p2=p)

        # print(out.shape)
        # 2. Attention along the (p1 p2) dimension
        for i in range(len(self.mha)):
            attn_out, _ = self.mha[i](out, out, out)
            out = self.proj[i](self.act(attn_out)) + out
            # out = (self.act(attn_out / out.shape[-1])) + out
            
        
        # 4. Reshape back to the original (b, c, H, W)
        out = rearrange(out, '(b h w) (p1 p2) c -> b c (h p1) (w p2)', 
                        b=b, p1=p, p2=p, h=h_grid, w=w_grid)
        
        return out
    
class PatchAttLayer(nn.Module):
    def __init__(self, in_dim, p=16, heads = 1, layers=3):
        super().__init__()
        print('ATT:', in_dim, p)
        self.fnet = PatchAtt(in_dim, p, heads, layers)
        self.rnet = PatchAtt(in_dim, p, heads, layers)

    def forward(self, x):
        return self.fnet(x)
    def reverse(self, x):
        return self.rnet(x)