import os
os.environ['CC'] = 'gcc'
os.environ['CXX'] = 'g++'
os.environ['TRITON_BACKEND'] = 'cuda'
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import jpcm.draw as draw
cmap = draw.cmap

import json

from einops import rearrange

from torchvision.utils import save_image

def plot(recon, output_dir, name, nrow=4):
    if recon.shape[1] == 1:
        cmap = draw.cmap
        recon = recon.squeeze(1)
        # normalize to [0,1] for colormap
        recon = (recon - recon.min()) / (recon.max() - recon.min() + 1e-8)
        # cmap returns RGBA, take RGB and convert to tensor in CHW format
        recon = torch.from_numpy(cmap(recon)[...,:3]).permute(0, 3, 1, 2).float()
        save_image(recon[:8], os.path.join(output_dir, name), nrow=nrow)
    else:
        raise NotImplementedError("Plotting only implemented for single-channel images")

def rplot(recon, output_dir, name):
    # recon is B Y C H W, make grid of recon, error, input
    recon = rearrange(recon, 'b y t c h w -> b c (y h) (t w)')
    plot(recon, output_dir, name, nrow=3)

def reconstruction(net, loader, dirs):
    with torch.no_grad():
        n = len(loader)
        results = []
        loss = 0.0
        rpn_list = []
        for fused_batch in loader:
            rpns, batch = fused_batch 
            batch = batch.to(next(net.parameters()).device)
            
            latent = net.compute_latent(rpns)
            
            zloss = 0.0
            y_hats = []
            x = batch[:, 0]  # B C H W
            for i in range(batch.shape[1] - 1):                
                y_hat = net.gen(x, x, latent=latent)
                y_hats.append(y_hat.detach())
                # update
                x = y_hat

            y_hat = torch.stack(y_hats, dim=1)  # B T C H W
            y = batch[:, 1:]  # B T C H W
            zloss = F.mse_loss(y_hat, y) / F.mse_loss(y, y.mean(dim=(-2,-1), keepdim=True))
            
            stack = torch.stack([y, y_hat, y_hat - y], dim=1)  # P Y T C H W

            results.append(stack.detach().cpu())
            rpn_list.append(rpns)
            loss += zloss.item() / n
        results = torch.stack(results, dim=0) # B P Y T C H W # B is batch time, P is pde, Y is type
        
        torch.save(results, os.path.join(dirs[1], "reconstructions.pt"))
        rplot(results[0], dirs[0], "surrogate_reco_batchfirst.png")
        rplot(results[-1], dirs[0], "surrogate_reco_batchlast.png")
        with open(os.path.join(dirs[0], f'rpns_saved.json'),'w') as f:
            json.dump(data, f, indent=4)
        
iter = 0        
def quick_reconstruction(net, rpns, batch, dirs, info, **kwargs):
    global iter
    if iter % 200 == 0:
        with torch.no_grad():
            loss = 0.0
    
            batch = batch.to(next(net.parameters()).device)
                
            y_hats = []
            x = batch[:, 0]  # B C H W
            for i in range(batch.shape[1] - 1):                
                y_hat = net.gen(x, x, **kwargs)
                y_hats.append(y_hat.detach())
                # update
                x = y_hat

            y_hat = torch.stack(y_hats, dim=1)  # B T C H W
            y = batch[:, 1:]  # B T C H W
            # zloss = F.mse_loss(y_hat, y) / F.mse_loss(y, y.mean(dim=(-2,-1), keepdim=True))
            
            stack = torch.stack([y, y_hat, y_hat - y], dim=1)  # B Y T C H W
            stack = stack.detach().cpu()
            rplot(stack[0:4], dirs[0], f"surrogate_reco_batch_{info}_{iter:04d}.png")
            with open(os.path.join(dirs[0], f'rpns_{iter:04d}.txt'),'w') as f:
                f.write('\n'.join(rpns))
                
    iter += 1
    
# def reconstruction_step(x, y, net):
#     start = torch.cuda.Event(enable_timing=True)
#     end = torch.cuda.Event(enable_timing=True)
    
#     times = []
    
    
#             start.record()
#             end.record()
            
#             torch.cuda.synchronize()
#             times.append(start.elapsed_time(end))  #
            
#             data.append(x_hat.detach())
        
#         torch.cuda.synchronize()
#         torch.cuda.empty_cache()
        
#         data = torch.cat(data, dim=0).cpu()
#         torch.save(data, os.path.join(dirs[1], "generated_samples.pt"))
        
#         times = np.array(times)
#         min_time = times.min()
#         max_time = times.max()
#         median_time = np.median(times)
#         net.log('gen_time_min', min_time)
#         net.log('gen_time_max', max_time)
#         net.log('gen_time', median_time)
        
#         with open(os.path.join(dirs[1], "generation_times_summary.txt"), "w") as f:
#             f.write(f"Min time: {min_time:.2f} ms\n")
#             f.write(f"Max time: {max_time:.2f} ms\n")
#             f.write(f"Median time: {median_time:.2f} ms\n")
#         np.save(os.path.join(dirs[1], "generation_times.npy"), times)
        
#         plot(torch.from_numpy(data[:8]), dirs[0], "generated_samples.png")      
