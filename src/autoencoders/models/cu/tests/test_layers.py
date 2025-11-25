# generate random data for testing cu layers
import torch
import numpy as np
from matplotlib import pyplot as plt

sizes=[
    # (16, 1, 28, 28),
    (32, 3, 64, 64),
    (4, 3, 128, 128),
    (4, 3, 512, 512),
]

def random_reduce(err, weights=None):
    if weights is None:
        weights = torch.rand_like(err)
    loss = weights * (err) ** 2
    return loss.mean(), weights

def get_random_data(batch_size: int, channels: int, height: int, width: int) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(batch_size, channels, height, width, device="cuda")

def _plot_diff(true, cu, title="Difference for B0 C0"):
    true = true[0,0].detach().cpu().numpy()
    cu = cu[0,0].detach().cpu().numpy()
    diff = (true - cu)
    fig, ax = plt.subplots(3,1, figsize=(6,12))
    
    im = ax[0].imshow(true, cmap='coolwarm')
    ax[0].set_title("True")
    plt.colorbar(im, ax=ax[0])
    
    im = ax[1].imshow(cu, cmap='coolwarm')
    ax[1].set_title("CU")
    plt.colorbar(im, ax=ax[1])

    im = ax[2].imshow(diff, cmap='hot')
    ax[2].set_title("Difference")
    plt.colorbar(im, ax=ax[2])

    plt.suptitle(title)
    plt.savefig("diff.png")
    plt.close()
    return "See diff.png for difference heatmap"

def _check(true_func, cu_func):
    for size in sizes:
        x = get_random_data(*size)
        x.requires_grad_()
        
        x_2 = x.clone().detach()
        x_2.requires_grad_()
        
        try:
            y = true_func(x)
            y_hat = cu_func(x_2)
            assert torch.allclose(y_hat, y), f"Forward check failed for size {size} -- {_plot_diff(y, y_hat)}"
            
            lz, wt = random_reduce(y)
            lz.backward()
            x_g = x.grad
            
            random_reduce(y_hat, weights=wt)[0].backward()
            x_g_hat = x_2.grad
            assert torch.allclose(x_g, x_g_hat), f"Gradient check failed for size {size} -- {_plot_diff(x_g, x_g_hat)}"
        except torch.AcceleratorError as e:
            print(f"Size {size} failed due to CUDA error: {e}")
            raise e
        

