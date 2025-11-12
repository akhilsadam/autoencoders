# generate random data for testing cu layers
import torch

sizes=[
    (16, 1, 28, 28),
    (32, 3, 64, 64),
    (4, 3, 128, 128),
]

def random_reduce(err):
    weights = torch.rand_like(err)
    loss = weights * (err) ** 2
    return loss.mean()

def get_random_data(batch_size: int, channels: int, height: int, width: int) -> torch.Tensor:
    torch.manual_seed(42)
    return torch.randn(batch_size, channels, height, width, device="cuda")

def _check(true_func, cu_func):
    for size in sizes:
        x = get_random_data(*size)
        x.requires_grad_()
        
        x_2 = x.clone().detach()
        x_2.requires_grad_()
        
        y = true_func(x)
        y_hat = cu_func(x_2)
        assert torch.allclose(y_hat, y), f"Forward check failed for size {size}"
        
        random_reduce(y).backward()
        x_g = x.grad
        
        random_reduce(y_hat).backward()
        x_g_hat = x_2.grad
        assert torch.allclose(x_g, x_g_hat), f"Gradient check failed for size {size}"
        
        

